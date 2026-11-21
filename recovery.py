"""Stuck detection and recovery agent for the autonomous driver.

Telemetry decides that the car is stuck (throttle is applied but it is not moving)
and the car starts backing out at once. Meanwhile a vision LLM looks at the camera
frame; its answer says which side the obstacle is on, which steers the rest of the
manoeuvre, or that the road is clear, which calls it off. Without an LLM (no
LLM_API_KEY, a slow answer, or a failed call) the manoeuvre runs on a neutral plan.

Self-check: python recovery.py
"""
import base64
import json
import os
import re
import threading
import time

from openai import OpenAI

# Tuning knobs, overridable with environment variables (no rebuild needed)
STUCK_SPEED = float(os.environ.get("STUCK_SPEED", 2.5))  # mph; a wedged car still creeps at ~1 mph, cruising is ~9
STUCK_SECONDS = float(os.environ.get("STUCK_SECONDS", 3.0))  # this long under throttle below STUCK_SPEED = stuck
REVERSE_SECONDS = float(os.environ.get("REVERSE_SECONDS", 2.5))  # first attempt; repeats within a minute back up longer
FORWARD_SECONDS = 2.0  # each forward push of the multi-point turn
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", 20))  # seconds before an unanswered LLM request is dropped
REVERSE_THROTTLE = -1.0  # full reverse; the simulator reverses on negative throttle below 5 mph
REVERSE_STEER = float(os.environ.get("REVERSE_STEER", 0.6))
CREEP_SPEED = 0.05  # mph; a blocked car reads 0.00, one that is slowly backing out reads more
SETTLE_SECONDS = 1.0  # a stroke gets this long to get the car going before a dead stop ends it
HINT_STEER = 0.3  # the driving model steering this hard while stuck hints at the side it is caught on
MAX_VETOES = 1  # LLM "not stuck" answers accepted in a row before physics wins

# Any OpenAI-compatible endpoint; the Docker image defaults to OpenRouter
llm = None
if os.environ.get("LLM_API_KEY"):
    llm = OpenAI(base_url=os.environ.get("LLM_BASE_URL"), api_key=os.environ["LLM_API_KEY"],
                 timeout=LLM_TIMEOUT, max_retries=0)

PROMPT = """You are the recovery agent of a self-driving car in a driving simulator.
The image is the car's front camera, with the car's own hood at the bottom edge.
The car has had throttle applied for several seconds and is not moving, so something is holding it.

Decide:
1. stuck: answer true if anything could be holding the car: a wall, rock, embankment, barrier,
   post or curb ahead of it or touching the hood at either side, or a tilted horizon (the car is
   riding up on something). Visible road ahead does not mean the car is free. Answer false only
   if the road is clear and level with nothing near the car, so it is just starting or climbing.
2. obstacle: on which side of the car is the thing holding it: "left", "ahead" or "right"?

Reply with JSON only:
{"stuck": true or false, "obstacle": "left" or "ahead" or "right", "reason": "<at most 12 words>"}"""


def parse_verdict(text):
    """Pull the verdict out of an LLM reply; free models like to wrap JSON in prose or code fences."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in the reply {text[:80]!r}")
    data = json.loads(match.group())
    return {
        "stuck": str(data["stuck"]).lower() == "true",
        "obstacle": data.get("obstacle") if data.get("obstacle") in ("left", "right") else "ahead",
        "reason": str(data.get("reason", ""))[:80],
    }


def ask_llm(image_b64):
    """Have the vision LLM judge the camera frame. None if no LLM is configured or the call fails."""
    if llm is None:
        return None
    try:
        reply = llm.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
            ]}],
            max_tokens=4000,  # reasoning models spend most of this thinking before the JSON
            temperature=0,
        )
        return parse_verdict(reply.choices[0].message.content or "")
    except Exception as e:
        print("🤖 Agent: vision model gave no usable answer:", e)
        return None


def ask_in_background(judge, image_b64):
    """Run the judge off the control loop; the returned dict gains a "verdict" key once it answers.

    A plain thread and a dict the control loop only reads: nothing the loop could ever wait on.
    """
    box = {}

    def run():
        try:
            box["verdict"] = judge(image_b64)
        except Exception as e:
            print("🤖 Agent: vision model check failed:", e)
            box["verdict"] = None

    threading.Thread(target=run, daemon=True).start()
    return box


def check_llm():
    """Startup check: is the vision LLM reachable from here, before the car needs it?"""
    started = time.time()
    try:
        llm.chat.completions.create(model=os.environ["LLM_MODEL"], max_tokens=20,
                                    messages=[{"role": "user", "content": "Reply with OK"}])
        print(f"🤖 Recovery agent: vision LLM answered a test request in {time.time() - started:.1f}s")
    except Exception as e:
        print(f"🤖 Recovery agent: vision LLM not reachable, recovery will run without it ({e})")


if llm:
    threading.Thread(target=check_llm, daemon=True).start()


def save_frame(image_b64):
    """Keep the camera frame the agent judged, for debugging and demos."""
    os.makedirs("stuck_frames", exist_ok=True)
    path = time.strftime("stuck_frames/%Y%m%d-%H%M%S.jpg")
    with open(path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    return path


class Recovery:
    def __init__(self, judge=ask_llm, clock=time.monotonic):
        self.judge = judge
        self.clock = clock
        self.stuck_since = None  # when the car stopped responding to throttle
        self.advice = None  # holds the vision model's verdict for this recovery once it has answered
        self.vetoes = 0  # LLM "not stuck" answers since the car last moved
        self.move = None  # (steering, throttle) of the manoeuvre step being driven right now
        self.move_started = 0.0
        self.move_seconds = 0.0
        self.moves = []  # (steering, throttle, seconds) still to come in this recovery
        self.reverse_seconds = REVERSE_SECONDS
        self.side = "ahead"  # side of the obstacle the current plan assumes
        self.recoveries = []  # start times of recent recoveries

    def step(self, speed, steering, throttle, image_b64):
        """Return the (steering, throttle) to send: the model's own, or the recovery manoeuvre's."""
        now = self.clock()
        if self.move:
            age = now - self.move_started
            on_time = age < self.move_seconds
            reversing = self.move[1] < 0
            # A car that is creeping back is getting somewhere, so that reverse gets up to three
            # times as long. A stroke that has hit a dead stop is over at once: a hooked car only
            # comes free by rocking back and forth. A forward push also ends once the car rolls
            creeping = CREEP_SPEED < speed <= STUCK_SPEED and age < 3 * self.move_seconds
            stopped = age > SETTLE_SECONDS and speed < CREEP_SPEED
            if not stopped and ((reversing and (on_time or creeping))
                                or (not reversing and on_time and speed <= STUCK_SPEED)):
                if self.advice and "verdict" in self.advice:
                    verdict, self.advice = self.advice["verdict"], None
                    self.take_advice(verdict, now)
                if self.move:
                    return self.move
            elif speed <= STUCK_SPEED and self.moves:
                return self.next_move(now)
            else:
                print(f"🤖 Agent: {'rolling again' if speed > STUCK_SPEED else 'still not free'} "
                      f"at {speed:.1f} mph, back to the model")
                self.move, self.moves = None, []
        if speed > STUCK_SPEED or throttle <= 0.1:
            # Moving: an answer still on its way no longer matters
            self.stuck_since, self.advice, self.vetoes = None, None, 0
            return steering, throttle
        if self.stuck_since is None:
            self.stuck_since = now
        if now - self.stuck_since < STUCK_SECONDS:
            return steering, throttle

        # Throttle applied but no real movement for STUCK_SECONDS. Start backing out now and have
        # the vision model look at the frame meanwhile; it never keeps the car waiting
        self.stuck_since = None
        self.recoveries = [t for t in self.recoveries if now - t < 60] + [now]
        attempt = len(self.recoveries)  # within the last minute; it never parks the car, it keeps trying
        self.reverse_seconds = REVERSE_SECONDS + 0.5 * min(attempt - 1, 2)  # repeats back up a little further
        print(f"🤖 Agent: car not moving for {STUCK_SECONDS:.0f}s, recovery attempt {attempt}; "
              f"vision model is looking at {save_frame(image_b64)}")
        if attempt >= 3:
            # The simulator's reverse is about a third as strong as its forward drive, so a car
            # hooked on a post or nose-down against something can be beyond any manoeuvre
            print("🤖 Agent: third attempt within a minute. If it stays caught here, press Esc in the "
                  "simulator and enter Autonomous Mode again to put the car back on the road")
        self.advice = ask_in_background(self.judge, image_b64)
        # Until the vision model answers: a driving model steering hard one way while stuck is
        # usually pulling away from what it is caught on
        self.plan("left" if steering >= HINT_STEER else "right" if steering <= -HINT_STEER else "ahead", attempt)
        return self.next_move(now)

    def take_advice(self, verdict, now):
        """The vision model has answered while the car is manoeuvring."""
        if verdict is None:
            return  # nothing usable: carry on with the neutral plan
        if not verdict["stuck"] and self.vetoes < MAX_VETOES:
            self.vetoes += 1
            print(f"🤖 Agent: vision model says not stuck ({verdict['reason']}), back to the model")
            self.move, self.moves = None, []
            return
        side = verdict["obstacle"]
        note = "" if verdict["stuck"] else " (says not stuck again, but the car is still not moving)"
        print(f"🤖 Agent: vision model: {verdict['reason']}, obstacle {side}{note}")
        if side != "ahead" and side != self.side:
            self.plan(side, len(self.recoveries))
            self.next_move(now)

    def plan(self, side, attempt):
        """Queue the strokes of a multi-point turn: (steering, throttle, seconds) each."""
        self.side = side
        away = {"left": 1.0, "right": -1.0}.get(side)
        if away:
            # Back out with the wheels towards the obstacle and push forward with them away from
            # it: both walk the nose off it. One gentle reverse, which frees a car with room, then
            # full lock. If that fails the car may be pinned along its side: wheels away, then straight
            steers = [-away * REVERSE_STEER, -away, -away, -away, away, 0.0]
        else:
            # No side known: straight back first, and each repeat attempt starts with a different one
            steers = [0.0, REVERSE_STEER, -REVERSE_STEER]
            steers = steers[(attempt - 1) % 3:] + steers[:(attempt - 1) % 3]
            away = 1.0 if attempt % 2 else -1.0
        self.moves = [move for steer in steers
                      for move in ((steer, REVERSE_THROTTLE, self.reverse_seconds), (away, 1.0, FORWARD_SECONDS))]

    def next_move(self, now):
        """Start the next step of the manoeuvre."""
        steer, throttle, self.move_seconds = self.moves.pop(0)
        self.move, self.move_started = (steer, throttle), now
        print(f"🤖 Agent: {'reversing' if throttle < 0 else 'pushing forward'} {self.move_seconds:.1f}s "
              f"with steering {steer:+.1f}")
        return self.move


def demo():
    import tempfile
    os.chdir(tempfile.mkdtemp())  # stuck_frames/ lands in a throwaway folder
    img = base64.b64encode(b"jpeg").decode()
    model = (0.1, 1.0)  # what the steering model wants
    straight_back = (0.0, REVERSE_THROTTLE)
    swing_left = (REVERSE_STEER, REVERSE_THROTTLE)  # obstacle on the right
    creep = STUCK_SPEED - 0.1  # a wedged car is not at a clean 0 mph
    t = [0.0]

    def agent(judge):
        r = Recovery(judge=judge, clock=lambda: t[0])
        return r, lambda speed: r.step(speed, *model, img)

    def answered(r):
        """Wait for the judge, which runs on its own thread."""
        while "verdict" not in r.advice:
            time.sleep(0.001)

    def get_stuck(r, drive):
        """Sit under throttle until the agent starts backing out; return its move once the vision model has answered."""
        drive(creep)
        t[0] += STUCK_SECONDS + 0.1
        assert drive(creep)[1] == REVERSE_THROTTLE  # backing out at once, not waiting for the answer
        answered(r)
        return drive(creep)

    # Stuck for STUCK_SECONDS: back straight out at once; the vision model's answer then steers it
    r, drive = agent(lambda _: {"stuck": True, "obstacle": "right", "reason": "wall"})
    assert drive(9) == model
    t[0] += 1
    assert drive(creep) == model
    t[0] += STUCK_SECONDS - 0.1
    assert drive(creep) == model and not r.move
    t[0] += 0.2
    assert drive(creep) == straight_back
    answered(r)
    assert drive(creep) == swing_left
    t[0] += REVERSE_SECONDS - 0.1
    assert drive(4) == swing_left
    t[0] += 0.2
    assert drive(4) == model

    # Still stuck every time: it never parks the car, and each repeat backs up a little longer (capped)
    for attempt in (2, 3, 4, 5):
        assert get_stuck(r, drive) == swing_left
        assert r.move_seconds == REVERSE_SECONDS + 0.5 * min(attempt - 1, 2)
        t[0] += r.move_seconds + 0.1
        assert drive(9) == model  # rolling after the reverse: straight back to the model

    # Boxed in: backing out does not get the car rolling, so it works through a multi-point turn
    r, drive = agent(lambda _: {"stuck": True, "obstacle": "left", "reason": "post"})
    back_toward, back_full = (-REVERSE_STEER, REVERSE_THROTTLE), (-1.0, REVERSE_THROTTLE)
    push = (1.0, 1.0)  # forward, full lock away from the obstacle
    assert get_stuck(r, drive) == back_toward
    t[0] += REVERSE_SECONDS + 0.1
    assert drive(0.2) == back_toward  # creeping back: keep at it rather than give up on it
    t[0] += 2 * REVERSE_SECONDS
    assert drive(0.2) == push  # the creep got nowhere in three times the time
    t[0] += SETTLE_SECONDS - 0.1
    assert drive(0.0) == push  # not moving yet, but a stroke gets a moment to get going
    t[0] += 0.2
    assert drive(0.0) == back_full  # dead stop: rock the other way at once, now at full lock
    t[0] += SETTLE_SECONDS + 0.1
    assert drive(0.2) == back_full
    assert drive(0.0) == push  # hit the stop again
    t[0] += 0.5
    assert drive(3) == model and not r.moves  # rolling on the forward push: back to the model

    # Before the vision model answers, a driving model steering hard right is taken as "obstacle
    # on the left"; an answer naming the same side changes nothing, the other side restarts the plan
    for verdict_side, expected in (("left", back_toward), ("right", swing_left)):
        r = Recovery(judge=lambda _, s=verdict_side: {"stuck": True, "obstacle": s, "reason": "post"}, clock=lambda: t[0])
        hard_right = lambda speed: r.step(speed, 0.6, 1.0, img)
        hard_right(creep)
        t[0] += STUCK_SECONDS + 0.1
        assert hard_right(creep) == back_toward
        started = r.move_started
        t[0] += 0.5
        answered(r)
        assert hard_right(creep) == expected and (r.move_started == started) == (verdict_side == "left")

    # Vision model vetoes ("just climbing"): the reverse is called off, but physics wins after MAX_VETOES
    r, drive = agent(lambda _: {"stuck": False, "obstacle": "left", "reason": "clear road"})
    for _ in range(MAX_VETOES):
        assert get_stuck(r, drive) == model
    assert get_stuck(r, drive) == (-REVERSE_STEER, REVERSE_THROTTLE)

    # A veto survives creeping, and is forgotten once the car really moves
    r, drive = agent(lambda _: {"stuck": False, "obstacle": "ahead", "reason": "clear road"})
    get_stuck(r, drive)
    assert r.vetoes == 1
    drive(creep)
    assert r.vetoes == 1
    drive(9)
    assert r.vetoes == 0

    # The answer comes after the car is already rolling: not acted on
    r, drive = agent(lambda _: {"stuck": True, "obstacle": "left", "reason": "post"})
    drive(creep)
    t[0] += STUCK_SECONDS + 0.1
    assert drive(creep) == straight_back
    t[0] += REVERSE_SECONDS + 0.1
    answered(r)
    assert drive(9) == model and not r.move

    # Vision model never answers: the neutral plan runs on its own, no waiting
    gate = threading.Event()
    r, drive = agent(lambda _: gate.wait())
    drive(creep)
    t[0] += STUCK_SECONDS + 0.1
    assert drive(creep) == straight_back
    t[0] += SETTLE_SECONDS + 0.1
    assert drive(0.0) == push
    t[0] += SETTLE_SECONDS + 0.1
    assert drive(0.0) == (REVERSE_STEER, REVERSE_THROTTLE)
    gate.set()

    # No LLM configured: straight back first, then a different start on each repeat attempt
    r, drive = agent(lambda _: None)
    assert get_stuck(r, drive) == straight_back
    t[0] += r.move_seconds + 0.1
    drive(9)
    assert get_stuck(r, drive) == (REVERSE_STEER, REVERSE_THROTTLE)
    t[0] += r.move_seconds + 0.1
    drive(9)
    assert get_stuck(r, drive) == (-REVERSE_STEER, REVERSE_THROTTLE)

    assert parse_verdict('```json\n{"stuck": true, "obstacle": "left", "reason": "bank on left"}\n```') == \
        {"stuck": True, "obstacle": "left", "reason": "bank on left"}
    assert parse_verdict('Sure! {"stuck": "False", "obstacle": "up"}') == \
        {"stuck": False, "obstacle": "ahead", "reason": ""}
    try:
        parse_verdict("")
        raise AssertionError("an empty reply must be rejected")
    except ValueError:
        pass
    print("recovery self-check PASS")


if __name__ == "__main__":
    demo()
