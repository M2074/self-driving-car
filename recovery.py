"""Stuck detection and recovery agent for the autonomous driver.

Telemetry raises the suspicion (throttle is applied but the car is not moving),
then a vision LLM looks at the camera frame to confirm it and say where the road
is. The car reverses out and the steering model takes over again. Without an LLM
(no LLM_API_KEY, no answer in time, or the call fails) it reverses straight back.

Self-check: python recovery.py
"""
import base64
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# Tuning knobs, overridable with environment variables (no rebuild needed)
STUCK_SPEED = float(os.environ.get("STUCK_SPEED", 2.5))  # mph; a wedged car still creeps at ~1 mph, cruising is ~9
STUCK_SECONDS = float(os.environ.get("STUCK_SECONDS", 3.0))  # this long under throttle below STUCK_SPEED = suspected stuck
REVERSE_SECONDS = float(os.environ.get("REVERSE_SECONDS", 2.5))  # first attempt; repeats within a minute back up longer
FORWARD_SECONDS = 2.0  # each forward push of the multi-point turn
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", 20))  # seconds to wait for a verdict before recovering without one
REVERSE_THROTTLE = -1.0  # full reverse; the simulator reverses on negative throttle below 5 mph
REVERSE_STEER = float(os.environ.get("REVERSE_STEER", 0.6))
CREEP_SPEED = 0.05  # mph; a blocked car reads 0.00, one that is slowly backing out reads more
MAX_VETOES = 1  # LLM "not stuck" answers accepted in a row before physics wins

# Any OpenAI-compatible endpoint; the Docker image defaults to OpenRouter
llm = None
if os.environ.get("LLM_API_KEY"):
    llm = OpenAI(base_url=os.environ.get("LLM_BASE_URL"), api_key=os.environ["LLM_API_KEY"],
                 timeout=LLM_TIMEOUT, max_retries=0)

# The LLM runs off the control loop: a blocked server stops answering Socket.IO
# pings and the simulator drops the connection. Spare workers so a call abandoned
# at the deadline or on reconnect does not hold up the next one.
executor = ThreadPoolExecutor(max_workers=4)

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
        print("🤖 Agent: LLM gave no usable answer:", e)
        return None


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
        self.pending = None  # verdict the LLM is still working on
        self.pending_since = 0.0
        self.vetoes = 0  # LLM "not stuck" answers since the car last moved
        self.move = None  # (steering, throttle) of the manoeuvre step being driven right now
        self.move_started = 0.0
        self.move_seconds = 0.0
        self.moves = []  # (steering, throttle, seconds) still to come in this recovery
        self.recoveries = []  # start times of recent recoveries

    def step(self, speed, steering, throttle, image_b64):
        """Return the (steering, throttle) to send: the model's own, or the recovery manoeuvre's."""
        now = self.clock()
        if self.move:
            age = now - self.move_started
            if self.move[1] < 0:
                # Backing out. A car that is creeping back is getting somewhere, so it gets up to
                # three times as long; one at a dead stop does not
                creeping = CREEP_SPEED < speed <= STUCK_SPEED and age < 3 * self.move_seconds
                if age < self.move_seconds or creeping:
                    return self.move
            elif speed <= STUCK_SPEED and age < self.move_seconds:
                return self.move  # pushing forward, not rolling yet
            if speed <= STUCK_SPEED and self.moves:
                return self.next_move(now)
            print(f"🤖 Agent: {'rolling again' if speed > STUCK_SPEED else 'still not free'} "
                  f"at {speed:.1f} mph, back to the model")
            self.move, self.moves = None, []
        if speed > STUCK_SPEED or throttle <= 0.1:
            # Moving again: a verdict still on its way no longer matters
            self.stuck_since, self.pending, self.vetoes = None, None, 0
            return steering, throttle
        if self.pending:
            # Keep pushing while the LLM looks at the frame, but only until the deadline
            if not self.pending.done() and now - self.pending_since < LLM_TIMEOUT:
                return steering, throttle
            verdict = self.pending.result() if self.pending.done() else None
            if not self.pending.done():
                print(f"🤖 Agent: no LLM answer within {LLM_TIMEOUT:.0f}s")
            self.pending = None
            return self.act_on(verdict, steering, throttle)
        if self.stuck_since is None:
            self.stuck_since = now
        if now - self.stuck_since < STUCK_SECONDS:
            return steering, throttle

        # Throttle applied but no real movement for STUCK_SECONDS: have the frame judged
        self.stuck_since = None
        print(f"🤖 Agent: car not moving for {STUCK_SECONDS:.0f}s, judging frame {save_frame(image_b64)}")
        self.pending, self.pending_since = executor.submit(self.judge, image_b64), now
        return steering, throttle

    def act_on(self, verdict, steering, throttle):
        if verdict and not verdict["stuck"] and self.vetoes < MAX_VETOES:
            self.vetoes += 1
            print(f"🤖 Agent: LLM says not stuck ({verdict['reason']}), keep driving")
            return steering, throttle

        now = self.clock()
        self.recoveries = [t for t in self.recoveries if now - t < 60] + [now]
        attempt = len(self.recoveries)  # within the last minute; it never parks the car, it keeps trying

        side = verdict["obstacle"] if verdict else "ahead"
        reason = verdict["reason"] if verdict else "no LLM verdict"
        if verdict and not verdict["stuck"]:
            reason = "still not moving, overruling the LLM"
        away = {"left": 1.0, "right": -1.0}.get(side)  # full lock away from the obstacle
        if away:
            # Wheels towards the obstacle swings the nose away from it. A car pinned along its
            # side cannot back out that way, so later tries turn them away, then go straight
            steers = [-away * REVERSE_STEER, away * REVERSE_STEER, 0.0]
        else:
            # No side known: straight back first, and each repeat attempt starts with a different one
            steers = [0.0, REVERSE_STEER, -REVERSE_STEER]
            steers = steers[(attempt - 1) % 3:] + steers[:(attempt - 1) % 3]
            away = 1.0 if attempt % 2 else -1.0
        seconds = REVERSE_SECONDS + 0.5 * min(attempt - 1, 2)  # repeats back up a little further
        # A multi-point turn: back out, and if that has not got the car rolling, push forward
        # with the wheels away from the obstacle, then back out another way
        self.moves = [move for steer in steers
                      for move in ((steer, REVERSE_THROTTLE, seconds), (away, 1.0, FORWARD_SECONDS))]
        print(f"🤖 Agent: recovering, attempt {attempt} ({reason}), obstacle {side}")
        return self.next_move(now)

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
    swing_left = (REVERSE_STEER, REVERSE_THROTTLE)  # obstacle on the right
    creep = STUCK_SPEED - 0.1  # a wedged car is not at a clean 0 mph
    t = [0.0]

    def agent(judge):
        r = Recovery(judge=judge, clock=lambda: t[0])
        return r, lambda speed: r.step(speed, *model, img)

    def get_stuck(r, drive):
        """Creep under throttle until the frame is sent off; return the answer once the verdict is in."""
        drive(creep)
        t[0] += STUCK_SECONDS + 0.1
        assert drive(creep) == model  # still pushing while the LLM thinks
        r.pending.result()
        return drive(creep)

    # LLM confirms: wait STUCK_SECONDS, keep pushing while it thinks, reverse away from the obstacle, hand back
    r, drive = agent(lambda _: {"stuck": True, "obstacle": "right", "reason": "wall"})
    assert drive(9) == model
    t[0] += 1
    assert drive(creep) == model
    t[0] += STUCK_SECONDS - 0.1
    assert drive(creep) == model and not r.pending
    t[0] += 0.2
    assert drive(creep) == model and r.pending
    r.pending.result()
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
    r, drive = agent(lambda _: {"stuck": True, "obstacle": "left", "reason": "barrier"})
    back_toward, back_away = (-REVERSE_STEER, REVERSE_THROTTLE), (REVERSE_STEER, REVERSE_THROTTLE)
    push = (1.0, 1.0)  # forward, full lock away from the obstacle
    assert get_stuck(r, drive) == back_toward
    t[0] += REVERSE_SECONDS + 0.1
    assert drive(0.2) == back_toward  # creeping back: keep at it rather than give up on it
    t[0] += 2 * REVERSE_SECONDS
    assert drive(0.2) == push  # the creep got nowhere in three times the time
    t[0] += FORWARD_SECONDS + 0.1
    assert drive(0.2) == back_away
    t[0] += REVERSE_SECONDS + 0.1
    assert drive(0.0) == push  # dead stop: no extra time for that one
    t[0] += 0.5
    assert drive(3) == model and not r.moves  # rolling on the forward push: back to the model

    # LLM vetoes ("just climbing"): keep driving, but physics wins after MAX_VETOES
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

    # The car frees itself while the LLM is thinking: the verdict is dropped
    r, drive = agent(lambda _: {"stuck": True, "obstacle": "right", "reason": "wall"})
    drive(creep)
    t[0] += STUCK_SECONDS + 0.1
    drive(creep)
    assert r.pending
    assert drive(9) == model and not r.pending

    # LLM never answers: straight reverse at the deadline
    gate = threading.Event()
    r, drive = agent(lambda _: gate.wait())
    drive(creep)
    t[0] += STUCK_SECONDS + 0.1
    assert drive(creep) == model
    t[0] += LLM_TIMEOUT - 0.1
    assert drive(creep) == model
    t[0] += 0.2
    assert drive(creep) == (0.0, REVERSE_THROTTLE)
    gate.set()

    # No LLM configured: straight reverse first, then alternate sides while it stays stuck
    r, drive = agent(lambda _: None)
    assert get_stuck(r, drive) == (0.0, REVERSE_THROTTLE)
    t[0] += r.move_seconds + 0.1
    drive(9)
    assert get_stuck(r, drive) == (REVERSE_STEER, REVERSE_THROTTLE)
    t[0] += r.move_seconds + 0.1
    drive(9)
    assert get_stuck(r, drive) == (-REVERSE_STEER, REVERSE_THROTTLE)

    assert parse_verdict('```json\n{"stuck": true, "obstacle": "left", "reason": "bank on left"}\n```') == \
        {"stuck": True, "obstacle": "left", "reason": "bank on left"}
    assert parse_verdict('Sure! {"stuck": "False", "obstacle": "up"}') == {"stuck": False, "obstacle": "ahead", "reason": ""}
    try:
        parse_verdict("")
        raise AssertionError("an empty reply must be rejected")
    except ValueError:
        pass
    print("recovery self-check PASS")


if __name__ == "__main__":
    demo()
