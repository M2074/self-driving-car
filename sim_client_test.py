"""Pretends to be the Udacity simulator: connects to the server on port 4567 and
checks normal driving, then that a car which stops moving gets reversed out and
handed back to the model.
Run: python sim_client_test.py [camera_frame.jpg]   (server must be running; PORT=... if not on 4567)"""
import base64, os, sys, time
import socketio

steer = {}
sio = socketio.Client()
sio.on("steer", lambda data: steer.update(data))
sio.connect(f"http://localhost:{os.environ.get('PORT', 4567)}", transports=["websocket"])
time.sleep(1)

with open(sys.argv[1] if len(sys.argv) > 1 else "test.jpg", "rb") as f:
    img = base64.b64encode(f.read()).decode()


def drive(speed):
    """Send one telemetry frame, return the (steering, throttle) the server answers with."""
    steer.clear()  # also drops the (0,0) steer the server sends on connect
    sio.emit("telemetry", {"speed": str(speed), "steering_angle": "0", "throttle": "0", "image": img})
    for _ in range(3000):  # model's first predict is slow (graph tracing)
        if "steering_angle" in steer:
            return float(steer["steering_angle"]), float(steer["throttle"])
        time.sleep(0.02)
    raise AssertionError("no steer response from server")


def drive_until(speed, done, seconds):
    """Keep sending frames at this speed until done(throttle); return the last answer."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        s, t = drive(speed)
        if done(t):
            return s, t
    raise AssertionError(f"condition not met within {seconds}s, last throttle {t}")


try:
    # Moving normally: the model steers, throttle follows speed
    s, t = drive(5.0)
    assert -1.5 <= s <= 1.5, steer
    assert abs(t - 0.5) < 1e-6, f"throttle should be 1 - speed/maxSpeed = 0.5, got {t}"
    print(f"PASS driving   steering={s:.3f} throttle={t:.2f}")

    # Throttle applied, speed stays 0: the agent must reverse (a slow LLM or its veto only delays it)...
    start = time.time()
    s, t = drive_until(0.0, lambda t: t < 0, 400)
    print(f"PASS recovery  reversing after {time.time() - start:.1f}s  steering={s:+.1f} throttle={t:.2f}")

    # ...and then give control back to the model
    s, t = drive_until(4.0, lambda t: t > 0, 30)
    assert abs(t - 0.6) < 1e-6, f"throttle should be back to 1 - 4/10 = 0.6, got {t}"
    print(f"PASS handback  steering={s:.3f} throttle={t:.2f}")

    # Stuck again straight away: it must try again, never park the car at zero throttle
    seen = []
    s, t = drive_until(0.0, lambda t: seen.append(t) or t < 0, 400)
    assert 0.0 not in seen, "the agent parked the car instead of retrying"
    print(f"PASS retry     reversing again after {len(seen)} frames, none at zero throttle  steering={s:+.1f}")
finally:
    sio.disconnect()  # otherwise the client's background threads keep a failed run alive
