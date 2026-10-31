"""Pretends to be the Udacity simulator: connects to the server on port 4567,
sends one telemetry frame, and asserts a steer command comes back.
Run: python sim_client_test.py   (server must be running)"""
import base64, sys, time
import socketio

steer = {}
sio = socketio.Client()
sio.on("steer", lambda data: steer.update(data))
sio.connect("http://localhost:4567", transports=["websocket"])
time.sleep(1)
steer.clear()  # drop the (0,0) steer the server sends on connect

with open("test.jpg", "rb") as f:
    img = base64.b64encode(f.read()).decode()
sio.emit("telemetry", {"speed": "5.0", "steering_angle": "0", "throttle": "0", "image": img})

for _ in range(300):  # model's first predict is slow (graph tracing)
    if "steering_angle" in steer:
        break
    time.sleep(0.2)
sio.disconnect()

assert "steering_angle" in steer, "no steer response from server"
s, t = float(steer["steering_angle"]), float(steer["throttle"])
assert -1.5 <= s <= 1.5, steer
assert abs(t - 0.5) < 1e-6, f"throttle should be 1 - speed/maxSpeed = 0.5, got {t}"
print(f"PASS steering={s:.3f} throttle={t:.2f}")
