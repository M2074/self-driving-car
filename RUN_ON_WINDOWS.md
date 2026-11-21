# Self-Driving Car Demo — How to Run (Windows)

You received two things: `self-driving-car-image.tar.gz` (the AI driver, as a Docker image) and this file. The car simulator itself is a free download (step 3).

## Requirements
- Windows 10/11, 64-bit
- 8 GB+ RAM (16 GB is plenty), ~5 GB free disk
- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) installed and running

## Steps

### 1. Load the image (one time)
Open PowerShell in the folder containing the file:
```powershell
docker load -i self-driving-car-image.tar.gz
```

### 2. Start the AI driver
```powershell
docker run --rm -p 127.0.0.1:4567:4567 adityyaa/self-driving-car:latest
```
Wait until it prints `✅ Model loaded successfully!` and `wsgi starting up`. Leave this window open.

### 3. Get the simulator (one time)
Download the Udacity simulator for Windows (~107 MB):
https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip

Unzip it anywhere. No installation needed.

### 4. Drive
1. Run `beta_simulator.exe` from the unzipped folder
2. Pick a resolution (800×600, "Fastest" quality is fine) → Play
3. Choose the first track → **Autonomous Mode**

The car connects to the AI on port 4567 and drives itself. The PowerShell window shows live steering values.

## Optional: AI stuck recovery

**Use the Version 2 simulator for this** (`beta_simulator.exe` from the download in step 3). The older Version 1 build (folder `simulator-windows-64`, file `Default Windows desktop 64-bit.exe`, dark mountain track) does not respond to reverse, so the car cannot back out of anything in it.

The `:agent` version of the driver gets itself unstuck. If the car has throttle but has not really moved for 3 seconds, it reverses out and lets the model drive again. With a free OpenRouter key, a vision AI looks at the camera frame first to confirm the car is stuck and to see which side the obstacle is on, so the car backs out away from it.

1. Get a free API key:
   - Open https://openrouter.ai/keys and sign in (Google, GitHub or email)
   - Click **Create Key**, give it any name, and copy the key. It starts with `sk-or-v1-` and is shown only once
   - Free models need no payment details
2. Start the AI driver with the key in place of `sk-or-your-key`:
```powershell
docker run --rm -p 127.0.0.1:4567:4567 -e LLM_API_KEY=sk-or-your-key adityyaa/self-driving-car:agent
```

It prints `🤖 Recovery agent: vision LLM openrouter/free` at startup and `🤖 Agent: ...` lines whenever it steps in. Without the key it still recovers, by reversing straight back.

- Ran it before? Get the newest version first: `docker pull adityyaa/self-driving-car:agent` (Docker never re-downloads a version it already has)
- Keep the frames it judged: add `-v ${PWD}/stuck_frames:/app/stuck_frames`
- Use one specific model instead of whichever free one is available: add `-e LLM_MODEL=google/gemma-4-31b-it:free`
- Car runs wide on bends and hits posts or barriers: slow it down with `-e MAX_SPEED=7` (default 10 mph)
- If the agent reports that reverse does nothing, you are on the Version 1 simulator: switch to `beta_simulator.exe`. To put the car back on the road right away, press Esc and enter Autonomous Mode again
- Tune it: `-e STUCK_SECONDS=3 -e REVERSE_SECONDS=4.5` (seconds of full-throttle reverse per try)

## Troubleshooting
- **Car doesn't move:** make sure step 2's window shows `Simulator connected` after you enter Autonomous Mode. If not, check nothing else uses port 4567.
- **Docker errors on load/run:** Docker Desktop must be running (whale icon in system tray), with Linux containers mode (default).
- **Windows SmartScreen blocks the simulator:** click "More info" → "Run anyway" (it's the standard open-source Udacity simulator).

To stop: close the simulator, press Ctrl+C in PowerShell.
