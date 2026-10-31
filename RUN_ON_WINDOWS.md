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

## Troubleshooting
- **Car doesn't move:** make sure step 2's window shows `Simulator connected` after you enter Autonomous Mode. If not, check nothing else uses port 4567.
- **Docker errors on load/run:** Docker Desktop must be running (whale icon in system tray), with Linux containers mode (default).
- **Windows SmartScreen blocks the simulator:** click "More info" → "Run anyway" (it's the standard open-source Udacity simulator).

To stop: close the simulator, press Ctrl+C in PowerShell.
