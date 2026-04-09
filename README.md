# aero_webcam_teleop

> **Pure-Python webcam teleoperation for the TetherIA Aero Hand Open.**
> No ROS2 required — runs anywhere Python 3.10+ works.

---

## Overview

This package mirrors the ROS2 `aero_hand_open_teleop` + `aero_hand_open` stack
from the [TetherIA/aero-hand-open](https://github.com/TetherIA/aero-hand-open)
repository, but as a self-contained Python module:

| ROS2 node / package | This file |
|---|---|
| `aero_hand_node` (hardware bridge) | `hand_bridge.py` — `AeroHandBridge` |
| `mediapipe_mocap` | `mediapipe_mocap.py` — `MediaPipeMocap` |
| `mediapipe_retargeting` | `mediapipe_retargeting.py` — `MediaPipeRetargeting` |
| launch file / orchestrator | `pipeline.py` — `WebcamTeleopPipeline` |

### Architecture

```
Webcam
  │
  ▼
MediaPipeMocap          (mediapipe_mocap.py)
  │  HandMocap (21 keypoints @ ~30 Hz)
  ▼
MediaPipeRetargeting    (mediapipe_retargeting.py)
  │  16 joint angles (degrees), EMA-smoothed @ ~30 Hz
  ▼
[control thread @ 20 Hz]
  │
  ▼
AeroHandBridge          (hand_bridge.py)
  │  aero_open_sdk.AeroHand.set_joint_positions()
  ▼
Aero Hand Open (ESP32-S3 @ 921600 baud)
```

---

## Installation

```bash
# Clone the repo (or copy this folder)
git clone <this-repo>
cd aero_webcam_teleop

# Install dependencies
pip install opencv-python mediapipe numpy

# Install the TetherIA SDK (only needed for real hardware)
pip install aero-open-sdk     # or follow SDK setup from docs.tetheria.ai

# Install this package
pip install -e .

#create a virtual enviorment for python 3.11
cd C:\Users\jorge\OneDrive\Documents\Tetherhand
py -3.11 -m venv venv_aero
#can create your own virtual env name (venv_aero)
venv_aero\Scripts\activate
#important!!! when making the virtual env make sure you are outside the project folder for
#ex cd C:\Users\jorge\OneDrive\Documents\Tetherhand\aero_teleop_webcam <- project folder,
#must be cd C:\Users\jorge\OneDrive\Documents\Tetherhand to succesfully create a venv.
```

---

## Usage

### Mock mode (no hardware — great for development)

```bash
python -m aero_webcam_teleop.pipeline
```

Commands are printed to the console; the annotated webcam window shows live
joint-angle bars for each finger.

### Real hardware

```bash
# Linux
python -m aero_webcam_teleop.pipeline --port /dev/ttyACM0

# Windows
python -m aero_webcam_teleop.pipeline --port COM3

# Left hand
python -m aero_webcam_teleop.pipeline --port /dev/ttyACM0 --side left

# All options
python -m aero_webcam_teleop.pipeline --help
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--port` | None | Serial port (omit = mock mode) |
| `--baudrate` | 921600 | Serial baud rate |
| `--side` | right | `right` or `left` |
| `--camera` | 0 | Webcam device index |
| `--fps` | 30 | Capture rate (Hz) |
| `--control-hz` | 20 | Command send rate to hand (Hz) |
| `--smoothing` | 0.35 | EMA alpha [0,1]; lower = smoother |
| `--verbose` | off | Print joint angles each frame |

### Keyboard shortcuts (click the video window first)

| Key | Action |
|---|---|
| `c` | **Calibrate** — arm calibration, then show open flat palm |
| `h` | **Home** — run on-board homing routine (~1-2 min) |
| `o` | **Open** hand (all joints to 0°) |
| `f` | **Fist** (close all fingers) |
| `q` / ESC | Quit |

---

## Calibration

For best tracking accuracy, calibrate once per session:

1. Run the pipeline.
2. Hold your hand **open and flat** in front of the camera.
3. Press **`c`** in the video window.

This records the open-palm pose as the zero reference and removes systematic
offsets from the retargeting.

---

## Python API

```python
from aero_webcam_teleop import WebcamTeleopPipeline

pipeline = WebcamTeleopPipeline(
    port="/dev/ttyACM0",   # or None for mock
    side="right",
    smoothing_alpha=0.35,
)
pipeline.run()             # blocks; press q to exit
```

Custom callback example (build your own pipeline):

```python
from aero_webcam_teleop import (
    MediaPipeMocap, MediaPipeRetargeting, AeroHandBridge, HandMocap
)

retargeting = MediaPipeRetargeting(smoothing_alpha=0.4)
bridge = AeroHandBridge(port="/dev/ttyACM0")

def on_mocap(mocap: HandMocap):
    angles = retargeting.retarget(mocap)
    bridge.send_joint_positions(angles)

MediaPipeMocap(hand_side="right", on_mocap=on_mocap).start()
```

---

## Joint mapping

All 16 joints match the SDK convention exactly (degrees, 0 = fully extended):

| # | Joint | Range (°) |
|---|---|---|
| 0 | thumb_cmc_abd | 0–100 |
| 1 | thumb_cmc_flex | 0–55 |
| 2 | thumb_mcp | 0–90 |
| 3 | thumb_ip | 0–90 |
| 4–6 | index_mcp/pip/dip | 0–90 |
| 7–9 | middle_mcp/pip/dip | 0–90 |
| 10–12 | ring_mcp/pip/dip | 0–90 |
| 13–15 | pinky_mcp/pip/dip | 0–90 |

---

## Troubleshooting

**No hand detected** — ensure adequate lighting and that the hand occupies
a good portion of the frame.

**Laggy / jittery control** — lower `--fps` and/or reduce `--smoothing`
(try 0.2–0.3).

**Permission denied on serial port (Linux)**

```bash
sudo usermod -a -G dialout $USER
# then log out and back in
```

**`aero_open_sdk` not found** — the pipeline automatically falls back to
mock mode. Install the SDK from [docs.tetheria.ai](https://docs.tetheria.ai).

---

## License

This package is released under **Apache 2.0**, consistent with the TetherIA
SDK and firmware license. The Aero Hand Open hardware design files are
CC BY-NC-SA 4.0.
