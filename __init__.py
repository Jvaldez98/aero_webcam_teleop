"""
aero_webcam_teleop
==================
Pure-Python webcam teleoperation pipeline for the TetherIA Aero Hand Open.

Mirrors the ROS2 stack (aero_hand_open + aero_hand_open_teleop) but runs
without any ROS2 dependency, using:

  • OpenCV           — camera capture
  • MediaPipe        — hand landmark detection
  • aero-open-sdk    — serial communication with the hand

Quick start
-----------
  # Mock mode (no hardware needed):
  python -m aero_webcam_teleop.pipeline

  # With hardware:
  python -m aero_webcam_teleop.pipeline --port /dev/ttyACM0
"""

from .mediapipe_mocap import HandMocap, Keypoint3D, MediaPipeMocap
from .mediapipe_retargeting import MediaPipeRetargeting, RetargetingConfig
from .handbridge import AeroHandBridge
from .pipeline import WebcamTeleopPipeline

__all__ = [
    "HandMocap",
    "Keypoint3D",
    "MediaPipeMocap",
    "MediaPipeRetargeting",
    "RetargetingConfig",
    "AeroHandBridge",
    "WebcamTeleopPipeline",
]