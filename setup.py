"""
setup.py — aero_webcam_teleop
"""
from setuptools import setup, find_packages

setup(
    name="aero_webcam_teleop",
    version="0.1.0",
    description=(
        "Webcam hand-tracking teleoperation pipeline for the "
        "TetherIA Aero Hand Open (no ROS2 required)"
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.8",
        "mediapipe>=0.10",
        "numpy>=1.24",
        # aero-open-sdk is optional (mock mode runs without it)
        # "aero-open-sdk",
    ],
    entry_points={
        "console_scripts": [
            "aero-webcam-teleop=aero_webcam_teleop.pipeline:main",
        ]
    },
)