"""
mediapipe_mocap.py
==================
Mirrors the ROS2 `mediapipe_mocap` node.

Captures frames from either:
  - Intel RealSense D405 depth camera (preferred, auto-detected)
  - Standard webcam fallback (if RealSense is not connected)

Runs MediaPipe Hands and emits a structured HandMocap dataclass containing
21 3-D keypoints (x, y, z) plus a handedness label.

When using the D405, the z coordinate is replaced with true metric depth
(in meters) from the depth sensor, instead of MediaPipe's estimated relative
depth. This gives significantly more accurate 3D hand pose.

Usage (standalone test):
    python -m aero_webcam_teleop.mediapipe_mocap
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
try:
    _mp_hands   = mp.solutions.hands
    _mp_drawing = mp.solutions.drawing_utils
    _mp_styles  = mp.solutions.drawing_styles
except AttributeError:
    from mediapipe.python.solutions import hands         as _mp_hands
    from mediapipe.python.solutions import drawing_utils as _mp_drawing
    from mediapipe.python.solutions import drawing_styles as _mp_styles

# RealSense — optional, gracefully skipped if not installed
try:
    import pyrealsense2 as rs
    _REALSENSE_AVAILABLE = True
except ImportError:
    _REALSENSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Keypoint3D:
    x: float = 0.0   # normalised [0,1] (image width)
    y: float = 0.0   # normalised [0,1] (image height)
    z: float = 0.0   # metric depth in meters (D405) or relative (webcam)


@dataclass
class HandMocap:
    """21-keypoint hand pose from MediaPipe."""
    side: str = "right"
    keypoints: List[Keypoint3D] = field(default_factory=lambda: [Keypoint3D()] * 21)
    timestamp: float = 0.0
    depth_source: str = "webcam"   # "realsense" | "webcam"


LM = _mp_hands.HandLandmark


# ---------------------------------------------------------------------------
# Camera backends
# ---------------------------------------------------------------------------

class _WebcamBackend:
    """Standard OpenCV webcam capture."""

    def __init__(self, camera_index: int):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        self.source_label = "WEBCAM"

    def read(self) -> Tuple[bool, np.ndarray, Optional[np.ndarray]]:
        """Returns (success, rgb_frame, depth_frame_or_None)."""
        ret, frame = self.cap.read()
        if not ret:
            return False, np.zeros((480, 640, 3), dtype=np.uint8), None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True, rgb, None

    def get_depth_at(self, x_norm: float, y_norm: float, frame_shape: tuple) -> float:
        """Webcam has no depth — returns 0."""
        return 0.0

    def release(self):
        self.cap.release()


class _RealSenseBackend:
    """Intel RealSense D405 capture with aligned depth."""

    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        # D405 optimal close-range settings
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)

        # Align depth to color frame
        self.align = rs.align(rs.stream.color)

        # Depth scale (converts raw units -> meters)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self._last_depth: Optional[np.ndarray] = None
        self.source_label = "REALSENSE D405"

    def read(self) -> Tuple[bool, np.ndarray, Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return False, np.zeros((480, 640, 3), dtype=np.uint8), None

        bgr = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        self._last_depth = depth
        return True, rgb, depth

    def get_depth_at(self, x_norm: float, y_norm: float, frame_shape: tuple) -> float:
        """Look up metric depth (meters) at a normalised (x, y) position."""
        if self._last_depth is None:
            return 0.0
        h, w = frame_shape[:2]
        px = int(np.clip(x_norm * w, 0, w - 1))
        py = int(np.clip(y_norm * h, 0, h - 1))
        # 5x5 median patch to reduce noise
        patch = self._last_depth[
            max(0, py - 2):py + 3,
            max(0, px - 2):px + 3
        ]
        nonzero = patch[patch > 0]
        return float(np.median(nonzero)) if len(nonzero) > 0 else 0.0

    def release(self):
        self.pipeline.stop()


def _init_camera(camera_index: int, force_webcam: bool = False):
    """
    Try RealSense first; fall back to webcam automatically.
    Returns (backend, used_realsense: bool).
    """
    if not force_webcam and _REALSENSE_AVAILABLE:
        try:
            backend = _RealSenseBackend()
            print("[MediaPipeMocap] RealSense D405 detected — using depth camera.")
            return backend, True
        except Exception as e:
            print(f"[MediaPipeMocap] RealSense not available ({e}). Falling back to webcam.")

    backend = _WebcamBackend(camera_index)
    print(f"[MediaPipeMocap] Using webcam (index {camera_index}).")
    return backend, False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MediaPipeMocap:
    """
    Wraps MediaPipe Hands and delivers HandMocap callbacks at ~target_fps.

    Automatically uses RealSense D405 if connected, otherwise falls back
    to the standard webcam.

    Parameters
    ----------
    camera_index    : CV2 device index used only if RealSense is unavailable
    target_fps      : approximate capture rate (Hz)
    hand_side       : filter results to "right" or "left" (None = first detected)
    on_mocap        : callback(HandMocap) called each frame a hand is detected
    on_frame        : optional callback(annotated_bgr_frame) for display
    show_window     : whether to call cv2.imshow internally
    force_webcam    : set True to skip RealSense even if connected
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: float = 30.0,
        hand_side: Optional[str] = "right",
        on_mocap: Optional[Callable[[HandMocap], None]] = None,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
        show_window: bool = True,
        force_webcam: bool = False,
    ):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.hand_side = hand_side
        self.on_mocap = on_mocap
        self.on_frame = on_frame
        self.show_window = show_window
        self.force_webcam = force_webcam
        self._running = False

    def start(self):
        """Blocking capture loop. Press 'q' or ESC to exit."""
        self._running = True

        camera, using_realsense = _init_camera(self.camera_index, self.force_webcam)
        depth_source = "realsense" if using_realsense else "webcam"
        interval = 1.0 / self.target_fps

        with _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            while self._running:
                t0 = time.monotonic()

                ret, rgb, depth = camera.read()
                if not ret:
                    continue

                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                # Convert back to BGR for OpenCV display
                annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_lm, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                    ):
                        mp_label = handedness.classification[0].label
                        physical_side = "left" if mp_label == "Right" else "right"

                        if self.hand_side is not None and physical_side != self.hand_side:
                            continue

                        # Build keypoints — use real depth for z if available
                        kps = []
                        for lm in hand_lm.landmark:
                            if using_realsense:
                                z = camera.get_depth_at(lm.x, lm.y, rgb.shape)
                            else:
                                z = lm.z  # MediaPipe relative depth estimate
                            kps.append(Keypoint3D(x=lm.x, y=lm.y, z=z))

                        mocap = HandMocap(
                            side=physical_side,
                            keypoints=kps,
                            timestamp=time.monotonic(),
                            depth_source=depth_source,
                        )

                        if self.on_mocap:
                            self.on_mocap(mocap)

                        # Draw skeleton
                        _mp_drawing.draw_landmarks(
                            annotated,
                            hand_lm,
                            _mp_hands.HAND_CONNECTIONS,
                            _mp_styles.get_default_hand_landmarks_style(),
                            _mp_styles.get_default_hand_connections_style(),
                        )

                        # Label
                        h, w = annotated.shape[:2]
                        cx = int(hand_lm.landmark[LM.MIDDLE_FINGER_MCP].x * w)
                        cy = int(hand_lm.landmark[LM.MIDDLE_FINGER_MCP].y * h) - 20
                        cv2.putText(
                            annotated,
                            f"{physical_side.upper()} hand",
                            (cx - 40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                # Camera source indicator
                src_color = (0, 200, 255) if using_realsense else (180, 180, 180)
                cv2.putText(
                    annotated,
                    f"CAM: {camera.source_label}",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    src_color,
                    1,
                )

                if self.on_frame:
                    self.on_frame(annotated)

                if self.show_window:
                    cv2.imshow("Aero Hand - Webcam MoCap", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                elapsed = time.monotonic() - t0
                sleep_t = interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        camera.release()
        cv2.destroyAllWindows()
        self._running = False

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _print(mocap: HandMocap):
        wrist = mocap.keypoints[LM.WRIST]
        tip   = mocap.keypoints[LM.INDEX_FINGER_TIP]
        print(
            f"[{mocap.side}|{mocap.depth_source}] "
            f"wrist=({wrist.x:.2f},{wrist.y:.2f}) "
            f"index_tip=({tip.x:.2f},{tip.y:.2f}, z={tip.z:.3f}m)"
        )

    MediaPipeMocap(on_mocap=_print).start()