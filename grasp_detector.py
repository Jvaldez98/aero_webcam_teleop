"""
grasp_detector.py
=================
Object detection module using YOLOv8.

Detects objects in a camera frame and returns bounding boxes,
labels, confidence scores, and estimated real-world size.

When a RealSense depth frame is available, object size is estimated
in real centimeters. Without depth, pixel-based size estimation is
used as a fallback.

Install dependency:
    pip install ultralytics

Supported objects (manually defined for SMILE Lab demo):
    sphere, wooden_sphere, cylinder, cube,
    rectangular_prism, squish_ball

YOLO's default 80-class vocabulary is used for detection, then
labels are mapped to our object library via the label mapper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[GraspDetector] ultralytics not installed. Run: pip install ultralytics")


# ---------------------------------------------------------------------------
# YOLO label -> our object library mapping
# ---------------------------------------------------------------------------
# Maps YOLO's 80-class labels to our defined object set.
# Extend this as you add more objects to OBJECT_DEFAULTS in grasp_poses.py.

YOLO_TO_OBJECT: dict = {
    # Spheres
    "sports ball":      "wooden_sphere",
    "orange":           "wooden_sphere",
    "apple":            "wooden_sphere",
    "tennis ball":      "wooden_sphere",
    "baseball":         "wooden_sphere",
    # Cylinders
    "cup":              "cylinder",
    "bottle":           "cylinder",
    "vase":             "cylinder",
    "bowl":             "cylinder",
    "wine glass":       "cylinder",
    # Rectangular prisms
    "book":             "rectangular_prism",
    "laptop":           "rectangular_prism",
    "cell phone":       "rectangular_prism",
    "remote":           "rectangular_prism",
    "box":              "rectangular_prism",
    "suitcase":         "rectangular_prism",
    # Pyramid / weird shapes — YOLO has no pyramid class,
    # closest match is potted plant or traffic cone
    "traffic cone":     "pyramid",
    "potted plant":     "pyramid",
}

# Direct label matches (for when you add custom YOLO training later)
DIRECT_LABELS = {
    "sphere", "wooden_sphere", "cylinder", "cube",
    "rectangular_prism", "squish_ball",
}


@dataclass
class Detection:
    label: str                        # our object library label
    yolo_label: str                   # raw YOLO class name
    confidence: float                 # 0.0-1.0
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) pixels
    size_px: float                    # approximate diameter/diagonal in pixels
    size_cm: Optional[float] = None   # real-world size in cm (D405 only)
    depth_m: Optional[float] = None   # distance to object center in meters


class GraspDetector:
    """
    Wraps YOLOv8 for object detection in the grasp pipeline.

    Parameters
    ----------
    model_size      : YOLOv8 variant — "n" (nano/fastest) to "x" (largest)
    confidence      : Minimum detection confidence threshold
    device          : "cpu" | "cuda" | "mps" (auto-detected if None)
    """

    def __init__(
        self,
        model_size: str = "n",
        confidence: float = 0.45,
        device: Optional[str] = None,
    ):
        self.confidence = confidence
        self._model = None

        if not _YOLO_AVAILABLE:
            print("[GraspDetector] Running without YOLO — no object detection.")
            return

        model_name = f"yolov8{model_size}.pt"
        print(f"[GraspDetector] Loading {model_name}...")
        self._model = YOLO(model_name)

        if device:
            self._model.to(device)

        print(f"[GraspDetector] Ready. Confidence threshold: {confidence}")

    def detect(
        self,
        frame_rgb: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """
        Run detection on an RGB frame.

        Parameters
        ----------
        frame_rgb   : RGB image as numpy array (H, W, 3)
        depth_frame : Optional aligned depth frame in meters (H, W) float32
                      — provided by RealSense backend

        Returns
        -------
        List of Detection objects sorted by confidence (highest first).
        """
        if self._model is None:
            return []

        h, w = frame_rgb.shape[:2]
        results = self._model(
            frame_rgb,
            conf=self.confidence,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            yolo_label = results.names[cls_id].lower()

            # Map to our object library
            if yolo_label in DIRECT_LABELS:
                obj_label = yolo_label
            else:
                obj_label = YOLO_TO_OBJECT.get(yolo_label, "unknown")

            # Size estimation
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            size_px = float(max(bbox_w, bbox_h))

            # Real-world depth + size from D405
            size_cm = None
            depth_m = None
            if depth_frame is not None:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                patch = depth_frame[
                    max(0, cy - 5):cy + 6,
                    max(0, cx - 5):cx + 6,
                ]
                nonzero = patch[patch > 0]
                if len(nonzero) > 0:
                    depth_m = float(np.median(nonzero))
                    # Approximate object size in cm using pinhole geometry
                    # Assumes ~70 degree FOV for D405 at 640px width
                    fov_px_per_meter = w / (2 * depth_m * np.tan(np.radians(35)))
                    size_cm = (size_px / fov_px_per_meter) * 100.0

            detections.append(Detection(
                label=obj_label,
                yolo_label=yolo_label,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                size_px=size_px,
                size_cm=size_cm,
                depth_m=depth_m,
            ))

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def draw_detections(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        highlight_top: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a BGR frame.
        The top detection is highlighted in a different color.
        """
        out = frame_bgr.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            is_top = highlight_top and i == 0

            color = (0, 255, 150) if is_top else (150, 150, 150)
            thickness = 2 if is_top else 1

            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Label text
            size_str = f" {det.size_cm:.1f}cm" if det.size_cm else f" {det.size_px:.0f}px"
            depth_str = f" @ {det.depth_m:.2f}m" if det.depth_m else ""
            label_str = (
                f"{det.label} ({det.confidence:.0%}){size_str}{depth_str}"
            )

            label_y = max(y1 - 8, 14)
            cv2.putText(
                out, label_str,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 if is_top else 0.35,
                color,
                1 if is_top else 1,
            )

        return out