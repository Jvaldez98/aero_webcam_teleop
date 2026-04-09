"""
mediapipe_retargeting.py
========================
Converts HandMocap (21 MediaPipe keypoints) into 16 joint angles (degrees)
for the TetherIA Aero Hand Open.

Key design decisions based on SDK docs and hardware testing
-----------------------------------------------------------
1.  The SDK's set_joint_positions handles thumb coupling internally via its
    joints_to_actuations mapping — we just need to give it correct joint angles
    for all 4 thumb joints (thumb_cmc_abd, thumb_cmc_flex, thumb_mcp, thumb_ip).

2.  MediaPipe reports raw angles that are systematically lower than the true
    anatomical ROM because:
      a) Single-camera z-depth is noisy → bone vectors are foreshortened
      b) Thumb segments are short → angular change per mm is small in image space
      c) Wrist-relative coordinates compress distal joints

3.  We use per-joint output scaling (JOINT_SCALE) to map the observed
    MediaPipe range → the robot's full SDK joint range.
    Thumb joints get 2.0–2.5x because they're most affected by (b) and (c).
    Finger joints get 1.6x because MediaPipe typically reports ~55° max
    even for a fully closed fist (true anatomical max ~90°).

4.  Calibration (press 'c' with open palm) removes systematic zero offsets.
    After calibration, close your fist and check the bar display reaches max.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .mediapipe_mocap import HandMocap, Keypoint3D

import mediapipe as mp
try:
    _mp_hands = mp.solutions.hands
except AttributeError:
    from mediapipe.python.solutions import hands as _mp_hands  # type: ignore

LM = _mp_hands.HandLandmark


# ---------------------------------------------------------------------------
# Joint limits — exactly as defined in SDK docs
# ---------------------------------------------------------------------------

JOINT_NAMES: List[str] = [
    "thumb_cmc_abd",   # 0   max 100°
    "thumb_cmc_flex",  # 1   max  55°
    "thumb_mcp",       # 2   max  90°
    "thumb_ip",        # 3   max  90°
    "index_mcp",       # 4   max  90°
    "index_pip",       # 5   max  90°
    "index_dip",       # 6   max  90°
    "middle_mcp",      # 7   max  90°
    "middle_pip",      # 8   max  90°
    "middle_dip",      # 9   max  90°
    "ring_mcp",        # 10  max  90°
    "ring_pip",        # 11  max  90°
    "ring_dip",        # 12  max  90°
    "pinky_mcp",       # 13  max  90°
    "pinky_pip",       # 14  max  90°
    "pinky_dip",       # 15  max  90°
]

JOINT_MIN_DEG: List[float] = [0.0] * 16
JOINT_MAX_DEG: List[float] = [
    100.0, 55.0, 90.0, 90.0,   # thumb
     90.0, 90.0, 90.0,          # index
     90.0, 90.0, 90.0,          # middle
     90.0, 90.0, 90.0,          # ring
     90.0, 90.0, 90.0,          # pinky
]

# ---------------------------------------------------------------------------
# Per-joint output scale factors
#
# These map the raw MediaPipe angle geometry → the robot's full ROM.
#
# Thumb:
#   - thumb_cmc_abd  (0): abduction measured as angle between palm-projected
#     rays — tends to be accurate but under-ranged. Scale 1.8.
#   - thumb_cmc_flex (1): short segment, heavily foreshortened. Scale 2.5.
#   - thumb_mcp      (2): medium segment. Scale 2.0.
#   - thumb_ip       (3): distal, most foreshortened. Scale 2.2.
#
# Fingers:
#   - MCP joints (4,7,10,13): largest segment, least foreshortened. Scale 1.5.
#   - PIP joints (5,8,11,14): medium. Scale 1.7.
#   - DIP joints (6,9,12,15): smallest, most foreshortened. Scale 1.8.
#
# These are tuned so that a fully closed human fist maps to ~90° on all joints.
# Adjust if your hand's anatomy differs significantly.
# ---------------------------------------------------------------------------
JOINT_SCALE: List[float] = [
    1.8, 2.5, 2.0, 2.2,   # thumb (abd, flex, mcp, ip)
    1.5, 1.7, 1.8,         # index  (mcp, pip, dip)
    1.5, 1.7, 1.8,         # middle
    1.5, 1.7, 1.8,         # ring
    1.5, 1.7, 1.8,         # pinky
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _kp_to_vec(kp: Keypoint3D) -> np.ndarray:
    return np.array([kp.x, kp.y, kp.z], dtype=np.float64)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.acos(cos_a))


def _flexion_deg(
    proximal: np.ndarray,
    mid: np.ndarray,
    distal: np.ndarray,
    joint_max: float,
    scale: float = 1.0,
) -> float:
    """
    Compute joint flexion from three consecutive landmark positions.
    Angle between (proximal→mid) and (mid→distal), scaled and clamped.
    """
    v1 = mid - proximal
    v2 = distal - mid
    raw_rad = _angle_between(v1, v2)
    # Map [0, π] → [0, joint_max], apply scale, clamp
    scaled = (raw_rad / math.pi) * joint_max * scale
    return float(np.clip(scaled, 0.0, joint_max))


# ---------------------------------------------------------------------------
# Config + retargeting class
# ---------------------------------------------------------------------------

@dataclass
class RetargetingConfig:
    """Per-user normalisation — calibrated at open-palm rest pose."""
    offsets: List[float] = field(default_factory=lambda: [0.0] * 16)
    scales:  List[float] = field(default_factory=lambda: [1.0] * 16)


class MediaPipeRetargeting:
    """
    Converts HandMocap → 16 joint angles (degrees) for the Aero Hand Open.

    The SDK's set_joint_positions handles thumb coupling internally, so we
    compute all 4 thumb joint angles independently and let the SDK sort out
    the 3-actuator coupling.

    Parameters
    ----------
    config          : RetargetingConfig for per-user calibration
    smoothing_alpha : EMA smoothing [0,1]. Lower = smoother but laggier.
                      0.35 is a good starting point.
    """

    def __init__(
        self,
        config: Optional[RetargetingConfig] = None,
        smoothing_alpha: float = 0.35,
    ):
        self.config = config or RetargetingConfig()
        self.alpha  = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self._prev: Optional[List[float]] = None

    def retarget(self, mocap: HandMocap) -> List[float]:
        """Convert HandMocap → 16 joint angles in degrees."""
        kp = mocap.keypoints
        angles = [0.0] * 16

        def v(idx: int) -> np.ndarray:
            return _kp_to_vec(kp[idx])

        # ── Thumb ─────────────────────────────────────────────────────────
        # thumb_cmc_abd (0): angle between thumb CMC ray and index MCP ray,
        # projected onto the palm plane, so rotation around palm normal.
        palm_normal = self._palm_normal(kp)
        thumb_ray   = v(LM.THUMB_CMC)        - v(LM.WRIST)
        index_ray   = v(LM.INDEX_FINGER_MCP) - v(LM.WRIST)
        thumb_proj  = thumb_ray  - np.dot(thumb_ray,  palm_normal) * palm_normal
        index_proj  = index_ray  - np.dot(index_ray,  palm_normal) * palm_normal
        abd_rad     = _angle_between(thumb_proj, index_proj)
        angles[0]   = float(np.clip(
            (abd_rad / math.pi) * JOINT_MAX_DEG[0] * JOINT_SCALE[0],
            0.0, JOINT_MAX_DEG[0]
        ))

        # thumb_cmc_flex (1): flexion at CMC — wrist → CMC → MCP
        angles[1] = _flexion_deg(
            v(LM.WRIST), v(LM.THUMB_CMC), v(LM.THUMB_MCP),
            JOINT_MAX_DEG[1], JOINT_SCALE[1]
        )
        # thumb_mcp (2): flexion at MCP — CMC → MCP → IP
        angles[2] = _flexion_deg(
            v(LM.THUMB_CMC), v(LM.THUMB_MCP), v(LM.THUMB_IP),
            JOINT_MAX_DEG[2], JOINT_SCALE[2]
        )
        # thumb_ip (3): flexion at IP — MCP → IP → TIP
        angles[3] = _flexion_deg(
            v(LM.THUMB_MCP), v(LM.THUMB_IP), v(LM.THUMB_TIP),
            JOINT_MAX_DEG[3], JOINT_SCALE[3]
        )

        # ── Index ──────────────────────────────────────────────────────────
        angles[4] = _flexion_deg(v(LM.WRIST),            v(LM.INDEX_FINGER_MCP), v(LM.INDEX_FINGER_PIP), JOINT_MAX_DEG[4],  JOINT_SCALE[4])
        angles[5] = _flexion_deg(v(LM.INDEX_FINGER_MCP), v(LM.INDEX_FINGER_PIP), v(LM.INDEX_FINGER_DIP), JOINT_MAX_DEG[5],  JOINT_SCALE[5])
        angles[6] = _flexion_deg(v(LM.INDEX_FINGER_PIP), v(LM.INDEX_FINGER_DIP), v(LM.INDEX_FINGER_TIP), JOINT_MAX_DEG[6],  JOINT_SCALE[6])

        # ── Middle ─────────────────────────────────────────────────────────
        angles[7] = _flexion_deg(v(LM.WRIST),              v(LM.MIDDLE_FINGER_MCP), v(LM.MIDDLE_FINGER_PIP), JOINT_MAX_DEG[7],  JOINT_SCALE[7])
        angles[8] = _flexion_deg(v(LM.MIDDLE_FINGER_MCP),  v(LM.MIDDLE_FINGER_PIP), v(LM.MIDDLE_FINGER_DIP), JOINT_MAX_DEG[8],  JOINT_SCALE[8])
        angles[9] = _flexion_deg(v(LM.MIDDLE_FINGER_PIP),  v(LM.MIDDLE_FINGER_DIP), v(LM.MIDDLE_FINGER_TIP), JOINT_MAX_DEG[9],  JOINT_SCALE[9])

        # ── Ring ───────────────────────────────────────────────────────────
        angles[10] = _flexion_deg(v(LM.WRIST),            v(LM.RING_FINGER_MCP), v(LM.RING_FINGER_PIP), JOINT_MAX_DEG[10], JOINT_SCALE[10])
        angles[11] = _flexion_deg(v(LM.RING_FINGER_MCP),  v(LM.RING_FINGER_PIP), v(LM.RING_FINGER_DIP), JOINT_MAX_DEG[11], JOINT_SCALE[11])
        angles[12] = _flexion_deg(v(LM.RING_FINGER_PIP),  v(LM.RING_FINGER_DIP), v(LM.RING_FINGER_TIP), JOINT_MAX_DEG[12], JOINT_SCALE[12])

        # ── Pinky ──────────────────────────────────────────────────────────
        angles[13] = _flexion_deg(v(LM.WRIST),      v(LM.PINKY_MCP), v(LM.PINKY_PIP), JOINT_MAX_DEG[13], JOINT_SCALE[13])
        angles[14] = _flexion_deg(v(LM.PINKY_MCP),  v(LM.PINKY_PIP), v(LM.PINKY_DIP), JOINT_MAX_DEG[14], JOINT_SCALE[14])
        angles[15] = _flexion_deg(v(LM.PINKY_PIP),  v(LM.PINKY_DIP), v(LM.PINKY_TIP), JOINT_MAX_DEG[15], JOINT_SCALE[15])

        # ── Per-user calibration ───────────────────────────────────────────
        cfg = self.config
        angles = [
            float(np.clip(
                a * cfg.scales[i] + cfg.offsets[i],
                JOINT_MIN_DEG[i],
                JOINT_MAX_DEG[i],
            ))
            for i, a in enumerate(angles)
        ]

        # ── EMA smoothing ──────────────────────────────────────────────────
        if self._prev is None:
            self._prev = angles[:]
        else:
            angles = [
                self.alpha * a + (1.0 - self.alpha) * p
                for a, p in zip(angles, self._prev)
            ]
            self._prev = angles[:]

        return angles

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _palm_normal(kp: List[Keypoint3D]) -> np.ndarray:
        """Unit normal to the palm plane."""
        wrist = _kp_to_vec(kp[LM.WRIST])
        index = _kp_to_vec(kp[LM.INDEX_FINGER_MCP])
        pinky = _kp_to_vec(kp[LM.PINKY_MCP])
        n     = np.cross(index - wrist, pinky - wrist)
        norm  = np.linalg.norm(n)
        return n / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

    def calibrate(self, open_palm_mocap: HandMocap):
        """
        Record open-palm reference and zero out systematic offsets.
        Hold hand open flat, then press 'c' in the pipeline window.
        After calibrating, close your fist and check bars reach max.
        If fingers still don't reach max, increase JOINT_SCALE values.
        """
        raw = self.retarget(open_palm_mocap)
        self.config.offsets = [-r for r in raw]
        self._prev = None
        print("[RetargetingConfig] Calibration complete.")
        print("  Offsets:", [round(o, 2) for o in self.config.offsets])
        print("  Now close your fist — check bars reach maximum.")
        print("  If not, increase JOINT_SCALE values in mediapipe_retargeting.py.")