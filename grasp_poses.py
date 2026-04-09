"""
grasp_poses.py
==============
Grasp pose library for the TetherIA Aero Hand Open.

Maps grasp types to 16 joint angles (degrees) in the order:
  thumb_cmc_abd, thumb_cmc_flex, thumb_mcp, thumb_ip,
  index_mcp,  index_pip,  index_dip,
  middle_mcp, middle_pip, middle_dip,
  ring_mcp,   ring_pip,   ring_dip,
  pinky_mcp,  pinky_pip,  pinky_dip

Force levels map to a speed scalar (0.0-1.0) applied to NORMAL_SPEED
in the bridge. Light grip = slow/gentle, firm = full normal speed.
"""

from __future__ import annotations
from typing import Dict, List, NamedTuple


class GraspPose(NamedTuple):
    name: str
    angles: List[float]          # 16 joint angles in degrees
    description: str


class ForceProfile(NamedTuple):
    level: str                   # "light" | "medium" | "firm"
    speed_scalar: float          # multiplier on NORMAL_SPEED (0.0-1.0)
    description: str


# ---------------------------------------------------------------------------
# Force profiles
# ---------------------------------------------------------------------------

FORCE_PROFILES: Dict[str, ForceProfile] = {
    "light": ForceProfile(
        level="light",
        speed_scalar=0.3,
        description="Gentle grip — fragile or compliant objects"
    ),
    "medium": ForceProfile(
        level="medium",
        speed_scalar=0.6,
        description="Moderate grip — everyday rigid objects"
    ),
    "firm": ForceProfile(
        level="firm",
        speed_scalar=1.0,
        description="Full grip — heavy or slippery objects"
    ),
}


# ---------------------------------------------------------------------------
# Grasp pose library
# ---------------------------------------------------------------------------
# Joint order reference:
#   [0]  thumb_cmc_abd   max=100
#   [1]  thumb_cmc_flex  max=55
#   [2]  thumb_mcp       max=90
#   [3]  thumb_ip        max=90
#   [4]  index_mcp       max=90
#   [5]  index_pip       max=90
#   [6]  index_dip       max=90
#   [7]  middle_mcp      max=90
#   [8]  middle_pip      max=90
#   [9]  middle_dip      max=90
#   [10] ring_mcp        max=90
#   [11] ring_pip        max=90
#   [12] ring_dip        max=90
#   [13] pinky_mcp       max=90
#   [14] pinky_pip       max=90
#   [15] pinky_dip       max=90

GRASP_POSES: Dict[str, GraspPose] = {

    # POWER GRASP — all fingers wrap around object, thumb opposes
    # Best for: cylinder, large sphere, rectangular prism (wide grip)
    "power": GraspPose(
        name="Power Grasp",
        angles=[
            60.0, 35.0, 55.0, 40.0,   # thumb: spread, partially flexed
            75.0, 70.0, 60.0,          # index
            75.0, 70.0, 60.0,          # middle
            70.0, 65.0, 55.0,          # ring
            65.0, 60.0, 50.0,          # pinky
        ],
        description="Full hand wrap — cylinder, large objects"
    ),

    # PINCH (2-finger) — index tip meets thumb tip
    # Best for: small cube, pen, small sphere
    "pinch": GraspPose(
        name="Pinch Grasp",
        angles=[
            40.0, 30.0, 50.0, 45.0,   # thumb toward index
            55.0, 50.0, 40.0,          # index partially curled
            15.0, 10.0,  5.0,          # middle slightly open
            10.0,  5.0,  0.0,          # ring open
             5.0,  0.0,  0.0,          # pinky open
        ],
        description="Index-thumb pinch — small precise objects"
    ),

    # TRIPOD — thumb + index + middle tips meet
    # Best for: cube, small rectangular prism, medium sphere
    "tripod": GraspPose(
        name="Tripod Grasp",
        angles=[
            45.0, 32.0, 52.0, 45.0,   # thumb toward center
            55.0, 50.0, 40.0,          # index
            55.0, 50.0, 40.0,          # middle
            15.0, 10.0,  5.0,          # ring relaxed
             5.0,  0.0,  0.0,          # pinky open
        ],
        description="3-finger precision — medium objects, cube/prism"
    ),

    # LATERAL PINCH — thumb presses against side of index
    # Best for: flat rectangular prism, thin objects
    "lateral_pinch": GraspPose(
        name="Lateral Pinch",
        angles=[
            20.0, 25.0, 40.0, 35.0,   # thumb low abduction, pushes laterally
            45.0, 40.0, 30.0,          # index moderately curled
            20.0, 15.0, 10.0,          # middle slightly curled
            10.0,  5.0,  0.0,          # ring open
             5.0,  0.0,  0.0,          # pinky open
        ],
        description="Side pinch — flat or thin objects"
    ),

    # PALMAR GRASP — object rests in palm, all fingers close over it
    # Best for: squish ball, large soft objects
    "palmar": GraspPose(
        name="Palmar Grasp",
        angles=[
            55.0, 30.0, 45.0, 35.0,   # thumb wraps from side
            60.0, 55.0, 45.0,          # index
            65.0, 60.0, 50.0,          # middle (primary contact)
            60.0, 55.0, 45.0,          # ring
            55.0, 50.0, 40.0,          # pinky
        ],
        description="Palm cradle — soft/squish objects, large spheres"
    ),

    # OPEN — fully open hand
    "open": GraspPose(
        name="Open Hand",
        angles=[0.0] * 16,
        description="Fully open — approach or release"
    ),
}


# ---------------------------------------------------------------------------
# Object defaults — used as fallback if LLM call fails
# ---------------------------------------------------------------------------

OBJECT_DEFAULTS: Dict[str, tuple] = {
    "sphere":             ("power",         "medium", "Wrap around sphere"),
    "wooden_sphere":      ("power",         "firm",   "Rigid sphere, firm grip"),
    "cylinder":           ("power",         "medium", "Standard power grasp"),
    "cube":               ("tripod",        "medium", "3-finger on cube faces"),
    "rectangular_prism":  ("lateral_pinch", "medium", "Flat side contact"),
    "squish_ball":        ("palmar",        "light",  "Soft — gentle palmar cradle"),
    "small_object":       ("pinch",         "light",  "Generic small object"),
    "large_object":       ("power",         "firm",   "Generic large object"),
    "flat_object":        ("lateral_pinch", "medium", "Generic flat object"),
    "unknown":            ("tripod",        "medium", "Safe default"),
    "pyramid": ("tripod", "medium", "Tripod grasp on pyramid apex faces"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pose(grasp_type: str) -> GraspPose:
    """Return GraspPose by name, falling back to tripod if unknown."""
    return GRASP_POSES.get(grasp_type, GRASP_POSES["tripod"])


def get_force(force_level: str) -> ForceProfile:
    """Return ForceProfile by level, falling back to medium."""
    return FORCE_PROFILES.get(force_level, FORCE_PROFILES["medium"])


def get_default_for_object(object_label: str) -> tuple:
    """
    Returns (GraspPose, ForceProfile) for a known object label.
    Falls back to 'unknown' entry if label not in library.
    """
    key = object_label.lower().replace(" ", "_")
    grasp_type, force_level, _ = OBJECT_DEFAULTS.get(
        key, OBJECT_DEFAULTS["unknown"]
    )
    return get_pose(grasp_type), get_force(force_level)


def interpolate_angles(
    start: List[float],
    end: List[float],
    t: float,
) -> List[float]:
    """
    Linear interpolation between two 16-angle poses.
    t=0.0 -> start, t=1.0 -> end.
    """
    t = max(0.0, min(1.0, t))
    return [s + (e - s) * t for s, e in zip(start, end)]