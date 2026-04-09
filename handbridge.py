"""
hand_bridge.py
==============
Mirrors the ROS2 `aero_hand_node` (hardware node).

Wraps the TetherIA `aero_open_sdk.AeroHand` class and exposes
the same control interface as the ROS2 node.

If the SDK is not installed or the hand is not connected, the bridge
runs in **mock mode** — it prints commands instead of sending them.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

try:
    from aero_open_sdk.aero_hand import AeroHand as _AeroHand  # type: ignore
    _SDK_AVAILABLE = True
except ImportError:
    _AeroHand = None  # type: ignore
    _SDK_AVAILABLE = False


JOINT_NAMES: List[str] = [
    "thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip",
    "index_mcp",  "index_pip",  "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp",   "ring_pip",   "ring_dip",
    "pinky_mcp",  "pinky_pip",  "pinky_dip",
]

# SDK joint limits (degrees) — from docs.tetheria.ai/docs/sdk
JOINT_MIN_DEG: List[float] = [0.0] * 16
JOINT_MAX_DEG: List[float] = [
    100.0, 55.0, 90.0, 90.0,   # thumb
     90.0, 90.0, 90.0,          # index
     90.0, 90.0, 90.0,          # middle
     90.0, 90.0, 90.0,          # ring
     90.0, 90.0, 90.0,          # pinky
]

# ---------------------------------------------------------------------------
# Speed constants
# SDK range: 0–32766. Max by default (firmware runs full speed).
# Keep homing slow to protect tendons — use ~5% of max.
# Normal teleoperation uses a higher value for responsiveness.
# ---------------------------------------------------------------------------
HOMING_SPEED    = 1500   # ~5% of max — very gentle, tendon-safe
NORMAL_SPEED    = 20000  # ~60% of max — good balance for teleop
FULL_SPEED      = 32766  # SDK maximum


class AeroHandBridge:
    """
    Hardware bridge — wraps the SDK or runs in mock mode.

    Parameters
    ----------
    port            : Serial port, e.g. "COM3" or "/dev/ttyACM0".
                      If None or SDK not found → mock mode.
    baudrate        : Serial baud rate (default 921600).
    side            : "right" | "left" (informational).
    verbose         : Print every command in mock mode.
    homing_speed    : Speed (0–32766) used during homing. Default is
                      intentionally very low to protect tendons.
    normal_speed    : Speed restored after homing / used during teleop.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 921_600,
        side: str = "right",
        verbose: bool = True,
        homing_speed: int = HOMING_SPEED,
        normal_speed: int = NORMAL_SPEED,
    ):
        self.side = side
        self.verbose = verbose
        self.homing_speed = homing_speed
        self.normal_speed = normal_speed
        self._mock = True
        self._hand: Optional[_AeroHand] = None

        if port and _SDK_AVAILABLE:
            try:
                self._hand = _AeroHand(port=port)
                self._mock = False
                print(f"[AeroHandBridge] Connected to {side} hand on {port}")
                # Set safe operating speed immediately on connect
                self._set_all_speeds(self.normal_speed)
            except Exception as exc:
                print(f"[AeroHandBridge] Could not connect to {port}: {exc}")
                print("[AeroHandBridge] Running in MOCK mode.")
        else:
            if not _SDK_AVAILABLE:
                print(
                    "[AeroHandBridge] aero_open_sdk not found — running in MOCK mode.\n"
                    "  Install with: pip install aero-open-sdk"
                )
            else:
                print("[AeroHandBridge] No port specified — running in MOCK mode.")

    # ------------------------------------------------------------------
    # Internal speed helpers
    # ------------------------------------------------------------------

    def _set_all_speeds(self, speed: int):
        """Set all 7 actuators to the given speed value."""
        if self._hand:
            speed = max(0, min(32766, speed))
            for actuator_id in range(7):
                self._hand.set_speed(actuator_id, speed)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    # Measured safe actuation maxima per actuator (degrees), from hardware testing.
    # Used as a hard post-send clamp via set_actuations if needed.
    # [thumb_abd, thumb_flex, thumb_tendon, index, middle, ring, pinky]
    SAFE_ACTUATION_MAX = [100.0, 131.0, 274.0, 250.0, 240.0, 230.0, 220.0]

    def send_joint_positions(self, angles_deg: List[float]):
        """
        Send 16 joint angles (degrees) to the hand.

        Passes angles directly to set_joint_positions — the SDK's internal
        joints_to_actuations mapping handles the thumb coupling correctly.
        Joint angles are clamped to SDK limits before sending.
        """
        assert len(angles_deg) == 16, f"Expected 16 angles, got {len(angles_deg)}"
        if not self._mock and self._hand:
            # Clamp each joint to its SDK-defined limit before sending.
            # This is the only clamping we do — we trust the SDK's
            # joints_to_actuations mapping to handle thumb coupling correctly.
            clamped = [
                float(max(JOINT_MIN_DEG[i], min(JOINT_MAX_DEG[i], a)))
                for i, a in enumerate(angles_deg)
            ]
            self._hand.set_joint_positions(clamped)
        elif self.verbose:
            pairs = {JOINT_NAMES[i]: f"{a:.1f}°" for i, a in enumerate(angles_deg)}
            print(f"[MOCK send_joint_positions] {pairs}")

    def send_compact_positions(self, angles_deg: List[float]):
        """Send 7 compact joint angles in degrees."""
        assert len(angles_deg) == 7, f"Expected 7 angles, got {len(angles_deg)}"
        if not self._mock and self._hand:
            self._hand.set_joint_positions(angles_deg)
        elif self.verbose:
            labels = ["thumb_abd", "thumb_flex", "thumb_mcp_ip",
                      "index", "middle", "ring", "pinky"]
            pairs = {labels[i]: f"{a:.1f}°" for i, a in enumerate(angles_deg)}
            print(f"[MOCK send_compact_positions] {pairs}")

    def home(self, speed: int = 500):
        """
        Tendon-safe homing using send_homing() at a very low speed.

        Speed 500/32766 (~1.5% of max) is slow enough that the tendon
        seats gently against the hard stop without snapping. Tested and
        confirmed on hardware.

        Parameters
        ----------
        speed : actuator speed during homing (0–32766, default 500).
                Lower = safer but slower. Do not exceed 1500 without
                first verifying cable tension by hand.
        """
        if not self._mock and self._hand:
            print(f"[AeroHandBridge] Setting homing speed: {speed}/32766 ({100*speed//32766}% of max)")
            for i in range(7):
                self._hand.set_speed(i, speed)
            time.sleep(0.2)
            print("[AeroHandBridge] Homing… (watch the hand, disconnect USB if anything looks wrong)")
            self._hand.send_homing()
            print("[AeroHandBridge] Homing complete.")
            # Restore normal operating speed
            self._set_all_speeds(self.normal_speed)
            print(f"[AeroHandBridge] Speed restored to {self.normal_speed}/32766")
        else:
            print("[MOCK home] Homing command sent (mock, no hardware).")

    def open_hand(self):
        """Move all joints to open-palm pose (all zeros)."""
        self.send_joint_positions([0.0] * 16)

    def close_hand(self):
        """Move to a closed-fist approximation."""
        self.send_joint_positions([
            50.0, 27.5, 45.0, 45.0,   # thumb
            90.0, 90.0, 90.0,          # index
            90.0, 90.0, 90.0,          # middle
            90.0, 90.0, 90.0,          # ring
            90.0, 90.0, 90.0,          # pinky
        ])

    def set_teleop_speed(self, speed: int):
        """Manually override all actuator speeds (0–32766)."""
        if not self._mock and self._hand:
            self._set_all_speeds(speed)
            print(f"[AeroHandBridge] Speed set to {speed}/32766 ({100*speed//32766}%)")

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_actuator_states(self) -> Dict[str, List[float]]:
        """Returns actuations, speeds, currents, temperatures (all length 7)."""
        if not self._mock and self._hand:
            return {
                "actuations":   self._hand.get_actuations(),
                "speeds":       self._hand.get_actuator_speeds(),
                "currents":     self._hand.get_actuator_currents(),
                "temperatures": self._hand.get_actuator_temperatures(),
            }
        return {
            "actuations":   [0.0] * 7,
            "speeds":       [0.0] * 7,
            "currents":     [0.0] * 7,
            "temperatures": [25.0] * 7,
        }

    def close(self):
        """Release serial connection."""
        if self._hand:
            try:
                self.open_hand()
                time.sleep(0.5)
            finally:
                pass
        print("[AeroHandBridge] Closed.")