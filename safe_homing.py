"""
safe_homing.py
==============
Tendon-safe homing for the TetherIA Aero Hand Open.

Per-actuator strategy based on measured current sensor reliability:

  ID  Name               Sensor    Strategy
  0   thumb_cmc_abd      noisy     position target (drive to known safe limit)
  1   thumb_cmc_flex     noisy     position target
  2   thumb_tendon       inverted  abs(current) threshold, inverted sign
  3   index_tendon       noisy     position target
  4   middle_tendon      clean     current threshold (positive)
  5   ring_tendon        clean     current threshold (positive)
  6   pinky_tendon       clean     current threshold (positive)

For position-based actuators, we drive slowly to a conservative target
that seats the tendon without hitting the hard stop aggressively.
"""

from __future__ import annotations

import time
import argparse
from typing import List, Optional, Dict

try:
    from aero_open_sdk.aero_hand import AeroHand  # type: ignore
except ImportError:
    AeroHand = None  # type: ignore

# ---------------------------------------------------------------------------
# Per-actuator configuration
# ---------------------------------------------------------------------------

ACTUATOR_NAMES = [
    "thumb_cmc_abd",   # 0
    "thumb_cmc_flex",  # 1
    "thumb_tendon",    # 2
    "index_tendon",    # 3
    "middle_tendon",   # 4
    "ring_tendon",     # 5
    "pinky_tendon",    # 6
]

# Strategy types
STRATEGY_CURRENT  = "current"   # stop when abs(current) exceeds threshold
STRATEGY_POSITION = "position"  # drive to a fixed target angle, no current check

# Per-actuator config:
# (strategy, torque, current_threshold_ma, position_target_deg, current_sign)
# current_sign: +1 = positive current expected at load, -1 = inverted
ACTUATOR_CONFIG = {
    #   id  strategy            torque  threshold  pos_target  sign
    0: (STRATEGY_POSITION,      80,     None,       85.0,      +1),  # thumb abd — noisy sensor
    1: (STRATEGY_POSITION,      80,     None,       45.0,      +1),  # thumb flex — noisy sensor
    2: (STRATEGY_CURRENT,       80,     400,        None,      -1),  # thumb tendon — inverted
    3: (STRATEGY_POSITION,      100,    None,       240.0,     +1),  # index — noisy sensor
    4: (STRATEGY_CURRENT,       100,    500,        None,      +1),  # middle — clean
    5: (STRATEGY_CURRENT,       100,    600,        None,      +1),  # ring — clean
    6: (STRATEGY_CURRENT,       100,    200,        None,      +1),  # pinky — clean, lower threshold
}

BACKOFF_DEG       = 6.0    # degrees to back off after current-based contact
POLL_INTERVAL_S   = 0.05   # 20 Hz
TIMEOUT_S         = 40.0
HOMING_SPEED      = 1200   # very slow for all actuators during homing


class HomingResult:
    CONTACT  = "contact"
    POSITION = "position_target"
    TIMEOUT  = "timeout"
    SNAP     = "snap"
    SKIPPED  = "skipped"


def _emergency_stop(hand: "AeroHand", reason: str):
    try:
        hand.ctrl_torque([0] * 7)
    except Exception:
        pass
    print(f"\n{'!' * 60}")
    print(f"  EMERGENCY STOP — {reason}")
    print(f"{'!' * 60}\n")


def safe_home(
    hand: "AeroHand",
    actuators: Optional[List[int]] = None,
    backoff_deg: float = BACKOFF_DEG,
    timeout_s: float = TIMEOUT_S,
) -> Dict[int, str]:

    if actuators is None:
        actuators = list(range(7))

    results = {i: HomingResult.SKIPPED for i in range(7)}

    print("\n" + "=" * 60)
    print("  Safe Manual Homing (per-actuator strategy)")
    print("=" * 60)
    for act_id in actuators:
        strategy, torque, threshold, pos_target, sign = ACTUATOR_CONFIG[act_id]
        if strategy == STRATEGY_CURRENT:
            detail = f"current  threshold=±{threshold} mA  sign={'inverted' if sign < 0 else 'normal'}"
        else:
            detail = f"position target={pos_target}°"
        print(f"  [{act_id}] {ACTUATOR_NAMES[act_id]:<20} {detail}")
    print("=" * 60 + "\n")

    # Set slow speed on all actuators before any motion
    for i in range(7):
        hand.set_speed(i, HOMING_SPEED)
    time.sleep(0.1)

    snap_detected = False

    for act_id in actuators:
        if snap_detected:
            print(f"[safe_home] Skipping {ACTUATOR_NAMES[act_id]} — snap halt active.")
            results[act_id] = HomingResult.SNAP
            continue

        strategy, torque, threshold, pos_target, sign = ACTUATOR_CONFIG[act_id]
        name = ACTUATOR_NAMES[act_id]

        print(f"[safe_home] ── {name} (id={act_id}, strategy={strategy}) ──")

        # ----------------------------------------------------------------
        if strategy == STRATEGY_POSITION:
            # Drive to the known safe target position directly
            try:
                current_pos = hand.get_actuations()
                target = current_pos[:]
                target[act_id] = pos_target
                hand.set_actuations(target)

                # Wait until the actuator reaches the target (or timeout)
                t_start = time.monotonic()
                while True:
                    time.sleep(POLL_INTERVAL_S)
                    try:
                        pos = hand.get_actuations()
                        dist = abs(pos[act_id] - pos_target)
                    except Exception:
                        dist = 999
                    elapsed = time.monotonic() - t_start
                    if dist < 3.0:
                        print(f"[safe_home] {name}: reached target {pos_target}° ✓ ({elapsed:.1f}s)")
                        results[act_id] = HomingResult.POSITION
                        break
                    if elapsed >= timeout_s:
                        print(f"[safe_home] {name}: timeout — dist={dist:.1f}° from target")
                        results[act_id] = HomingResult.TIMEOUT
                        break
                    if int(elapsed) % 2 == 0 and elapsed > 0.1:
                        print(f"[safe_home]   {name}: {elapsed:.0f}s  dist={dist:.1f}°", end="\r")
                print()

            except Exception as e:
                print(f"[safe_home] {name}: position move failed: {e}")
                results[act_id] = HomingResult.TIMEOUT

        # ----------------------------------------------------------------
        elif strategy == STRATEGY_CURRENT:
            # Apply torque and watch for current spike
            torque_cmd = [0] * 7
            torque_cmd[act_id] = torque
            hand.ctrl_torque(torque_cmd)

            t_start       = time.monotonic()
            loaded_cycles = 0
            was_loaded    = False
            result        = HomingResult.TIMEOUT
            LOADED_CONFIRM_MA = threshold * 0.25  # 25% of threshold = "under load"
            SNAP_DROP_MA      = 35                # near-zero after load = snap

            while True:
                elapsed = time.monotonic() - t_start
                try:
                    raw = hand.get_actuator_currents()[act_id]
                    current_ma = raw * sign  # normalise sign
                except Exception:
                    current_ma = 0.0

                # Track loading
                if current_ma >= LOADED_CONFIRM_MA:
                    loaded_cycles += 1
                    if loaded_cycles >= 3:
                        was_loaded = True
                else:
                    loaded_cycles = 0

                # Snap detection — current drops to near-zero after confirmed load
                if was_loaded and (raw * sign) < SNAP_DROP_MA:
                    _emergency_stop(
                        hand,
                        f"SNAP suspected on {name} — current dropped to {raw:.0f} mA"
                    )
                    snap_detected = True
                    result = HomingResult.SNAP
                    break

                # Contact detection
                if current_ma >= threshold:
                    print(
                        f"[safe_home] {name}: contact at {raw:.0f} mA "
                        f"(signed: {current_ma:.0f}) ({elapsed:.1f}s)"
                    )
                    result = HomingResult.CONTACT
                    break

                if elapsed >= timeout_s:
                    print(f"[safe_home] {name}: timeout, last current={raw:.0f} mA")
                    result = HomingResult.TIMEOUT
                    break

                if int(elapsed) % 2 == 0 and elapsed > 0.1:
                    print(
                        f"[safe_home]   {name}: {elapsed:.0f}s  "
                        f"current={raw:.0f} mA (signed={current_ma:.0f})",
                        end="\r"
                    )
                time.sleep(POLL_INTERVAL_S)

            print()
            hand.ctrl_torque([0] * 7)
            results[act_id] = result
            time.sleep(0.15)

            if snap_detected:
                break

            # Back off after contact
            if result == HomingResult.CONTACT:
                try:
                    positions = hand.get_actuations()
                    backoff_target = positions[:]
                    backoff_target[act_id] = positions[act_id] + backoff_deg
                    hand.set_actuations(backoff_target)
                    time.sleep(0.3)
                    print(f"[safe_home] {name}: backed off {backoff_deg}° ✓")
                except Exception as e:
                    print(f"[safe_home] {name}: back-off failed: {e}")

        time.sleep(0.2)

    # ── Final state ───────────────────────────────────────────────────────
    hand.ctrl_torque([0] * 7)
    time.sleep(0.3)

    # Seat the position register at open before returning
    try:
        hand.set_joint_positions([0.0] * 16)
        time.sleep(0.5)
        hand.set_joint_positions([0.0] * 16)
        time.sleep(0.3)
    except Exception as e:
        print(f"[safe_home] Warning: could not set open position: {e}")

    print("\n── Homing summary ──────────────────────────────────────")
    for act_id, res in results.items():
        if res != HomingResult.SKIPPED:
            icon = {
                HomingResult.CONTACT:  "✓",
                HomingResult.POSITION: "✓",
                HomingResult.TIMEOUT:  "~",
                HomingResult.SNAP:     "✗ SNAP",
            }.get(res, "?")
            print(f"  [{icon}] {ACTUATOR_NAMES[act_id]}: {res}")
    print()

    if snap_detected:
        print("  ⚠  Snap halt — inspect tendons before continuing.\n")
    else:
        print("[safe_home] Homing complete. Hand is in open-palm position.")
        print("  You can now safely launch the pipeline.\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Safe torque-limited homing for TetherIA Aero Hand Open"
    )
    parser.add_argument("--port", required=True,
                        help="Serial port e.g. COM3 or /dev/ttyACM0")
    parser.add_argument("--actuators", type=int, nargs="+", default=None,
                        help="Actuator IDs to home (default all 0-6). "
                             "E.g.: --actuators 3 4 5 6")
    parser.add_argument("--backoff", type=float, default=BACKOFF_DEG)
    args = parser.parse_args()

    if AeroHand is None:
        print("ERROR: aero_open_sdk not installed.")
        return

    print(f"Connecting to {args.port}…")
    hand = AeroHand(port=args.port)
    print("Connected.\n")

    safe_home(hand, actuators=args.actuators, backoff_deg=args.backoff)


if __name__ == "__main__":
    main()