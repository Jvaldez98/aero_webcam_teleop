"""
pipeline.py
===========
Top-level orchestrator — wires together:

  MediaPipeMocap  →  MediaPipeRetargeting  →  AeroHandBridge
                            +
  GraspDetector   →  GraspPlanner          →  GraspPoses → AeroHandBridge

Keys (click the video window first):
  c  — calibrate (hold open palm, then press c)
  h  — home the hand (safe, speed 500/32766)
  o  — open hand / release grasp
  f  — close fist
  g  — trigger one-shot grasp detection
       (press again while HOLDING to release)
  a  — toggle continuous auto-detection
  q  — quit

Usage
-----
  # Mock mode (no hardware)
  python -m aero_webcam_teleop.pipeline

  # With real hardware
  python -m aero_webcam_teleop.pipeline --port COM3

  # Force webcam even if RealSense is connected
  python -m aero_webcam_teleop.pipeline --force-webcam

  # Faster auto-detection
  python -m aero_webcam_teleop.pipeline --auto-detect-s 2.0
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .mediapipe_mocap import HandMocap, MediaPipeMocap
from .mediapipe_retargeting import MediaPipeRetargeting, RetargetingConfig
from .handbridge import AeroHandBridge, NORMAL_SPEED
from .grasp_detector import GraspDetector, Detection
from .grasp_planner import GraspPlanner, GraspPlan
from .grasp_poses import get_pose, get_force, interpolate_angles, GRASP_POSES


# ---------------------------------------------------------------------------
# Grasp mode state machine
# ---------------------------------------------------------------------------

class GraspState:
    IDLE      = "idle"
    DETECTING = "detecting"
    PLANNING  = "planning"
    EXECUTING = "executing"
    HOLDING   = "holding"


class WebcamTeleopPipeline:
    """
    Full pipeline: webcam/D405 → MediaPipe → retargeting → Aero Hand
                 + YOLO → Claude Haiku → grasp pose → Aero Hand

    Parameters
    ----------
    port            : Serial port for the hand (None = mock)
    baudrate        : Serial baud
    side            : "right" | "left"
    camera_index    : Webcam device index (used if RealSense unavailable)
    target_fps      : Camera capture rate (Hz)
    control_hz      : Rate at which commands are sent to the hand (Hz)
    smoothing_alpha : EMA smoothing [0,1]; lower = smoother but laggier
    verbose         : Print joint angles each frame
    force_webcam    : Skip RealSense even if connected
    auto_detect_s   : Interval (s) for continuous auto-detection mode
    """

    # Number of stable MediaPipe frames before sending to hand.
    # Prevents first-detection snap from noisy initial landmark estimates.
    _WARMUP_FRAMES = 5

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 921_600,
        side: str = "right",
        camera_index: int = 0,
        target_fps: float = 30.0,
        control_hz: float = 20.0,
        smoothing_alpha: float = 0.35,
        verbose: bool = False,
        force_webcam: bool = False,
        auto_detect_s: float = 3.0,
    ):
        self.side = side
        self.verbose = verbose
        self.control_hz = control_hz
        self.auto_detect_s = auto_detect_s

        # Shared state — teleop
        self._latest_mocap: Optional[HandMocap] = None
        self._latest_angles: Optional[list] = None
        self._lock = threading.Lock()
        self._calibrate_next = False
        self._running = False
        self._stable_frame_count: int = 0  # frames since last None detection

        # Shared state — grasp
        self._grasp_state = GraspState.IDLE
        self._grasp_plan: Optional[GraspPlan] = None
        self._grasp_start_angles: Optional[list] = None
        self._grasp_target_angles: Optional[list] = None
        self._grasp_exec_start: float = 0.0
        self._grasp_exec_duration: float = 1.5
        self._frozen_frame: Optional[np.ndarray] = None
        self._frozen_depth: Optional[np.ndarray] = None
        self._auto_detect = False
        self._last_auto_t = 0.0

        # OSD
        self._fps_counter = 0
        self._fps_last_t = time.monotonic()
        self._fps_display = 0.0

        # Components
        self.retargeting = MediaPipeRetargeting(
            config=RetargetingConfig(),
            smoothing_alpha=smoothing_alpha,
        )
        self.bridge = AeroHandBridge(
            port=port,
            baudrate=baudrate,
            side=side,
            verbose=verbose,
        )
        self.mocap = MediaPipeMocap(
            camera_index=camera_index,
            target_fps=target_fps,
            hand_side=side,
            on_mocap=self._on_mocap,
            on_frame=self._on_frame,
            show_window=True,
            force_webcam=force_webcam,
        )
        self.detector = GraspDetector(model_size="n", confidence=0.25)
        self.planner = GraspPlanner(
            on_plan=self._on_grasp_plan,
            cooldown_s=2.0,
        )

    # ------------------------------------------------------------------
    # Teleop callbacks
    # ------------------------------------------------------------------

    def _on_mocap(self, mocap: HandMocap):
        if self._calibrate_next:
            self.retargeting.calibrate(mocap)
            self._calibrate_next = False

        angles = self.retargeting.retarget(mocap)

        with self._lock:
            self._stable_frame_count += 1
            self._latest_mocap = mocap
            # Only expose angles after warmup AND only during IDLE
            # (don't let teleop override an active grasp execution)
            if (self._stable_frame_count >= self._WARMUP_FRAMES
                    and self._grasp_state == GraspState.IDLE):
                self._latest_angles = angles

    def _on_frame(self, frame: np.ndarray):
        """Called each frame with the annotated BGR image from the camera."""
        with self._lock:
            angles      = self._latest_angles
            grasp_state = self._grasp_state
            grasp_plan  = self._grasp_plan
            frozen      = self._frozen_frame is not None

        h, w = frame.shape[:2]

        # FPS counter
        self._fps_counter += 1
        now = time.monotonic()
        if now - self._fps_last_t >= 1.0:
            self._fps_display = self._fps_counter / (now - self._fps_last_t)
            self._fps_counter = 0
            self._fps_last_t = now

        # Auto-detection trigger
        if self._auto_detect and grasp_state == GraspState.IDLE:
            if now - self._last_auto_t >= self.auto_detect_s:
                self._last_auto_t = now
                self._trigger_detection(frame)

        # ---- Status panel ------------------------------------------------
        cv2.rectangle(frame, (0, 0), (w, 175), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 175), (30, 30, 30), 1)

        cv2.putText(frame, "Aero Hand — Webcam Teleop + Grasp AI",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.putText(frame,
                    "c=cal  h=home  o=open  f=fist  g=grasp(1shot)  a=auto  q=quit",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        hw_label   = f"[{self.side.upper()}] {'REAL' if not self.bridge._mock else 'MOCK'}"
        auto_label = "AUTO-ON" if self._auto_detect else "AUTO-OFF"
        llm_label  = "LLM:Claude" if self.planner._client else "LLM:FALLBACK"
        cv2.putText(frame,
                    f"{hw_label}  FPS:{self._fps_display:.1f}  {auto_label}  {llm_label}",
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 100), 1)

        # Grasp state indicator
        state_colors = {
            GraspState.IDLE:      (120, 120, 120),
            GraspState.DETECTING: (0,   200, 255),
            GraspState.PLANNING:  (0,   160, 255),
            GraspState.EXECUTING: (0,   255, 120),
            GraspState.HOLDING:   (255, 200,   0),
        }
        state_color = state_colors.get(grasp_state, (120, 120, 120))
        cv2.putText(frame, f"GRASP: {grasp_state.upper()}",
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1)

        # Grasp plan overlay
        if grasp_plan:
            src_tag = "[LLM]" if grasp_plan.source == "llm" else "[FBK]"
            cv2.putText(frame,
                        f"{src_tag} {grasp_plan.grasp_type} / {grasp_plan.force_level}",
                        (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 50), 1)
            reason = (grasp_plan.reasoning[:72] + "..."
                      if len(grasp_plan.reasoning) > 72 else grasp_plan.reasoning)
            cv2.putText(frame, reason,
                        (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)

        # Finger bars
        if angles:
            compact = [
                angles[0],
                max(angles[1], angles[2], angles[3]),
                angles[4], angles[7], angles[10], angles[13],
            ]
            maxes  = [100, 90, 90, 90, 90, 90]
            labels = ["TH-ABD", "TH-FLX", "Index", "Middle", "Ring", "Pinky"]
            bar_w  = (w - 20) // len(labels)
            for i, (val, mx, lbl) in enumerate(zip(compact, maxes, labels)):
                bx   = 10 + i * bar_w
                by   = 148
                fill = int((val / mx) * (bar_w - 6))
                cv2.rectangle(frame, (bx, by), (bx + bar_w - 4, by + 18), (60, 60, 60), -1)
                cv2.rectangle(frame, (bx, by), (bx + fill,       by + 18), (0, 200, 100), -1)
                cv2.putText(frame, lbl[:6],
                            (bx + 2, by + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (255, 255, 255), 1)

        # Frozen frame indicator
        if frozen:
            cv2.putText(frame, "[ FRAME FROZEN ]",
                        (w // 2 - 70, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # ------------------------------------------------------------------
    # Grasp detection + planning
    # ------------------------------------------------------------------

    def _trigger_detection(self, frame: np.ndarray):
        """Freeze the frame and kick off YOLO + LLM in a background thread."""
        with self._lock:
            if self._grasp_state != GraspState.IDLE:
                return
            self._grasp_state = GraspState.DETECTING
            self._frozen_frame = frame.copy()

        threading.Thread(target=self._run_detection, daemon=True).start()

    def _run_detection(self):
        """YOLO detection — runs in background thread."""
        with self._lock:
            frame = self._frozen_frame
            depth = self._frozen_depth

        if frame is None:
            with self._lock:
                self._grasp_state = GraspState.IDLE
            return

        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect(frame_rgb, depth_frame=depth)

        if not detections:
            print("[Pipeline] No objects detected — returning to teleop.")
            with self._lock:
                self._grasp_state  = GraspState.IDLE
                self._frozen_frame = None
            return

        top = detections[0]
        print(
            f"[Pipeline] Detected: {top.label} (YOLO:{top.yolo_label}) "
            f"({top.confidence:.0%})"
            + (f" size={top.size_cm:.1f}cm" if top.size_cm else "")
            + (f" depth={top.depth_m:.2f}m"  if top.depth_m else "")
        )

        with self._lock:
            self._grasp_state = GraspState.PLANNING

        self.planner.request_plan(
            object_label=top.label,
            size_cm=top.size_cm,
            depth_m=top.depth_m,
            confidence=top.confidence,
        )

    def _on_grasp_plan(self, plan: GraspPlan):
        """Called by GraspPlanner when LLM (or fallback) responds."""
        print(
            f"[Pipeline] Grasp plan: {plan.grasp_type} / {plan.force_level} "
            f"[{plan.source}] — {plan.reasoning}"
        )

        pose  = get_pose(plan.grasp_type)
        force = get_force(plan.force_level)

        target_angles = pose.angles[:]
        target_speed  = int(force.speed_scalar * NORMAL_SPEED)

        self.bridge.set_teleop_speed(target_speed)

        with self._lock:
            current = self._latest_angles or [0.0] * 16
            self._grasp_plan          = plan
            self._grasp_start_angles  = current[:]
            self._grasp_target_angles = target_angles
            self._grasp_exec_start    = time.monotonic()
            self._grasp_state         = GraspState.EXECUTING

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        interval = 1.0 / self.control_hz
        _prev_had_angles = False

        while self._running:
            t0 = time.monotonic()

            with self._lock:
                angles      = self._latest_angles
                grasp_state = self._grasp_state
                start_a     = self._grasp_start_angles
                target_a    = self._grasp_target_angles
                exec_start  = self._grasp_exec_start

                # Reset warmup when hand leaves frame so re-entry also
                # gets a clean warmup period before sending commands
                if angles is None and _prev_had_angles:
                    self._stable_frame_count = 0

            _prev_had_angles = angles is not None

            if grasp_state == GraspState.EXECUTING:
                elapsed = time.monotonic() - exec_start
                t       = min(elapsed / self._grasp_exec_duration, 1.0)
                interp  = interpolate_angles(start_a, target_a, t)
                self.bridge.send_joint_positions(interp)

                if t >= 1.0:
                    with self._lock:
                        self._grasp_state  = GraspState.HOLDING
                        self._frozen_frame = None
                    print("[Pipeline] Grasp pose reached — holding.")

            elif grasp_state == GraspState.HOLDING:
                if target_a:
                    self.bridge.send_joint_positions(target_a)

            elif grasp_state == GraspState.IDLE and angles is not None:
                self.bridge.send_joint_positions(angles)
                if self.verbose:
                    print(f"[control_loop] {[f'{a:.1f}' for a in angles]}")

            elapsed = time.monotonic() - t0
            sleep   = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    # ------------------------------------------------------------------
    # Key handler
    # ------------------------------------------------------------------

    def _handle_key(self, key: int):
        if key in (ord("q"), 27):
            self.mocap.stop()

        elif key == ord("c"):
            print("[Pipeline] Calibration armed — show open palm.")
            self._calibrate_next = True

        elif key == ord("h"):
            print("[Pipeline] Homing…")
            threading.Thread(target=self.bridge.home, daemon=True).start()

        elif key == ord("o"):
            print("[Pipeline] Opening hand / releasing grasp.")
            with self._lock:
                self._grasp_state  = GraspState.IDLE
                self._grasp_plan   = None
                self._frozen_frame = None
            self.bridge.open_hand()
            with self._lock:
                self._latest_angles = [0.0] * 16
            self.bridge.set_teleop_speed(NORMAL_SPEED)

        elif key == ord("f"):
            print("[Pipeline] Closing fist.")
            with self._lock:
                self._grasp_state = GraspState.IDLE
            self.bridge.close_hand()

        elif key == ord("g"):
            with self._lock:
                state = self._grasp_state
            if state == GraspState.HOLDING:
                print("[Pipeline] Releasing grasp — returning to teleop.")
                with self._lock:
                    self._grasp_state = GraspState.IDLE
                    self._grasp_plan  = None
                self.bridge.open_hand()
                self.bridge.set_teleop_speed(NORMAL_SPEED)
            elif state == GraspState.IDLE:
                print("[Pipeline] Grasp mode triggered — detecting objects…")
                with self._lock:
                    self._grasp_state = GraspState.DETECTING
                threading.Thread(
                    target=self._run_detection_from_live, daemon=True
                ).start()

        elif key == ord("a"):
            self._auto_detect = not self._auto_detect
            self._last_auto_t = 0.0
            print(
                f"[Pipeline] Auto-detect "
                f"{'ENABLED' if self._auto_detect else 'DISABLED'} "
                f"(every {self.auto_detect_s}s)"
            )

    def _run_detection_from_live(self):
        """Wait for a fresh frame to be stored then run detection."""
        time.sleep(0.1)
        with self._lock:
            frame = self._frozen_frame
            if frame is None:
                self._grasp_state = GraspState.IDLE
                return
        self._run_detection()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        self._override_keycheck()

        # Send open-hand position before starting the control loop.
        # Sets the servo position register to open-palm so the first
        # webcam frame doesn't snap the hand to closed.
        if not self.bridge._mock:
            print("[Pipeline] Initialising hand to open-palm position…")
            self.bridge.open_hand()
            time.sleep(0.8)

        ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        ctrl_thread.start()

        print("\n" + "=" * 62)
        print("  Aero Hand — Webcam Teleop + Grasp AI Pipeline")
        print("=" * 62)
        print(f"  Side    : {self.side}")
        print(f"  Hardware: {'REAL' if not self.bridge._mock else 'MOCK (no SDK / port)'}")
        print(f"  Control : {self.control_hz} Hz")
        print(f"  LLM     : {'Claude Haiku' if self.planner._client else 'FALLBACK MODE (add API key to .env)'}")
        print()
        print("  Keys (click the video window first):")
        print("    c  — calibrate (open palm reference)")
        print("    h  — home the hand (safe, speed 500/32766)")
        print("    o  — open hand / release grasp")
        print("    f  — close fist")
        print("    g  — trigger one-shot grasp detection")
        print("         (press again while HOLDING to release)")
        print("    a  — toggle continuous auto-detection")
        print("    q  — quit")
        print("=" * 62 + "\n")

        try:
            self.mocap.start()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.bridge.close()

    def _override_keycheck(self):
        """Route key presses through our handler and control imshow."""
        pipeline_self     = self
        original_on_frame = self.mocap.on_frame
        self.mocap.show_window = False

        def full_frame_handler(frame: np.ndarray):
            with pipeline_self._lock:
                if pipeline_self._grasp_state == GraspState.DETECTING:
                    pipeline_self._frozen_frame = frame.copy()

            if original_on_frame:
                original_on_frame(frame)

            cv2.imshow("Aero Hand - Webcam Teleop", frame)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                pipeline_self._handle_key(key)

        self.mocap.on_frame = full_frame_handler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TetherIA Aero Hand — Webcam Teleop + Grasp AI Pipeline"
    )
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (e.g. COM3). Omit for mock mode.")
    parser.add_argument("--baudrate",      type=int,   default=921_600)
    parser.add_argument("--side",          choices=["right", "left"], default="right")
    parser.add_argument("--camera",        type=int,   default=0,
                        help="Webcam index (used if RealSense unavailable)")
    parser.add_argument("--fps",           type=float, default=30.0)
    parser.add_argument("--control-hz",    type=float, default=20.0)
    parser.add_argument("--smoothing",     type=float, default=0.35)
    parser.add_argument("--verbose",       action="store_true")
    parser.add_argument("--force-webcam",  action="store_true",
                        help="Force webcam even if RealSense is connected")
    parser.add_argument("--auto-detect-s", type=float, default=3.0,
                        help="Seconds between auto-detections in auto mode")
    args = parser.parse_args()

    pipeline = WebcamTeleopPipeline(
        port=args.port,
        baudrate=args.baudrate,
        side=args.side,
        camera_index=args.camera,
        target_fps=args.fps,
        control_hz=args.control_hz,
        smoothing_alpha=args.smoothing,
        verbose=args.verbose,
        force_webcam=args.force_webcam,
        auto_detect_s=args.auto_detect_s,
    )
    pipeline.run()


if __name__ == "__main__":
    main()