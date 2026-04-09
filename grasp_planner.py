"""
grasp_planner.py
================
LLM-based grasp planner using Claude Haiku via the Anthropic API.

Given a detected object and optional depth/size info, asks Claude
to reason about the best grasp type and force level, returning a
structured plan that maps directly to the Aero Hand pose library.

API key is loaded from a .env file in the project root:
    ANTHROPIC_API_KEY=sk-ant-...

Install dependencies:
    pip install anthropic python-dotenv

Grasp types available (must match grasp_poses.py):
    power, pinch, tripod, lateral_pinch, palmar, open

Force levels:
    light, medium, firm
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    print("[GraspPlanner] python-dotenv not found. Run: pip install python-dotenv")

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    print("[GraspPlanner] anthropic not found. Run: pip install anthropic")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GraspPlan:
    object_label: str
    grasp_type: str          # matches key in GRASP_POSES
    force_level: str         # "light" | "medium" | "firm"
    reasoning: str           # one-sentence explanation from LLM
    source: str              # "llm" | "fallback"
    latency_ms: float = 0.0  # time taken for LLM call


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a robotic hand grasp planner for the TetherIA Aero Hand,
a 5-finger anthropomorphic robotic hand with 16 degrees of freedom.

Your job: given an object description, return the optimal grasp configuration
as valid JSON — no preamble, no markdown, just the JSON object.

Available grasp types:
- "power"         : Full hand wrap. Best for cylinders, large spheres, large prisms.
- "pinch"         : Index tip + thumb tip. Best for small objects, pens, coins.
- "tripod"        : Thumb + index + middle tips. Best for cubes, medium objects.
- "lateral_pinch" : Thumb presses side of index. Best for flat/thin objects.
- "palmar"        : Object rests in palm, all fingers close. Best for soft/squish objects.
- "open"          : Fully open hand. Use only for approach or release.

Force levels:
- "light"  : Soft/fragile/compliant objects (squish ball, foam, paper).
- "medium" : Standard rigid everyday objects (wood, plastic).
- "firm"   : Heavy, slippery, or dense objects (metal, glass, wet surfaces).

Return ONLY this JSON structure:
{
  "grasp_type": "<one of the types above>",
  "force_level": "<light|medium|firm>",
  "reasoning": "<one sentence explaining why>"
}"""


# ---------------------------------------------------------------------------
# Planner class
# ---------------------------------------------------------------------------

class GraspPlanner:
    """
    Async LLM grasp planner. Calls Claude Haiku in a background thread
    so it never blocks the 20Hz control loop.

    Parameters
    ----------
    on_plan     : Callback called with GraspPlan when LLM responds
    cooldown_s  : Minimum seconds between LLM calls (rate limiting)
    """

    VALID_GRASPS = {"power", "pinch", "tripod", "lateral_pinch", "palmar", "open"}
    VALID_FORCES = {"light", "medium", "firm"}

    # Fallback plans per object (used if LLM fails or is unavailable)
    FALLBACKS = {
        "sphere":            GraspPlan("sphere",            "power",         "medium", "Spherical — power wrap",          "fallback"),
        "wooden_sphere":     GraspPlan("wooden_sphere",     "power",         "firm",   "Rigid sphere — firm power wrap",  "fallback"),
        "cylinder":          GraspPlan("cylinder",          "power",         "medium", "Cylindrical — power wrap",        "fallback"),
        "cube":              GraspPlan("cube",              "tripod",        "medium", "Cubic — tripod on flat faces",    "fallback"),
        "rectangular_prism": GraspPlan("rectangular_prism", "lateral_pinch", "medium", "Flat object — lateral pinch",    "fallback"),
        "squish_ball":       GraspPlan("squish_ball",       "palmar",        "light",  "Soft ball — gentle palmar",       "fallback"),
        "unknown":           GraspPlan("unknown",           "tripod",        "medium", "Unknown — safe tripod default",   "fallback"),
    }

    def __init__(
        self,
        on_plan: Optional[Callable[[GraspPlan], None]] = None,
        cooldown_s: float = 2.0,
    ):
        self.on_plan = on_plan
        self.cooldown_s = cooldown_s
        self._last_call_t = 0.0
        self._pending = False

        self._client = None
        if _ANTHROPIC_AVAILABLE:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
                print("[GraspPlanner] Claude Haiku ready.")
            else:
                print("[GraspPlanner] ANTHROPIC_API_KEY not found in environment/.env")
                print("[GraspPlanner] Running in fallback-only mode.")
        else:
            print("[GraspPlanner] Running in fallback-only mode (no anthropic package).")

    def request_plan(
        self,
        object_label: str,
        size_cm: Optional[float] = None,
        depth_m: Optional[float] = None,
        confidence: float = 1.0,
    ):
        """
        Request a grasp plan asynchronously. Returns immediately.
        The on_plan callback is called when the plan is ready.

        Respects cooldown to avoid hammering the API.
        """
        now = time.monotonic()
        if self._pending:
            return  # already waiting on a response
        if now - self._last_call_t < self.cooldown_s:
            return  # cooldown not elapsed

        self._pending = True
        self._last_call_t = now

        thread = threading.Thread(
            target=self._call_llm,
            args=(object_label, size_cm, depth_m, confidence),
            daemon=True,
        )
        thread.start()

    def _build_prompt(
        self,
        object_label: str,
        size_cm: Optional[float],
        depth_m: Optional[float],
        confidence: float,
    ) -> str:
        parts = [f"Object: {object_label.replace('_', ' ')}"]
        parts.append(f"Detection confidence: {confidence:.0%}")
        if size_cm is not None:
            parts.append(f"Estimated size: {size_cm:.1f} cm")
        if depth_m is not None:
            parts.append(f"Distance from camera: {depth_m:.2f} m")
        parts.append(
            "Determine the best grasp type and force level for "
            "the TetherIA Aero Hand to pick up this object."
        )
        return "\n".join(parts)

    def _call_llm(
        self,
        object_label: str,
        size_cm: Optional[float],
        depth_m: Optional[float],
        confidence: float,
    ):
        t0 = time.monotonic()
        plan = None

        try:
            if self._client is not None:
                prompt = self._build_prompt(object_label, size_cm, depth_m, confidence)

                response = self._client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=256,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )

                raw = response.content[0].text.strip()

                # Strip markdown fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                data = json.loads(raw)

                grasp_type  = data.get("grasp_type",  "tripod")
                force_level = data.get("force_level", "medium")
                reasoning   = data.get("reasoning",   "")

                # Validate — fall back to safe defaults if LLM hallucinated
                if grasp_type not in self.VALID_GRASPS:
                    print(f"[GraspPlanner] Invalid grasp_type '{grasp_type}' — using tripod")
                    grasp_type = "tripod"
                if force_level not in self.VALID_FORCES:
                    print(f"[GraspPlanner] Invalid force_level '{force_level}' — using medium")
                    force_level = "medium"

                latency_ms = (time.monotonic() - t0) * 1000
                plan = GraspPlan(
                    object_label=object_label,
                    grasp_type=grasp_type,
                    force_level=force_level,
                    reasoning=reasoning,
                    source="llm",
                    latency_ms=latency_ms,
                )
                print(
                    f"[GraspPlanner] LLM plan: {grasp_type} / {force_level} "
                    f"({latency_ms:.0f}ms) — {reasoning}"
                )

        except Exception as exc:
            print(f"[GraspPlanner] LLM call failed: {exc} — using fallback")

        finally:
            self._pending = False

        # Use fallback if LLM failed
        if plan is None:
            plan = self.FALLBACKS.get(
                object_label.lower().replace(" ", "_"),
                self.FALLBACKS["unknown"],
            )

        if self.on_plan:
            self.on_plan(plan)

    @property
    def is_pending(self) -> bool:
        return self._pending