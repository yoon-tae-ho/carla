"""Observation normalization utilities for residual-RL suspension control."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_FEATURE_SCALES: Dict[str, float] = {
    "speed": 30.0,
    "local_vx": 30.0,
    "local_vy": 10.0,
    "vz": 5.0,
    "local_ax": 10.0,
    "local_ay": 10.0,
    "az": 10.0,
    "jerk_x": 50.0,
    "jerk_y": 50.0,
    "jerk_z": 50.0,
    "roll": 10.0,
    "pitch": 10.0,
    "yaw_rate": 60.0,
    "roll_rate": 60.0,
    "pitch_rate": 60.0,
    "body_activity": 1.0,
    "curvature": 0.20,
    "steer": 1.0,
    "throttle": 1.0,
    "brake": 1.0,
    "damper": 1.0,
    "spring": 1.0,
    "action": 1.0,
    "planning_available": 1.0,
    "time_to": 2.0,
}


class FixedScaleNormalizer:
    """Normalize named features with robust fixed scales or JSON statistics."""

    def __init__(
        self,
        normalizer_path: str = "",
        clip: float = 5.0,
        default_scales: Optional[Mapping[str, float]] = None,
    ):
        self.normalizer_path = os.path.expanduser(normalizer_path or "")
        self.clip = float(clip)
        self.default_scales = dict(default_scales or DEFAULT_FEATURE_SCALES)
        self.mean_by_name: Dict[str, float] = {}
        self.std_by_name: Dict[str, float] = {}
        self.load_error = ""
        self.loaded = False
        if self.normalizer_path:
            self._load_json(self.normalizer_path)

    def normalize(self, name: str, value: Any) -> float:
        raw = float(value)
        if name in self.mean_by_name and name in self.std_by_name:
            std = max(abs(self.std_by_name[name]), 1.0e-9)
            return (raw - self.mean_by_name[name]) / std
        return raw / max(abs(self.scale_for_name(name)), 1.0e-9)

    def clip_value(self, value: float) -> float:
        if self.clip <= 0.0:
            return value
        return max(-self.clip, min(self.clip, value))

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "normalizer_loaded": int(self.loaded),
            "normalizer_path": self.normalizer_path,
            "normalizer_error": self.load_error,
        }

    def scale_for_name(self, name: str) -> float:
        if name in self.default_scales:
            return self.default_scales[name]
        for token, scale in self.default_scales.items():
            if token and token in name:
                return scale
        return 1.0

    def _load_json(self, path: str) -> None:
        if not os.path.isfile(path):
            self.load_error = "normalizer_missing"
            return
        try:
            with open(path) as json_file:
                data = json.load(json_file)
            self._load_stats(data)
            self.loaded = bool(self.mean_by_name and self.std_by_name)
        except Exception as error:  # pragma: no cover - exact exception is not important.
            self.load_error = str(error)
            self.mean_by_name = {}
            self.std_by_name = {}
            self.loaded = False

    def _load_stats(self, data: Mapping[str, Any]) -> None:
        names = list(data.get("feature_names") or [])
        mean = data.get("mean", {})
        std = data.get("std", {})
        if isinstance(mean, Mapping):
            self.mean_by_name = {
                str(key): float(value)
                for key, value in mean.items()
                if _finite_number(value)
            }
        elif names:
            self.mean_by_name = {
                str(name): float(value)
                for name, value in zip(names, mean)
                if _finite_number(value)
            }
        if isinstance(std, Mapping):
            self.std_by_name = {
                str(key): float(value)
                for key, value in std.items()
                if _finite_number(value)
            }
        elif names:
            self.std_by_name = {
                str(name): float(value)
                for name, value in zip(names, std)
                if _finite_number(value)
            }


def normalizer_metadata(
    feature_names: Sequence[str],
    clip: float,
    action_semantics: str,
    max_damper_residual_scale: float,
) -> Dict[str, Any]:
    """Return deploy-time metadata for fixed-scale observation normalization."""

    names = [str(name) for name in feature_names]
    scales = {
        name: _scale_for_feature_name(name)
        for name in names
    }
    return {
        "normalizer_type": "fixed_scale",
        "source": "fixed_scale_phase4",
        "feature_names": names,
        "feature_scales": scales,
        "clip": float(clip),
        "observation_clip": float(clip),
        "mean": {
            name: 0.0
            for name in names
        },
        "std": scales,
        "action_semantics": str(action_semantics),
        "max_damper_residual_scale": float(max_damper_residual_scale),
        "spring_frozen": True,
    }


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _scale_for_feature_name(name: str) -> float:
    if name in DEFAULT_FEATURE_SCALES:
        return DEFAULT_FEATURE_SCALES[name]
    for token, scale in DEFAULT_FEATURE_SCALES.items():
        if token and token in name:
            return scale
    return 1.0
