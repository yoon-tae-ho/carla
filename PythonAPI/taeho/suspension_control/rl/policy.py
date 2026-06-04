"""Optional policy adapter for residual-RL suspension control."""

from __future__ import annotations

import os
import random
from typing import Any, List, Sequence


BUILTIN_POLICY_ALIASES = {
    "dummy_zero": "zero",
    "zero": "zero",
    "builtin_zero": "zero",
    "builtin:zero": "zero",
    "builtin://zero": "zero",
    "const_action_plus_0p02": "const_action_plus_0p02",
    "builtin:const_action_plus_0p02": "const_action_plus_0p02",
    "builtin://const_action_plus_0p02": "const_action_plus_0p02",
    "const_action_minus_0p02": "const_action_minus_0p02",
    "builtin:const_action_minus_0p02": "const_action_minus_0p02",
    "builtin://const_action_minus_0p02": "const_action_minus_0p02",
    "const_action_plus_0p10": "const_action_plus_0p10",
    "builtin:const_action_plus_0p10": "const_action_plus_0p10",
    "builtin://const_action_plus_0p10": "const_action_plus_0p10",
    "const_action_minus_0p10": "const_action_minus_0p10",
    "builtin:const_action_minus_0p10": "const_action_minus_0p10",
    "builtin://const_action_minus_0p10": "const_action_minus_0p10",
    "const_action_plus_0p25": "const_action_plus_0p25",
    "builtin:const_action_plus_0p25": "const_action_plus_0p25",
    "builtin://const_action_plus_0p25": "const_action_plus_0p25",
    "const_action_minus_0p25": "const_action_minus_0p25",
    "builtin:const_action_minus_0p25": "const_action_minus_0p25",
    "builtin://const_action_minus_0p25": "const_action_minus_0p25",
    "random_action_0p02": "random_action_0p02",
    "builtin:random_action_0p02": "random_action_0p02",
    "builtin://random_action_0p02": "random_action_0p02",
    "random_action_0p10": "random_action_0p10",
    "builtin:random_action_0p10": "random_action_0p10",
    "builtin://random_action_0p10": "random_action_0p10",
    "const_plus_0p02": "const_action_plus_0p02",
    "constant_plus_0p02": "const_action_plus_0p02",
    "builtin_const_plus_0p02": "const_action_plus_0p02",
    "builtin:const_plus_0p02": "const_action_plus_0p02",
    "builtin://const_plus_0p02": "const_action_plus_0p02",
    "const_minus_0p02": "const_action_minus_0p02",
    "constant_minus_0p02": "const_action_minus_0p02",
    "builtin_const_minus_0p02": "const_action_minus_0p02",
    "builtin:const_minus_0p02": "const_action_minus_0p02",
    "builtin://const_minus_0p02": "const_action_minus_0p02",
    "random_small": "random_action_0p02",
    "random_small_0p02": "random_action_0p02",
    "builtin_random_small": "random_action_0p02",
    "builtin_random_small_0p02": "random_action_0p02",
    "builtin:random_small": "random_action_0p02",
    "builtin:random_small_0p02": "random_action_0p02",
    "builtin://random_small": "random_action_0p02",
    "builtin://random_small_0p02": "random_action_0p02",
}

DEPRECATED_BUILTIN_POLICY_ALIASES = {
    "const_plus_0p02",
    "constant_plus_0p02",
    "builtin_const_plus_0p02",
    "builtin:const_plus_0p02",
    "builtin://const_plus_0p02",
    "const_minus_0p02",
    "constant_minus_0p02",
    "builtin_const_minus_0p02",
    "builtin:const_minus_0p02",
    "builtin://const_minus_0p02",
    "random_small",
    "random_small_0p02",
    "builtin_random_small",
    "builtin_random_small_0p02",
    "builtin:random_small",
    "builtin:random_small_0p02",
    "builtin://random_small",
    "builtin://random_small_0p02",
}

BUILTIN_CONSTANT_VALUES = {
    "zero": 0.0,
    "const_action_plus_0p02": 0.02,
    "const_action_minus_0p02": -0.02,
    "const_action_plus_0p10": 0.10,
    "const_action_minus_0p10": -0.10,
    "const_action_plus_0p25": 0.25,
    "const_action_minus_0p25": -0.25,
}

BUILTIN_RANDOM_AMPLITUDES = {
    "random_action_0p02": 0.02,
    "random_action_0p10": 0.10,
}


class PolicyAdapter:
    """Small wrapper around optional TorchScript or dummy zero policies."""

    def __init__(
        self,
        policy_path: str = "",
        action_dim: int = 4,
        allow_dummy_zero: bool = False,
    ):
        self.policy_path = os.path.expanduser(policy_path or "")
        self.action_dim = int(action_dim)
        self.allow_dummy_zero = bool(allow_dummy_zero)
        self._module = None
        self._torch = None
        self._dummy_zero = False
        self.builtin_id = ""
        self.alias_deprecated = False
        self._rng = random.Random(0)
        self.status = "unavailable"
        self.error = ""
        self._load()

    @property
    def is_available(self) -> bool:
        return bool(self.builtin_id) or self._module is not None

    def predict(
        self,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> List[float]:
        del deterministic
        if self.builtin_id in BUILTIN_CONSTANT_VALUES:
            value = BUILTIN_CONSTANT_VALUES[self.builtin_id]
            return [value for _ in range(self.action_dim)]
        if self.builtin_id in BUILTIN_RANDOM_AMPLITUDES:
            amplitude = BUILTIN_RANDOM_AMPLITUDES[self.builtin_id]
            return [
                self._rng.uniform(-amplitude, amplitude)
                for _ in range(self.action_dim)
            ]
        if self._module is None or self._torch is None:
            raise RuntimeError(self.error or "policy unavailable")
        torch = self._torch
        with torch.no_grad():
            tensor = torch.as_tensor(
                [list(observation)],
                dtype=torch.float32)
            output = self._module(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            values = output.reshape(-1).detach().cpu().tolist()
        if len(values) != self.action_dim:
            raise RuntimeError(
                "policy_action_shape_mismatch: expected %d values, got %d" %
                (self.action_dim, len(values)))
        return [float(value) for value in values]

    def diagnostics(self) -> dict:
        return {
            "policy_path": self.policy_path,
            "policy_status": self.status,
            "policy_error": self.error,
            "policy_dummy_zero": int(self._dummy_zero),
            "policy_builtin_id": self.builtin_id,
            "policy_alias_deprecated": int(self.alias_deprecated),
        }

    def _load(self) -> None:
        normalized_path = self.policy_path.strip().lower()
        builtin_id = BUILTIN_POLICY_ALIASES.get(normalized_path, "")
        if builtin_id:
            self.builtin_id = builtin_id
            self.alias_deprecated = normalized_path in DEPRECATED_BUILTIN_POLICY_ALIASES
            self._dummy_zero = builtin_id == "zero"
            self.status = (
                "dummy_zero_loaded"
                if builtin_id == "zero"
                else "%s_loaded" % builtin_id)
            return
        if not self.policy_path:
            if self.allow_dummy_zero:
                self.builtin_id = "zero"
                self._dummy_zero = True
                self.status = "dummy_zero"
            else:
                self.status = "missing_path"
                self.error = "policy_path_empty"
            return
        if not os.path.isfile(self.policy_path):
            if self.allow_dummy_zero:
                self.builtin_id = "zero"
                self._dummy_zero = True
                self.status = "dummy_zero_missing_file"
            else:
                self.status = "missing_file"
                self.error = "policy_file_missing"
            return
        try:
            import torch  # type: ignore
        except Exception as error:  # pragma: no cover - depends on environment.
            self.status = "torch_unavailable"
            self.error = str(error)
            return
        self._torch = torch
        try:
            self._module = self._load_torch_module(torch, self.policy_path)
            self._module.eval()
            self.status = "torchscript"
        except Exception as error:  # pragma: no cover - model format dependent.
            self._module = None
            self.status = "load_failed"
            self.error = str(error)

    def _load_torch_module(self, torch: Any, path: str) -> Any:
        suffix = os.path.splitext(path)[1].lower()
        if suffix in (".pt", ".ts", ".jit", ".torchscript"):
            return torch.jit.load(path, map_location="cpu")
        loaded = torch.load(path, map_location="cpu")
        if hasattr(loaded, "eval"):
            return loaded
        raise RuntimeError(
            "unsupported policy format; expected TorchScript module")
