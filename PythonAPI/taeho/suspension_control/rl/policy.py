"""Optional policy adapter for residual-RL suspension control."""

from __future__ import annotations

import os
from typing import Any, List, Sequence


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
        self.status = "unavailable"
        self.error = ""
        self._load()

    @property
    def is_available(self) -> bool:
        return self._dummy_zero or self._module is not None

    def predict(
        self,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> List[float]:
        del deterministic
        if self._dummy_zero:
            return [0.0 for _ in range(self.action_dim)]
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
        return [float(value) for value in values]

    def diagnostics(self) -> dict:
        return {
            "policy_path": self.policy_path,
            "policy_status": self.status,
            "policy_error": self.error,
            "policy_dummy_zero": int(self._dummy_zero),
        }

    def _load(self) -> None:
        if not self.policy_path:
            if self.allow_dummy_zero:
                self._dummy_zero = True
                self.status = "dummy_zero"
            else:
                self.status = "missing_path"
                self.error = "policy_path_empty"
            return
        if not os.path.isfile(self.policy_path):
            if self.allow_dummy_zero:
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
