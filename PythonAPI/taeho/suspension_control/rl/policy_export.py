"""Policy export helpers for residual-RL suspension control.

The module is importable without torch or stable-baselines3. Heavy dependencies
are imported only inside the functions that need them.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from .normalizer import normalizer_metadata


def export_sb3_actor_to_torchscript(
    model_path: str,
    output_path: str,
    observation_dim: int,
    action_dim: int = 4,
) -> str:
    """Load an SB3 SAC/TD3 model and export its actor to TorchScript."""

    try:
        from stable_baselines3 import SAC, TD3  # type: ignore
    except Exception as error:  # pragma: no cover - dependency-dependent.
        raise RuntimeError(
            "stable-baselines3 is required to export SB3 policy: %s" % error)

    last_error: Exception = RuntimeError("no SB3 loader attempted")
    for model_cls in (SAC, TD3):
        try:
            model = model_cls.load(os.path.expanduser(model_path), device="cpu")
            actor = getattr(model.policy, "actor", None)
            if actor is None:
                actor = model.policy
            return export_torch_module(
                actor,
                output_path=output_path,
                observation_dim=observation_dim,
                action_dim=action_dim)
        except Exception as error:  # pragma: no cover - model-format-dependent.
            last_error = error
    raise RuntimeError(
        "could not load SB3 policy from %s: %s" % (model_path, last_error))


def export_torch_module(
    module: Any,
    output_path: str,
    observation_dim: int,
    action_dim: int = 4,
) -> str:
    """Export a torch module as a bounded TorchScript actor.

    The saved actor accepts `(obs_dim,)` or `(N, obs_dim)` tensors and returns
    actions bounded to `[-1, 1]`. If scripting fails for a third-party actor,
    the function falls back to tracing with a 2-D example input, which is still
    compatible with `PolicyAdapter`.
    """

    try:
        import torch  # type: ignore
    except Exception as error:  # pragma: no cover - dependency-dependent.
        raise RuntimeError("torch is required to export TorchScript policy: %s" % error)
    _install_torch_wrappers()

    observation_dim = int(observation_dim)
    action_dim = int(action_dim)
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if not hasattr(module, "eval"):
        raise TypeError("module must be a torch.nn.Module-like object")

    module.eval()
    wrapper = _BoundedActionScriptWrapper(module, action_dim)
    try:
        scripted = torch.jit.script(wrapper)
    except Exception:
        trace_wrapper = _BoundedActionTraceWrapper(module, action_dim)
        example = torch.zeros((1, observation_dim), dtype=torch.float32)
        scripted = torch.jit.trace(trace_wrapper, example, check_trace=False)

    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scripted.save(output_path)
    return output_path


def write_normalizer_metadata(
    output_path: str,
    feature_names: Sequence[str],
    clip: float,
    action_semantics: str,
    max_damper_residual_scale: float,
) -> str:
    """Write deploy-time normalizer metadata next to an exported policy."""

    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata = normalizer_metadata(
        feature_names=feature_names,
        clip=clip,
        action_semantics=action_semantics,
        max_damper_residual_scale=max_damper_residual_scale)
    with open(output_path, "w") as json_file:
        json.dump(metadata, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    return output_path


class _BoundedActionScriptWrapper:
    pass


class _BoundedActionTraceWrapper:
    pass


def _install_torch_wrappers() -> None:
    """Define torch wrappers lazily so importing this module needs no torch."""

    global _BoundedActionScriptWrapper, _BoundedActionTraceWrapper
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - handled by caller.
        return

    class BoundedActionScriptWrapper(torch.nn.Module):  # type: ignore
        def __init__(self, actor: Any, action_dim: int):
            super().__init__()
            self.actor = actor
            self.action_dim = int(action_dim)

        def forward(self, observation):  # type: ignore
            one_dim = observation.dim() == 1
            obs = observation.unsqueeze(0) if one_dim else observation
            output = self.actor(obs)
            output = torch.tanh(output)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            output = output.reshape((output.size(0), -1))
            if output.size(1) < self.action_dim:
                padding = torch.zeros(
                    (output.size(0), self.action_dim - output.size(1)),
                    dtype=output.dtype,
                    device=output.device)
                output = torch.cat((output, padding), dim=1)
            elif output.size(1) > self.action_dim:
                output = output[:, :self.action_dim]
            if one_dim:
                return output.squeeze(0)
            return output

    class BoundedActionTraceWrapper(torch.nn.Module):  # type: ignore
        def __init__(self, actor: Any, action_dim: int):
            super().__init__()
            self.actor = actor
            self.action_dim = int(action_dim)

        def forward(self, observation):  # type: ignore
            one_dim = observation.dim() == 1
            obs = observation.unsqueeze(0) if one_dim else observation
            output = self.actor(obs)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = torch.tanh(output)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            output = output.reshape((output.size(0), -1))
            if output.size(1) < self.action_dim:
                padding = torch.zeros(
                    (output.size(0), self.action_dim - output.size(1)),
                    dtype=output.dtype,
                    device=output.device)
                output = torch.cat((output, padding), dim=1)
            elif output.size(1) > self.action_dim:
                output = output[:, :self.action_dim]
            if one_dim:
                return output.squeeze(0)
            return output

    _BoundedActionScriptWrapper = BoundedActionScriptWrapper
    _BoundedActionTraceWrapper = BoundedActionTraceWrapper
