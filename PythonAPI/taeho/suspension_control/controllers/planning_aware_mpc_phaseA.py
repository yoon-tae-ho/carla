"""Planning-aware v4 Phase A projection-aware MPC-lite controller."""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import ControllerContext, ControllerOutput
from .planning_aware_mpc_primary import (
    ComfortGuardResult,
    MpcDamperCandidate,
    PlanningAwareV3MpcPrimaryConfig,
    PlanningAwareV3MpcPrimaryController,
    PlanningAwareV3Preview,
    _WHEEL_LABELS,
    _all_finite,
    _match_length,
    _safe_float,
    build_planning_preview,
)


PLANNING_AWARE_V4_MPC_PHASEA_VERSION = (
    "planning_aware_v4_mpc_phaseA_authority_projection_counterfactual_v1")


PLANNING_AWARE_V4_PHASEA_DIAGNOSTIC_FIELDS = (
    "v4_phaseA_enabled",
    "v4_raw_candidate_count",
    "v4_raw_feasible_candidate_count",
    "v4_rerank_candidate_count",
    "v4_rerank_top_k",
    "v4_previous_candidate_included",
    "v4_neutral_candidate_included",
    "v4_raw_best_cost",
    "v4_raw_second_best_cost",
    "v4_raw_cost_margin",
    "v4_projected_best_cost",
    "v4_projected_second_best_cost",
    "v4_projected_cost_margin",
    "v4_previous_projected_cost",
    "v4_neutral_projected_cost",
    "v4_raw_winner_projected_cost",
    "v4_raw_winner_kind",
    "v4_projected_winner_kind",
    "v4_selected_candidate_kind",
    "v4_projection_changed_winner",
    "v4_selected_from_previous",
    "v4_selected_from_neutral",
    "v4_selected_from_mpc_grid",
    "v4_raw_to_constrained_l2",
    "v4_constrained_to_projected_l2",
    "v4_projected_to_final_l2",
    "v4_raw_to_final_l2",
    "v4_final_to_previous_l2",
    "v4_final_to_neutral_l2",
    "v4_raw_to_final_fl",
    "v4_raw_to_final_fr",
    "v4_raw_to_final_rl",
    "v4_raw_to_final_rr",
    "v4_projected_to_final_fl",
    "v4_projected_to_final_fr",
    "v4_projected_to_final_rl",
    "v4_projected_to_final_rr",
    "v4_selected_raw_damper_fl",
    "v4_selected_raw_damper_fr",
    "v4_selected_raw_damper_rl",
    "v4_selected_raw_damper_rr",
    "v4_selected_projected_damper_fl",
    "v4_selected_projected_damper_fr",
    "v4_selected_projected_damper_rl",
    "v4_selected_projected_damper_rr",
    "v4_selected_final_damper_fl",
    "v4_selected_final_damper_fr",
    "v4_selected_final_damper_rl",
    "v4_selected_final_damper_rr",
    "controller_compute_ms",
    "controller_compute_ms_raw_optimizer",
    "controller_compute_ms_projection_rerank",
    "controller_compute_ms_projection",
    "controller_compute_ms_final_guard",
    "debug_compute_sleep_ms",
    "planning_aware_v4_mode",
    "planning_aware_v4_fallback_reason",
)


@dataclass(frozen=True)
class PlanningAwareV4MpcPhaseAConfig(PlanningAwareV3MpcPrimaryConfig):
    """v4 Phase A config using the v3 authority profile as its base."""

    controller_version: str = PLANNING_AWARE_V4_MPC_PHASEA_VERSION
    base_profile: str = "authority"

    min_damper_scale: float = 0.75
    max_damper_scale: float = 1.25
    max_rate_up_scale_per_s: float = 2.0
    max_rate_down_scale_per_s: float = 1.6
    spring_scale: float = 1.0

    projection_aware_rerank_enabled: bool = True
    projection_rerank_top_k: int = 16
    projection_rerank_include_previous: bool = True
    projection_rerank_include_neutral: bool = True
    projection_rerank_include_raw_winner: bool = True
    projection_rerank_include_second_best: bool = True

    previous_candidate_enabled: bool = True
    neutral_candidate_enabled: bool = True

    comfort_margin_v4_enabled: bool = False
    selective_skyhook_prior_enabled: bool = False
    sequence_mpc_enabled: bool = False
    full_mpc_enabled: bool = False

    compute_time_diagnostics_enabled: bool = True
    debug_compute_sleep_ms: float = 0.0


class PlanningAwareV4MpcPhaseAController(PlanningAwareV3MpcPrimaryController):
    """Projection-aware Phase A rerank controller built on v3 MPC-lite."""

    name = "planning_aware_v4_mpc_phaseA_authority"
    requires_suspension_state = True

    def __init__(
        self,
        config: Optional[PlanningAwareV4MpcPhaseAConfig] = None,
    ):
        super().__init__(config or PlanningAwareV4MpcPhaseAConfig())

    def compute(self, context: ControllerContext) -> ControllerOutput:
        start_ns = time.perf_counter_ns()
        raw_optimizer_ms = 0.0
        projection_rerank_ms = 0.0
        projection_ms = 0.0
        final_guard_ms = 0.0
        cfg = self.config
        wheel_count = self._wheel_count(context)
        self._ensure_previous_count(wheel_count)

        shadow_output = self.skyhook_roll_v3_shadow.compute(context)
        shadow_dampers = self._command_dampers(
            shadow_output.command,
            wheel_count,
            default=1.0)
        shadow_usable, shadow_reason = self._shadow_output_usable(
            shadow_output,
            wheel_count)

        preview = build_planning_preview(context, cfg)
        comfort = self.comfort_guard.update(context, cfg)
        diagnostics = self._base_diagnostics(
            preview=preview,
            comfort=comfort,
            shadow_output=shadow_output,
            shadow_dampers=shadow_dampers,
            wheel_count=wheel_count)
        self._set_v4_default_diagnostics(diagnostics)

        def finish(command, mode: str, reason: str) -> ControllerOutput:
            diagnostics["planning_aware_v4_mode"] = mode
            diagnostics["planning_aware_v4_fallback_reason"] = reason
            self._ensure_required_diagnostics(diagnostics)
            self._ensure_v4_diagnostics(diagnostics)
            self._finalize_timing_diagnostics(
                diagnostics,
                start_ns=start_ns,
                raw_optimizer_ms=raw_optimizer_ms,
                projection_rerank_ms=projection_rerank_ms,
                projection_ms=projection_ms,
                final_guard_ms=final_guard_ms)
            return ControllerOutput(command=command, diagnostics=diagnostics)

        if not preview.valid:
            if shadow_usable:
                final_dampers = self._clamped_finite_dampers(
                    shadow_dampers,
                    wheel_count)
                command = self._command_from_dampers(final_dampers)
                diagnostics.update({
                    "planning_aware_v3_mode": "skyhook_roll_v3_fallback",
                    "planning_aware_v3_fallback_reason": (
                        preview.fallback_reason),
                    "mpc_best_cost": 0.0,
                    "mpc_second_best_cost": 0.0,
                    "mpc_cost_margin": 0.0,
                })
                self._set_command_diagnostics(
                    diagnostics,
                    desired=final_dampers,
                    projected=final_dampers,
                    final=final_dampers,
                    previous=self.previous_final_damper_scales,
                    projection_active=False)
                self.previous_final_damper_scales = list(final_dampers)
                return finish(
                    command,
                    "skyhook_roll_v3_fallback",
                    preview.fallback_reason)

            final_dampers = tuple(1.0 for _ in range(wheel_count))
            command = self._command_from_dampers(final_dampers)
            reason = "%s;skyhook_roll_v3_invalid:%s" % (
                preview.fallback_reason,
                shadow_reason)
            diagnostics.update({
                "planning_aware_v3_mode": "identity_fallback",
                "planning_aware_v3_fallback_reason": reason,
                "mpc_best_cost": 0.0,
                "mpc_second_best_cost": 0.0,
                "mpc_cost_margin": 0.0,
            })
            self._set_command_diagnostics(
                diagnostics,
                desired=final_dampers,
                projected=final_dampers,
                final=final_dampers,
                previous=self.previous_final_damper_scales,
                projection_active=False)
            self.previous_final_damper_scales = list(final_dampers)
            return finish(command, "identity_fallback", reason)

        raw_start_ns = time.perf_counter_ns()
        optimizer = self._generate_v3_raw_mpc_grid_candidates(
            context=context,
            preview=preview,
            comfort=comfort,
            shadow_dampers=shadow_dampers,
            top_k=self._projection_top_k())
        raw_optimizer_ms = _elapsed_ms(raw_start_ns)
        diagnostics.update(self._optimizer_diagnostics(optimizer, comfort))
        diagnostics.update(self._raw_v4_diagnostics(optimizer))

        if optimizer.selected is None:
            reason = "mpc_optimizer_failed:%s" % optimizer.failed_reason
            if shadow_usable:
                final_dampers = self._clamped_finite_dampers(
                    shadow_dampers,
                    wheel_count)
                command = self._command_from_dampers(final_dampers)
                diagnostics.update({
                    "planning_aware_v3_valid": 0,
                    "planning_aware_v3_mode": "skyhook_roll_v3_fallback",
                    "planning_aware_v3_fallback_reason": reason,
                })
            else:
                final_dampers = tuple(1.0 for _ in range(wheel_count))
                command = self._command_from_dampers(final_dampers)
                diagnostics.update({
                    "planning_aware_v3_valid": 0,
                    "planning_aware_v3_mode": "identity_fallback",
                    "planning_aware_v3_fallback_reason": (
                        "%s;skyhook_roll_v3_invalid:%s" %
                        (reason, shadow_reason)),
                })
            self._set_command_diagnostics(
                diagnostics,
                desired=final_dampers,
                projected=final_dampers,
                final=final_dampers,
                previous=self.previous_final_damper_scales,
                projection_active=False)
            self.previous_final_damper_scales = list(final_dampers)
            return finish(command, diagnostics["planning_aware_v3_mode"], reason)

        rerank_start_ns = time.perf_counter_ns()
        if bool(cfg.projection_aware_rerank_enabled):
            rerank_candidates = self._build_projection_rerank_candidates(
                optimizer,
                wheel_count)
            if not rerank_candidates:
                rerank_candidates = (
                    replace(optimizer.selected, selected=False),)
        else:
            rerank_candidates = (replace(optimizer.selected, selected=False),)
        evaluated = tuple(
            self._evaluate_candidate_as_applied(
                candidate,
                context,
                preview,
                comfort,
                shadow_dampers)
            for candidate in rerank_candidates)
        ranked = tuple(sorted(
            evaluated,
            key=functools.cmp_to_key(self._compare_projected_candidates)))
        projection_rerank_ms = _elapsed_ms(rerank_start_ns)

        selected = replace(ranked[0], selected=True)
        final_dampers = selected.final_dampers or tuple(
            1.0 for _ in range(wheel_count))
        projected = selected.projected_dampers or final_dampers
        constrained = selected.constrained_dampers or selected.raw_dampers
        if not _all_finite(final_dampers):
            final_dampers = tuple(1.0 for _ in range(wheel_count))
            projected = final_dampers
            diagnostics.update({
                "planning_aware_v3_valid": 0,
                "planning_aware_v3_mode": "identity_fallback",
                "planning_aware_v3_fallback_reason": (
                    "final_nonfinite_safety_guard"),
                "mpc_final_finite_guard_active": 1,
            })
            command = self._command_from_dampers(final_dampers)
            self._set_command_diagnostics(
                diagnostics,
                desired=selected.raw_dampers,
                projected=projected,
                final=final_dampers,
                previous=self.previous_final_damper_scales,
                projection_active=False)
            self.previous_final_damper_scales = list(final_dampers)
            return finish(
                command,
                "identity_fallback",
                "final_nonfinite_safety_guard")

        modal = self._effective_modal_from_dampers(final_dampers, preview)
        projection_active = self._l2_distance(constrained, projected) > 1.0e-12
        final_guard_active = self._l2_distance(projected, final_dampers) > 1.0e-12
        components = dict(selected.projected_components or selected.components)
        diagnostics.update({
            "planning_aware_v3_mode": "mpc_primary",
            "planning_aware_v3_fallback_reason": "none",
            "mpc_final_finite_guard_active": 0,
            "mpc_u_mean": modal[0],
            "mpc_u_roll_front": modal[1],
            "mpc_u_roll_rear": modal[2],
            "mpc_u_pitch": modal[3],
            "mpc_ltr_proxy_peak": _safe_float(
                selected.projected_ltr_proxy_peak,
                selected.ltr_proxy_peak),
            "mpc_projection_active": int(projection_active),
            "mpc_final_rate_guard_active": int(final_guard_active),
            "mpc_b2d_threshold_cost": components.get("b2d", 0.0),
            "mpc_body_cost": components.get("body", 0.0),
            "mpc_ltr_cost": components.get("ltr", 0.0),
            "mpc_command_effort_cost": components.get("command", 0.0),
            "mpc_slew_cost": components.get("slew", 0.0),
            "mpc_mean_damper_cost": components.get("mean", 0.0),
            "mpc_semi_active_feasibility_cost": components.get("feasible", 0.0),
            "skyhook_prior_penalty": components.get("prior", 0.0),
        })
        self._set_command_diagnostics(
            diagnostics,
            desired=selected.raw_dampers,
            projected=projected,
            final=final_dampers,
            previous=self.previous_final_damper_scales,
            projection_active=projection_active)
        diagnostics.update(self._rerank_v4_diagnostics(
            optimizer=optimizer,
            ranked=ranked,
            selected=selected,
            previous=self.previous_final_damper_scales,
            wheel_count=wheel_count))
        command = self._command_from_dampers(final_dampers)
        self.previous_final_damper_scales = list(final_dampers)
        return finish(command, "projection_rerank", "none")

    def _projection_top_k(self) -> int:
        return max(0, int(_safe_float(
            getattr(self.config, "projection_rerank_top_k", 16),
            16)))

    def _build_projection_rerank_candidates(
        self,
        optimizer: Any,
        wheel_count: int,
    ) -> Tuple[MpcDamperCandidate, ...]:
        candidates: List[MpcDamperCandidate] = []
        seen = set()

        def append(candidate: Optional[MpcDamperCandidate]) -> None:
            if candidate is None:
                return
            normalized = replace(candidate, selected=False)
            key = (
                normalized.kind,
                int(_safe_float(normalized.raw_index, 0)),
                tuple(_match_length(normalized.raw_dampers, wheel_count, 1.0)))
            if key in seen:
                return
            seen.add(key)
            candidates.append(normalized)

        for candidate in tuple(optimizer.raw_candidates)[:self._projection_top_k()]:
            append(candidate)
        raw_candidates = tuple(optimizer.raw_candidates)
        if bool(self.config.projection_rerank_include_raw_winner):
            append(raw_candidates[0] if raw_candidates else optimizer.selected)
        if bool(self.config.projection_rerank_include_second_best):
            append(raw_candidates[1] if len(raw_candidates) > 1 else None)
        if (
                bool(self.config.previous_candidate_enabled) and
                bool(self.config.projection_rerank_include_previous)):
            append(MpcDamperCandidate(
                kind="previous_hold",
                raw_dampers=tuple(_match_length(
                    self.previous_final_damper_scales,
                    wheel_count,
                    1.0)),
                raw_index=-2))
        if (
                bool(self.config.neutral_candidate_enabled) and
                bool(self.config.projection_rerank_include_neutral)):
            append(MpcDamperCandidate(
                kind="neutral_return",
                raw_dampers=tuple(1.0 for _ in range(wheel_count)),
                raw_index=-1))
        return tuple(candidates)

    def _compare_projected_candidates(
        self,
        left: MpcDamperCandidate,
        right: MpcDamperCandidate,
    ) -> int:
        left_cost = _safe_float(left.projected_cost, 1.0e12)
        right_cost = _safe_float(right.projected_cost, 1.0e12)
        if left_cost < right_cost - 1.0e-9:
            return -1
        if right_cost < left_cost - 1.0e-9:
            return 1
        left_key = self._projected_tie_key(left)
        right_key = self._projected_tie_key(right)
        if left_key < right_key:
            return -1
        if left_key > right_key:
            return 1
        return 0

    def _projected_tie_key(
        self,
        candidate: MpcDamperCandidate,
    ) -> Tuple[int, float, float, int]:
        kind_priority = {
            "previous_hold": 0,
            "neutral_return": 1,
        }.get(candidate.kind, 2)
        final = candidate.final_dampers or candidate.raw_dampers
        previous = _match_length(
            self.previous_final_damper_scales,
            len(final),
            1.0)
        final_to_previous = (
            self._l2_distance(final, previous)
            if candidate.kind == "mpc_grid" else 0.0)
        return (
            kind_priority,
            final_to_previous,
            _safe_float(candidate.raw_cost, 1.0e12),
            int(_safe_float(candidate.raw_index, 0)))

    def _raw_v4_diagnostics(self, optimizer: Any) -> Dict[str, Any]:
        return {
            "v4_raw_candidate_count": optimizer.candidate_count,
            "v4_raw_feasible_candidate_count": optimizer.feasible_candidate_count,
            "v4_raw_best_cost": _safe_float(optimizer.best_cost, 0.0),
            "v4_raw_second_best_cost": _safe_float(
                optimizer.second_best_cost,
                0.0),
            "v4_raw_cost_margin": _safe_float(optimizer.cost_margin, 0.0),
            "v4_raw_winner_kind": (
                optimizer.selected.kind if optimizer.selected is not None else ""),
        }

    def _rerank_v4_diagnostics(
        self,
        *,
        optimizer: Any,
        ranked: Sequence[MpcDamperCandidate],
        selected: MpcDamperCandidate,
        previous: Sequence[float],
        wheel_count: int,
    ) -> Dict[str, Any]:
        second = ranked[1] if len(ranked) > 1 else selected
        previous_candidate = self._first_candidate_of_kind(ranked, "previous_hold")
        neutral_candidate = self._first_candidate_of_kind(ranked, "neutral_return")
        raw_winner = (
            optimizer.raw_candidates[0]
            if optimizer.raw_candidates else optimizer.selected)
        raw_winner_projected = (
            self._matching_candidate(ranked, raw_winner)
            if raw_winner is not None else None)
        raw = _match_length(selected.raw_dampers, wheel_count, 1.0)
        constrained = _match_length(
            selected.constrained_dampers or raw,
            wheel_count,
            1.0)
        projected = _match_length(
            selected.projected_dampers or constrained,
            wheel_count,
            1.0)
        final = _match_length(selected.final_dampers or projected, wheel_count, 1.0)
        previous_values = _match_length(previous, wheel_count, 1.0)
        neutral_values = tuple(1.0 for _ in range(wheel_count))
        selected_key = self._candidate_identity_key(selected)
        raw_winner_key = (
            self._candidate_identity_key(raw_winner)
            if raw_winner is not None else None)
        diagnostics: Dict[str, Any] = {
            "v4_rerank_candidate_count": len(ranked),
            "v4_rerank_top_k": self._projection_top_k(),
            "v4_previous_candidate_included": int(previous_candidate is not None),
            "v4_neutral_candidate_included": int(neutral_candidate is not None),
            "v4_projected_best_cost": _safe_float(selected.projected_cost, 0.0),
            "v4_projected_second_best_cost": _safe_float(
                second.projected_cost,
                _safe_float(selected.projected_cost, 0.0)),
            "v4_projected_cost_margin": max(
                0.0,
                _safe_float(second.projected_cost, 0.0) -
                _safe_float(selected.projected_cost, 0.0)),
            "v4_previous_projected_cost": _safe_float(
                getattr(previous_candidate, "projected_cost", None),
                0.0),
            "v4_neutral_projected_cost": _safe_float(
                getattr(neutral_candidate, "projected_cost", None),
                0.0),
            "v4_raw_winner_projected_cost": _safe_float(
                getattr(raw_winner_projected, "projected_cost", None),
                0.0),
            "v4_projected_winner_kind": selected.kind,
            "v4_selected_candidate_kind": selected.kind,
            "v4_projection_changed_winner": int(selected_key != raw_winner_key),
            "v4_selected_from_previous": int(selected.kind == "previous_hold"),
            "v4_selected_from_neutral": int(selected.kind == "neutral_return"),
            "v4_selected_from_mpc_grid": int(selected.kind == "mpc_grid"),
            "v4_raw_to_constrained_l2": self._l2_distance(raw, constrained),
            "v4_constrained_to_projected_l2": self._l2_distance(
                constrained,
                projected),
            "v4_projected_to_final_l2": self._l2_distance(projected, final),
            "v4_raw_to_final_l2": self._l2_distance(raw, final),
            "v4_final_to_previous_l2": self._l2_distance(final, previous_values),
            "v4_final_to_neutral_l2": self._l2_distance(final, neutral_values),
        }
        for index, label in enumerate(_WHEEL_LABELS[:wheel_count]):
            diagnostics["v4_raw_to_final_%s" % label] = final[index] - raw[index]
            diagnostics["v4_projected_to_final_%s" % label] = (
                final[index] - projected[index])
            diagnostics["v4_selected_raw_damper_%s" % label] = raw[index]
            diagnostics["v4_selected_projected_damper_%s" % label] = (
                projected[index])
            diagnostics["v4_selected_final_damper_%s" % label] = final[index]
        return diagnostics

    def _matching_candidate(
        self,
        candidates: Sequence[MpcDamperCandidate],
        target: MpcDamperCandidate,
    ) -> Optional[MpcDamperCandidate]:
        target_key = self._candidate_identity_key(target)
        for candidate in candidates:
            if self._candidate_identity_key(candidate) == target_key:
                return candidate
        return None

    def _candidate_identity_key(
        self,
        candidate: MpcDamperCandidate,
    ) -> Tuple[str, int, Tuple[float, ...]]:
        return (
            candidate.kind,
            int(_safe_float(candidate.raw_index, 0)),
            tuple(candidate.raw_dampers))

    def _first_candidate_of_kind(
        self,
        candidates: Sequence[MpcDamperCandidate],
        kind: str,
    ) -> Optional[MpcDamperCandidate]:
        for candidate in candidates:
            if candidate.kind == kind:
                return candidate
        return None

    def _set_v4_default_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        diagnostics.update({
            "v4_phaseA_enabled": 1,
            "v4_rerank_top_k": self._projection_top_k(),
            "planning_aware_v4_mode": (
                "preview" if diagnostics.get("planning_aware_v3_valid") else
                "fallback"),
            "planning_aware_v4_fallback_reason": diagnostics.get(
                "planning_aware_v3_fallback_reason",
                ""),
            "debug_compute_sleep_ms": _safe_float(
                getattr(self.config, "debug_compute_sleep_ms", 0.0),
                0.0),
        })

    def _ensure_v4_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        for key in PLANNING_AWARE_V4_PHASEA_DIAGNOSTIC_FIELDS:
            if key in diagnostics:
                continue
            if key.endswith("_kind") or key.endswith("_mode") or key.endswith(
                    "_reason"):
                diagnostics[key] = ""
            else:
                diagnostics[key] = 0.0

    def _finalize_timing_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        *,
        start_ns: int,
        raw_optimizer_ms: float,
        projection_rerank_ms: float,
        projection_ms: float,
        final_guard_ms: float,
    ) -> None:
        debug_sleep_ms = max(
            0.0,
            _safe_float(getattr(self.config, "debug_compute_sleep_ms", 0.0), 0.0))
        if debug_sleep_ms > 0.0:
            time.sleep(debug_sleep_ms / 1000.0)
        if bool(getattr(self.config, "compute_time_diagnostics_enabled", True)):
            diagnostics.update({
                "controller_compute_ms": _elapsed_ms(start_ns),
                "controller_compute_ms_raw_optimizer": raw_optimizer_ms,
                "controller_compute_ms_projection_rerank": projection_rerank_ms,
                "controller_compute_ms_projection": projection_ms,
                "controller_compute_ms_final_guard": final_guard_ms,
                "debug_compute_sleep_ms": debug_sleep_ms,
            })


def _elapsed_ms(start_ns: int) -> float:
    return max(0.0, (time.perf_counter_ns() - start_ns) / 1.0e6)
