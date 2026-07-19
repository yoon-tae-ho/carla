"""Contracts for residual MPC helper transforms."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import PlanningInfo
from suspension_control.controllers.planning_aware_mpc_phaseA import (
    PlanningAwareV4MpcPhaseAController,
)
from suspension_control.controllers.planning_aware_mpc_primary import (
    build_planning_preview,
)
from suspension_control.controllers.residual_mpc_contracts import (
    MODAL_BASIS_NAMES,
    MODAL_TO_WHEEL_MATRIX,
    WHEEL_ORDER,
    modal_to_wheel_residual,
    wheel_residual_to_modal,
)

from test_planning_aware_mpc_phaseA_controller import _context, _planning


class ResidualMpcContractsTest(unittest.TestCase):

    def test_wheel_order_and_basis_names_are_fixed(self):
        self.assertEqual(
            ("front_left", "front_right", "rear_left", "rear_right"),
            WHEEL_ORDER)
        self.assertEqual(
            ("mean", "roll_front", "roll_rear", "pitch"),
            MODAL_BASIS_NAMES)
        self.assertEqual(
            (
                (1.0, -1.0, 0.0, 1.0),
                (1.0, 1.0, 0.0, 1.0),
                (1.0, 0.0, -1.0, -1.0),
                (1.0, 0.0, 1.0, -1.0),
            ),
            MODAL_TO_WHEEL_MATRIX)

    def test_unit_modal_vectors_have_expected_signs(self):
        cases = (
            ((1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)),
            ((0.0, 1.0, 0.0, 0.0), (-1.0, 1.0, 0.0, 0.0)),
            ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, -1.0, 1.0)),
            ((0.0, 0.0, 0.0, 1.0), (1.0, 1.0, -1.0, -1.0)),
        )
        for modal, expected in cases:
            self.assertEqual(expected, modal_to_wheel_residual(modal))

    def test_neutral_residual_round_trips_to_zero(self):
        self.assertEqual(
            (0.0, 0.0, 0.0, 0.0),
            modal_to_wheel_residual((0.0, 0.0, 0.0, 0.0)))
        self.assertEqual(
            (0.0, 0.0, 0.0, 0.0),
            wheel_residual_to_modal((0.0, 0.0, 0.0, 0.0)))

    def test_round_trip_within_tolerance(self):
        cases = (
            (0.02, -0.01, 0.015, 0.005),
            (-0.03, 0.02, -0.01, -0.015),
            (0.0, 0.04, -0.02, 0.01),
        )
        for modal in cases:
            wheel = modal_to_wheel_residual(modal)
            recovered = wheel_residual_to_modal(wheel)
            for actual, expected in zip(recovered, modal):
                self.assertAlmostEqual(expected, actual, places=12)

    def test_nonfinite_inputs_are_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                modal_to_wheel_residual((bad, 0.0, 0.0, 0.0))
            with self.assertRaises(ValueError):
                wheel_residual_to_modal((bad, 0.0, 0.0, 0.0))

    def test_bad_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            modal_to_wheel_residual((0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            wheel_residual_to_modal((0.0, 0.0, 0.0))

    def test_contract_matches_phaseA_neutral_preview_geometry_signs(self):
        controller = PlanningAwareV4MpcPhaseAController()
        preview = build_planning_preview(
            _context(planning=_planning(predicted_ay=(0.0, 0.0, 0.0, 0.0))),
            controller.config)
        modal = (0.02, -0.01, 0.015, 0.005)
        phase_a_dampers = controller._modal_to_dampers(
            preview=preview,
            u_mean=modal[0],
            u_roll_front=modal[1],
            u_roll_rear=modal[2],
            u_pitch=modal[3])
        phase_a_residual = tuple(value - 1.0 for value in phase_a_dampers)

        for actual, expected in zip(
                phase_a_residual,
                modal_to_wheel_residual(modal)):
            self.assertTrue(math.isclose(actual, expected, abs_tol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
