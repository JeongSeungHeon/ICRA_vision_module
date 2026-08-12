"""Safety contract tests for SAM3D/HOI-DETR target invalidation."""

from __future__ import annotations

import unittest

from perception.sam3d_live_runtime import enforce_sam3d_target_gate


class _SharedState:
    def __init__(self):
        self.calls = []

    def clear_target(self, *, reset_prediction=False, reset_arm=False):
        self.calls.append((reset_prediction, reset_arm))


class Sam3DSafetyGateTest(unittest.TestCase):
    def test_blocked_gate_hard_clears_prediction_and_arm(self):
        state = _SharedState()
        self.assertTrue(enforce_sam3d_target_gate(state, True))
        self.assertEqual(state.calls, [(True, True)])

    def test_open_gate_does_not_mutate_target_state(self):
        state = _SharedState()
        self.assertFalse(enforce_sam3d_target_gate(state, False))
        self.assertEqual(state.calls, [])


if __name__ == "__main__":
    unittest.main()
