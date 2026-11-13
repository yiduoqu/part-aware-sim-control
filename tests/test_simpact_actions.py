"""Core action/frame/aperture conversions; no simulator or model calls."""

import math
import unittest

from part_aware_control.actions import normalize_plan, rotate_world, width_to_command


class AirbotActions(unittest.TestCase):
    def test_proposal_and_refinement_have_same_world_delta(self):
        primitive = normalize_plan({"action_sequence": [{"type": "DESCEND", "delta_z": 0.12}]})[0]
        refined = normalize_plan(
            {
                "action_sequence": [
                    {
                        "type": "move",
                        "delta_x": 0,
                        "delta_y": 0,
                        "delta_z": -0.12,
                        "delta_roll": 0,
                        "delta_pitch": 0,
                        "delta_yaw": 0,
                    }
                ]
            }
        )[0]
        self.assertEqual(primitive.delta, refined.delta)
        with self.assertRaises(ValueError):
            normalize_plan([{"type": "DESCEND", "delta_z": -0.12}])

    def test_total_aperture_and_airbot_release(self):
        actions = normalize_plan([{"type": "GRASP", "grasp_width": 0.04}, {"type": "RELEASE"}])
        self.assertEqual(actions[0].width, 0.04)
        self.assertAlmostEqual(width_to_command(actions[0].width), 0.5)
        self.assertEqual(width_to_command(actions[1].width), 0)
        with self.assertRaises(ValueError):
            normalize_plan([{"type": "gripper_control", "width": 0.10}])

    def test_xyzw_world_rotation(self):
        x, y, z, w = rotate_world((0, 0, 0, 1), (0, 0, math.pi / 2))
        self.assertEqual((x, y), (0, 0))
        self.assertAlmostEqual(z, math.sqrt(0.5))
        self.assertAlmostEqual(w, math.sqrt(0.5))
        roll = rotate_world((0, 0, 0, 1), (math.pi / 2, 0, 0))
        turned = rotate_world(roll, (0, math.pi / 2, 0))
        for index, expected in enumerate((0.5, 0.5, -0.5, 0.5)):
            self.assertAlmostEqual(turned[index], expected)

    def test_invalid_numeric_action_does_not_reach_simulator(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                normalize_plan([{"type": "LIFT", "delta_z": value}])
