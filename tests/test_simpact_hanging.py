"""One core hanging-gate check, including the main false-positive outcomes."""

import copy
import unittest

from part_aware_control.hanging import measure_hanging, threading_geometry
from part_aware_control.tasks import Task


class HangingGate(unittest.TestCase):
    def test_supported_released_hang_and_failure_modes(self):
        task = Task(name="clawhauser_hang", minimum_lift=0.03)
        geometry = {
            "weight_n": 1.227,
            "rest_surface_z_world_m": 0.753,
            "rod_free_end_world_m": [0.473, 0.03, 1.017],
            "rod_length_m": 0.12,
            "finger_effort_limit_n": 20.0,
            "finger_force_tolerance": 0.5,
        }
        quality = {
            "minimum_J_by_part": [0.8, 0.4, 0.7],
            "inverted_tet_substeps_by_part": [0, 0, 0],
            "nonfinite_tet_substeps_by_part": [0, 0, 0],
        }
        sample = {
            "time_s": 0.0,
            "gripper": {"width": 0.08},
            "objects": {task.name: {"position": [0.473, 0.07, 0.90], "minimum_z": 0.80}},
            "hanging": {
                "rod_loop_support_n": 1.227,
                "rod_loop_contacts": 12,
                "rod_engagement_m": 0.04,
                "rod_axis_winding": 1.0,
                "minimum_hole_engagement_m": 0.035,
                "maximum_hole_engagement_m": 0.045,
                "loop_center_world_m": [0.473, 0.07, 1.01],
                "finger_contacts": 0,
                "finger_squeeze_n": [0.0, 0.0],
                "max_contact_penetration_m": 0.0002,
            },
        }
        samples = [copy.deepcopy(sample) for _ in range(4)]
        for index, time_s in enumerate([0.0, 0.5, 0.75, 1.0]):
            samples[index]["time_s"] = time_s
        samples[0]["objects"][task.name]["minimum_z"] = 0.753
        self.assertTrue(measure_hanging(task, samples, geometry, quality)["success"])
        for failure in ("table", "fingers", "unsupported", "tip", "drape", "inversion", "penetration"):
            with self.subTest(failure=failure):
                failed = copy.deepcopy(samples)
                bad_quality = copy.deepcopy(quality)
                for item in failed[1:]:
                    if failure == "table":
                        item["objects"][task.name]["minimum_z"] = 0.753
                    elif failure == "fingers":
                        item["hanging"]["finger_contacts"] = 1
                    elif failure == "unsupported":
                        item["hanging"]["rod_loop_support_n"] = 0.0
                    elif failure == "tip":
                        item["hanging"]["minimum_hole_engagement_m"] = 0.005
                    elif failure == "drape":
                        item["hanging"]["rod_axis_winding"] = 0.0
                    elif failure == "inversion":
                        bad_quality["inverted_tet_substeps_by_part"][1] = 1
                    elif failure == "penetration":
                        item["hanging"]["max_contact_penetration_m"] = 0.0006
                self.assertFalse(measure_hanging(task, failed, geometry, bad_quality)["success"])
        import numpy as np

        angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        contour = np.column_stack((0.015 * np.cos(angles), np.full(72, 0.04), 0.015 * np.sin(angles)))
        threaded = threading_geometry(contour, [0, 0, 0])
        self.assertAlmostEqual(abs(threaded["rod_axis_winding"]), 1.0)
        self.assertAlmostEqual(threaded["minimum_hole_engagement_m"], 0.04)
        self.assertAlmostEqual(threading_geometry(contour + np.array([0.04, 0, 0]), [0, 0, 0])["rod_axis_winding"], 0.0)
