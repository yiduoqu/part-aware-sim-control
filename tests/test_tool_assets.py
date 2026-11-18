# SPDX-License-Identifier: Apache-2.0
"""Every tool asset loads against its SOURCE.md, and its rigid groups and handle fit its mesh."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TIER = "fast"

FOLDER = Path(__file__).resolve().parent.parent / "examples" / "10_heterogeneous"


class ToolAssets(unittest.TestCase):
    def test_each_tool_loads_with_disjoint_groups_and_a_handle(self):
        if str(FOLDER) not in sys.path:
            sys.path.insert(0, str(FOLDER))
        import cable_scene
        import tool_scene

        for name, asset in tool_scene.TOOLS.items():
            with self.subTest(asset=name):
                vertices, tets, labels = cable_scene.load(asset)
                members = [p for g in asset.groups() for p in cable_scene._members(g)]
                self.assertEqual(len(members), len(set(members)), "groups overlap")
                self.assertTrue(set(members) <= set(np.unique(labels).tolist()))
                self.assertGreater(len(cable_scene.soft_labels(asset, labels)), 0)
                self.assertIn(asset.handle, members)
                grasp = tool_scene.grasp_handle(asset, vertices, tets, labels)
                self.assertGreater(grasp["half_width_m"], 0.0)

    def test_socket_stations_load(self):
        if str(FOLDER) not in sys.path:
            sys.path.insert(0, str(FOLDER))
        import cable_scene

        for name in ("socket_station", "socket_station_loose", "socket_station_snug"):
            with self.subTest(asset=name):
                path, connectors = cable_scene.fixture(name)
                self.assertTrue(path.exists())
                self.assertTrue(connectors)


if __name__ == "__main__":
    unittest.main()
