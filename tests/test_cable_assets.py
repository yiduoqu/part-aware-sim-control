# SPDX-License-Identifier: Apache-2.0
"""Every cable asset loads against its SOURCE.md, and its rigid groups fit its mesh."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

TIER = "fast"

FOLDER = Path(__file__).resolve().parent.parent / "examples" / "10_heterogeneous"


def _cable_scene():
    """``cable_scene`` by file path, with its folder importable for ``rigid_part_body``."""
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    name = "_rss_cable_scene"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, FOLDER / "cable_scene.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class CableAssets(unittest.TestCase):
    def test_each_asset_loads_with_consistent_groups(self):
        cable_scene = _cable_scene()
        for name, asset in cable_scene.ASSETS.items():
            with self.subTest(asset=name):
                _vertices, _tets, labels = cable_scene.load(asset)
                members = [p for g in asset.groups() for p in cable_scene._members(g)]
                self.assertTrue(members)
                self.assertEqual(len(members), len(set(members)), "groups overlap")
                self.assertTrue(set(members) <= set(np.unique(labels).tolist()))
                self.assertGreater(len(cable_scene.soft_labels(asset, labels)), 0)

    def test_socket_collision_mesh_is_well_formed(self):
        with np.load(FOLDER / "assets" / "socket_usb_a" / "physics" / "collision_mesh.npz") as mesh:
            vertices, faces = mesh["vertices_m"], mesh["faces"]
        self.assertGreaterEqual(faces.min(), 0)
        self.assertLess(faces.max(), len(vertices))


if __name__ == "__main__":
    unittest.main()
