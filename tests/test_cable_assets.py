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


def _module(stem: str):
    """``stem``.py from the examples folder by file path, with the folder importable for its own imports."""
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    name = f"_rss_{stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, FOLDER / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class CableAssets(unittest.TestCase):
    def test_each_asset_loads_with_consistent_groups(self):
        cable_scene, hetero_scene = _module("cable_scene"), _module("hetero_scene")
        for name, asset in cable_scene.ASSETS.items():
            with self.subTest(asset=name):
                _vertices, _tets, labels = hetero_scene.load(asset)
                members = [p for g in asset.groups() for p in hetero_scene._members(g)]
                self.assertTrue(members)
                self.assertEqual(len(members), len(set(members)), "groups overlap")
                self.assertTrue(set(members) <= set(np.unique(labels).tolist()))
                self.assertGreater(len(hetero_scene.soft_labels(asset, labels)), 0)

    def test_socket_collision_mesh_is_well_formed(self):
        with np.load(FOLDER / "assets" / "socket_usb_a" / "physics" / "collision_mesh.npz") as mesh:
            vertices, faces = mesh["vertices_m"], mesh["faces"]
        self.assertGreaterEqual(faces.min(), 0)
        self.assertLess(faces.max(), len(vertices))


if __name__ == "__main__":
    unittest.main()
