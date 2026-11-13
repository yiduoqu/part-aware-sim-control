"""Core orchestration checks with explicit fake VLM responses; no physics tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from part_aware_control.planner import measured_success_gate, parse_airbot_rollout, rejected_rollout, run_planning

try:
    import scipy  # noqa: F401
    import simpact  # noqa: F401
except ImportError:
    HAS_PLANNER = False
else:
    HAS_PLANNER = True


@unittest.skipUnless(HAS_PLANNER, "requires separate SIMPACT planner environment")
class TestPlanner(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "scene.png"
        Image.new("RGB", (8, 8), "white").save(self.image)
        self.observation = self.root / "initial.json"
        self.observation.write_text(json.dumps(self.trace(False)))

    def trace(self, success):
        return {
            "snapshots": [
                {
                    "label": "initial",
                    "gripper": {"position": [0, 0, 0.2], "orientation": [0, 0, 0, 1], "width": 0.05},
                    "objects": {"cube": {"position": [0, 0, 0.02]}},
                    "screenshot": str(self.image),
                },
                {
                    "label": "final",
                    "gripper": {"position": [0, 0, 0.3], "orientation": [0, 0, 0, 1], "width": 0.01},
                    "objects": {"cube": {"position": [0.1, 0, 0.02]}},
                    "screenshot": str(self.image),
                },
            ],
            "outcome": {"success": success, "valid": True},
        }

    def run_loop(self, successful_indices=(), confirmation_success=True, malformed=False, rejected_indices=()):
        calls = []
        response_count = [0]

        def model(contents):
            prompt = str(contents[1] if len(contents) == 2 else contents[0])
            if "exactly 10" in prompt:
                count = 3 if malformed else 10
                plans = [
                    {"description": str(i), "action_sequence": [{"type": "LIFT", "delta_z": 0.1}]} for i in range(count)
                ]
                return json.dumps({"action_proposals": plans})
            if "ONE revised" in prompt:
                response_count[0] += 1
                return json.dumps(
                    {
                        "action_proposals": [
                            {
                                "description": f"refine {response_count[0]}",
                                "action_sequence": [
                                    {
                                        "type": "move",
                                        "delta_x": 0,
                                        "delta_y": 0,
                                        "delta_z": 0.1,
                                        "delta_roll": 0,
                                        "delta_pitch": 0,
                                        "delta_yaw": 0,
                                    }
                                ],
                            }
                        ]
                    }
                )
            return json.dumps({"success": True, "confidence": 1.0, "reason": "valid", "remaining": ""})

        def worker(plan, index, out_dir):
            calls.append((index, str(out_dir), plan.description))
            out_dir.mkdir(parents=True)
            confirmed = "confirmation" in out_dir.parts
            if not confirmed and index in rejected_indices:
                return rejected_rollout(
                    {
                        "type": "InvalidPlan",
                        "message": "IK orientation misses by 51.42 degrees",
                        "snapshots": self.trace(False)["snapshots"],
                        "executed_actions": [],
                    },
                    plan,
                    json.loads(self.observation.read_text()),
                    out_dir,
                )
            success = confirmation_success if confirmed else index in successful_indices
            path = out_dir / "rollout.json"
            path.write_text(json.dumps(self.trace(success)))
            return path

        result = run_planning(
            observation_json=self.observation,
            instruction="place cube",
            out_dir=self.root / "run",
            rollout_fn=worker,
            generate_fn=model,
            provider="google",
            model="explicit-fake-model",
        )
        return result, calls

    def test_initial_success_executes_batch_then_confirms_first(self):
        result, calls = self.run_loop(successful_indices={2, 4})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actual_proposal_count"], 10)
        self.assertEqual(result["planning_rollouts_completed"], 10)
        self.assertEqual(result["refinement_iterations"], 0)
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(result["confirmation_rollouts_completed"], 1)
        self.assertEqual(calls[-1][2], "2")

    def test_refinement_budget_failure_has_no_confirmation(self):
        result, calls = self.run_loop()
        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["planning_rollouts_completed"], 13)
        self.assertEqual(result["refinement_iterations"], 3)
        self.assertEqual(result["confirmation_rollouts_attempted"], 0)
        self.assertEqual(len(calls), 13)
        final = json.loads((self.root / "run" / "selected_plan.json").read_text())
        self.assertEqual(final["action_proposals"][0]["description"], "refine 3")

    def test_first_refinement_success_confirmation_failure_stops(self):
        result, calls = self.run_loop(successful_indices={10}, confirmation_success=False)
        self.assertEqual(result["status"], "failure")
        self.assertTrue(result["planning_success"])
        self.assertFalse(result["confirmation_success"])
        self.assertEqual(result["refinement_iterations"], 1)
        self.assertEqual(len(calls), 12)

    def test_wrong_proposal_count_is_error_and_never_simulates(self):
        result, calls = self.run_loop(malformed=True)
        self.assertEqual(result["status"], "error")
        self.assertFalse(calls)
        self.assertEqual(result["actual_proposal_count"], 3)
        logs = json.loads((self.root / "run" / "vlm" / "vlm_calls.json").read_text())
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[-1]["proposal_count"], 3)

    def test_rejected_batch_and_refinement_keep_feedback_and_budget(self):
        result, calls = self.run_loop(successful_indices={11}, rejected_indices=set(range(11)))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["planning_rollouts_attempted"], 12)
        self.assertEqual(result["planning_rollouts_rejected"], 11)
        self.assertEqual(result["planning_rollouts_completed"], 1)
        self.assertEqual(result["refinement_iterations"], 2)
        self.assertEqual(result["confirmation_rollouts_completed"], 1)
        self.assertEqual(len(calls), 13)
        rejected = self.root / "run" / "rollouts" / "rollout_000.json"
        self.assertFalse(measured_success_gate(rejected)[0])
        text = parse_airbot_rollout(rejected)["text"]
        self.assertIn("EXECUTION REJECTED", text)
        self.assertIn("51.42 degrees", text)
        self.assertIn("Requested complete plan", text)

    def test_hanging_profile_uses_six_axis_initial_and_refined_plans(self):
        from part_aware_control.providers import response_schema

        schema = response_schema("propose", profile="clawhauser_hang")["properties"]["action_proposals"]
        self.assertEqual(schema["maxItems"], 10)
        variants = schema["items"]["properties"]["action_sequence"]["items"]["anyOf"]
        self.assertEqual({v["properties"]["type"]["enum"][0] for v in variants}, {"move", "gripper_control"})
        move = next(v for v in variants if v["properties"]["type"]["enum"] == ["move"])
        self.assertIn("delta_roll", move["required"])
        self.assertIn("delta_pitch", move["required"])
        self.assertEqual(
            response_schema("refine", profile="clawhauser_hang")["properties"]["action_proposals"]["maxItems"], 1
        )
        observation = self.trace(False)
        observation["task"] = {"name": "clawhauser_hang"}
        for snapshot in observation["snapshots"]:
            snapshot["hanging"] = {"hole_center_world_m": [0, 0, 1], "rod_loop_support_n": 0.0, "finger_contacts": 0}
        self.observation.write_text(json.dumps(observation))

        def provider(model, stage, *, profile):
            self.assertEqual((model, profile), ("gpt-6-sol", "clawhauser_hang"))

            def generate(contents):
                self.assertIn("rod", " ".join(c for c in contents if isinstance(c, str)).lower())
                if stage == "verify":
                    return json.dumps({"success": True, "confidence": 1, "reason": "valid", "remaining": ""})
                plan = {
                    "description": "turn and hang",
                    "action_sequence": [
                        {
                            "type": "move",
                            "delta_x": 0,
                            "delta_y": 0,
                            "delta_z": 0,
                            "delta_roll": 0.4,
                            "delta_pitch": -0.2,
                            "delta_yaw": 0,
                        },
                        {"type": "gripper_control", "width": 0.06},
                    ],
                }
                return json.dumps({"action_proposals": [plan] * (10 if stage == "propose" else 1)})

            return generate

        def worker(plan, index, out_dir):
            self.assertEqual(plan.action_sequence[0].delta_roll, 0.4)
            self.assertEqual(plan.action_sequence[0].delta_pitch, -0.2)
            out_dir.mkdir(parents=True)
            data = json.loads(json.dumps(observation))
            data["outcome"]["success"] = index == 10 or "confirmation" in out_dir.parts
            path = out_dir / "rollout.json"
            path.write_text(json.dumps(data))
            return path

        with patch("part_aware_control.providers.make_codex_generate_fn", side_effect=provider):
            result = run_planning(
                observation_json=self.observation,
                instruction="hang toy on rod",
                out_dir=self.root / "run",
                rollout_fn=worker,
                provider="codex",
                model="gpt-6-sol",
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["planner_profile"], "clawhauser_hang")
        self.assertEqual(result["planning_rollouts_completed"], 11)
        self.assertEqual(result["refinement_iterations"], 1)
        self.assertEqual(result["confirmation_rollouts_completed"], 1)
        self.assertIn("rod_loop_support_n", parse_airbot_rollout(self.observation)["text"])

    def test_airbot_trace_preserves_closure_and_total_width(self):
        parsed = parse_airbot_rollout(self.observation)
        self.assertIn("Initial total jaw opening: 0.050000", parsed["text"])
        self.assertIn("gripper_control(width=0.010000)", parsed["text"])
        self.assertFalse(measured_success_gate(self.observation)[0])


@unittest.skipUnless(HAS_PLANNER, "requires separate SIMPACT planner environment")
class TestCodexProvider(unittest.TestCase):
    def test_cli_preserves_model_schema_images_and_saved_auth(self):
        import subprocess

        from PIL import Image

        from part_aware_control.providers import make_codex_generate_fn, response_schema

        def fake_run(command, **kwargs):
            directory = kwargs["cwd"]
            self.assertTrue(str(directory).startswith(tempfile.gettempdir()))
            self.assertEqual(command[command.index("--model") + 1], "gpt-6-sol")
            self.assertIn("--ignore-user-config", command)
            self.assertIn("--ephemeral", command)
            self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
            self.assertEqual(command[-2:], ["--", "-"])
            self.assertEqual(kwargs["timeout"], 12)
            self.assertIn("no API key needed", kwargs["input"])
            image_path = Path(command[command.index("--image") + 1])
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (8, 8))
            schema_path = Path(command[command.index("--output-schema") + 1])
            self.assertEqual(json.loads(schema_path.read_text()), response_schema("verify"))
            self.assertEqual(response_schema("propose")["properties"]["action_proposals"]["maxItems"], 10)
            self.assertEqual(response_schema("refine")["properties"]["action_proposals"]["maxItems"], 1)
            Path(command[command.index("--output-last-message") + 1]).write_text('{"success":false}')
            return subprocess.CompletedProcess(command, 0, stdout="ignored events", stderr="")

        with patch("part_aware_control.providers.subprocess.run", side_effect=fake_run) as run:
            generate = make_codex_generate_fn("gpt-6-sol", "verify", timeout_s=12)
            response = generate(["no API key needed", Image.new("RGB", (8, 8))])
            self.assertEqual(response, '{"success":false}')
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
