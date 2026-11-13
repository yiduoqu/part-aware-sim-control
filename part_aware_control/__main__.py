"""Observe, script, replay, or plan; simulator imports are lazy for separate environments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from .actions import InvalidPlan, normalize_plan
from .tasks import Task, scripted_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("observe", "script", "worker", "plan"))
    parser.add_argument("--task", choices=("cube", "clawhauser", "clawhauser_hang"), default="cube")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--video", action="store_true", help="Record simulator execution to replay.mp4 at scene speed")
    parser.add_argument("--index", type=int, default=0, help="Rollout identifier for the worker")
    parser.add_argument("--task-config", type=Path, help="Exact Task JSON, shared with the initial observation")
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument(
        "--worker-command", help="ARM simulator Python command, e.g. /path/to/python -m part_aware_control"
    )
    parser.add_argument("--provider", choices=("codex", "google"), default="codex")
    parser.add_argument("--model", help="Explicit model override; defaults to gpt-6-sol for Codex")
    parser.add_argument("--cube-start", nargs=2, type=float, default=(-0.13, 0.0), metavar=("X", "Y"))
    args = parser.parse_args()
    if args.video and args.command == "plan":
        parser.error("--video records observe, script or worker executions; replay a selected plan with worker")
    args.model = args.model or ("gpt-6-sol" if args.provider == "codex" else "gemini-2.5-pro")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    task = Task(
        name=args.task,
        cube_start=tuple(args.cube_start),
        target_half=(0.06, 0.06) if args.task == "cube" else (0.06, 0.10),
        minimum_lift=0.04 if args.task == "cube" else 0.03,
        max_ik_error=0.002 if args.task == "clawhauser_hang" else 0.010,
    )
    if args.task_config:
        task = Task(**json.loads(args.task_config.read_text()))
    if args.command == "script" and task.name == "clawhauser_hang":
        parser.error("use the native Hanson reference or worker --plan with an explicit hanging sequence")
    if args.command == "plan":
        if not args.observation or not args.worker_command:
            parser.error("plan requires --observation and --worker-command")
        observation = json.loads(args.observation.read_text())
        task = Task(**observation["task"])
        args.instruction = args.instruction or (
            "Pick up Clawhauser, thread its ring onto the rod, release and withdraw so it remains hanging."
            if task.name == "clawhauser_hang"
            else "Pick up the object, place it in the target, release and withdraw."
        )
        task_path = args.out / "task.json"
        task_path.write_text(json.dumps(task.to_dict(), indent=2) + "\n")

        def rollout_fn(plan, index, out_dir):
            directory = Path(out_dir)
            directory.mkdir(parents=True, exist_ok=True)
            input_path = directory / "plan.json"
            input_path.write_text(json.dumps(plan.to_dict() if hasattr(plan, "to_dict") else plan, indent=2))
            try:
                normalize_plan(plan)
            except InvalidPlan as error:
                if task.name != "clawhauser_hang":
                    raise
                from .planner import rejected_rollout

                return rejected_rollout({"type": "InvalidPlan", "message": str(error)}, plan, observation, directory)
            command = [
                *shlex.split(args.worker_command),
                "worker",
                "--index",
                str(index),
                "--task-config",
                str(task_path),
                "--plan",
                str(input_path),
                "--out",
                str(directory),
            ]
            with (directory / "worker.log").open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if process.returncode:
                error_path = directory / "error.json"
                if error_path.exists():
                    error = json.loads(error_path.read_text())
                    if error["type"] == "InvalidPlan":
                        if task.name == "clawhauser_hang":
                            from .planner import rejected_rollout

                            return rejected_rollout(error, plan, observation, directory)
                        raise ValueError(error["message"])
                raise RuntimeError(f"simulator worker failed; see {directory / 'worker.log'}")
            return directory / "rollout.json"

        from .planner import run_planning

        result = run_planning(
            instruction=args.instruction,
            observation_json=args.observation,
            out_dir=args.out,
            rollout_fn=rollout_fn,
            provider=args.provider,
            model=args.model,
        )
        print(json.dumps(result, indent=2))
        if result["status"] == "error":
            raise SystemExit(2)
        return
    from .rollout import Rollout

    runner = Rollout(task, video=args.video)
    if args.command == "observe":
        path = runner.run(None, 0, args.out)
    elif args.command == "worker":
        if not args.plan:
            parser.error("worker requires --plan")
        try:
            path = runner.run(json.loads(args.plan.read_text()), args.index, args.out)
        except Exception as error:
            if not (args.out / "error.json").exists():
                (args.out / "error.json").write_text(
                    json.dumps({"type": type(error).__name__, "message": str(error)}) + "\n"
                )
            raise
    else:
        observation_path = runner.run(None, 0, args.out / "observation")
        observation = json.loads(observation_path.read_text())
        plan = scripted_plan(observation)
        (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        path = runner.run(plan, 0, args.out / "scripted")
        first = json.loads(path.read_text())
        summary = {
            "source": "scripted",
            "scripted_executions": 1,
            "requested_proposal_count": 0,
            "actual_proposal_count": 0,
            "refinement_iterations": 0,
            "model": None,
            "task": first["task"],
            "revisions": first["revisions"],
            "outcome": first["outcome"],
            "confirmation_executions": 0,
            "confirmation_outcome": None,
            "success": False,
        }
        if first["outcome"]["success"]:
            replay = json.loads(runner.run(plan, 1, args.out / "confirmation").read_text())
            import numpy as np

            same = all(
                np.allclose(first["snapshots"][0]["gripper"][key], replay["snapshots"][0]["gripper"][key], atol=1e-6)
                for key in ("position", "orientation", "width")
            )
            same = same and np.allclose(
                first["snapshots"][0]["objects"][task.name]["position"],
                replay["snapshots"][0]["objects"][task.name]["position"],
                atol=1e-6,
            )
            summary.update(
                {
                    "confirmation_executions": 1,
                    "confirmation_outcome": replay["outcome"],
                    "common_initial_state_reproduced": bool(same),
                    "success": bool(same and replay["outcome"]["success"]),
                }
            )
        (args.out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    print(path)


if __name__ == "__main__":
    main()
