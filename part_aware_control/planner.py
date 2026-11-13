"""Thin SIMPACT planner bridge; the simulator is an injected file-based worker."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path

SIMPACT_REVISION = "692aab2bb92e4df324d28c76630d17b08e814287"
INITIAL_CANDIDATES = 10
MAX_REFINEMENT_ITERATIONS = 3
PROMPTS = Path(__file__).with_name("prompts")
PROPOSAL_TYPES = {"PUSH", "LIFT", "DESCEND", "GRASP", "RELEASE", "ROTATE", "FLICK"}
REFINEMENT_TYPES = {"move", "gripper_control"}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rejected_rollout(error, plan, observation, out_dir):
    """Retain a known command rejection as feedback, never as a completed run."""
    if error.get("type") != "InvalidPlan":
        raise ValueError("only InvalidPlan can become rejected-plan feedback")
    snapshots = error.get("snapshots", [])
    reference_only = not snapshots
    if reference_only:
        snapshots = [dict(observation["snapshots"][0], label="reference_initial_not_executed")]
    data = {
        "execution_status": "rejected",
        "execution_error": {"type": error["type"], "message": error["message"]},
        "reference_initial_only": reference_only,
        "task": observation.get("task", {}),
        "geometry": observation.get("geometry", {}),
        "plan": plan.to_dict() if hasattr(plan, "to_dict") else plan,
        "executed_actions": error.get("executed_actions", []),
        "snapshots": snapshots,
        "outcome": {"success": False, "valid": False, "reason": "execution rejected; no completed physical outcome"},
    }
    path = Path(out_dir) / "rejected.json"
    _write_json(path, data)
    return path


def parse_airbot_rollout(rollout_json):
    """Read measured xyzw poses and TOTAL jaw opening, including jaw closure."""
    path = Path(rollout_json)
    data = json.loads(path.read_text())
    snaps = data["snapshots"]
    if not snaps:
        raise ValueError("AIRBOT rollout has no snapshots")
    lines = [f"Rollout: {path.name}; world coordinates in metres; quaternion xyzw."]
    if data.get("evidence_source"):
        lines.append("Evidence source: " + data["evidence_source"])
    if data.get("execution_status") == "rejected":
        lines.append("EXECUTION REJECTED before completion: " + json.dumps(data["execution_error"]))
        lines.append("Requested complete plan: " + json.dumps(data.get("plan")))
        if data.get("reference_initial_only"):
            lines.append("Initial snapshot is reference-only; this plan executed no simulation actions.")
    verdict = data.get("verdict")
    if verdict is not None:
        lines.append(f"VERIFIER OUTCOME: {'SUCCESS' if verdict['success'] else 'FAILURE'}")
        lines.append(f"Reason: {verdict.get('reason', '')}")
        lines.append(f"Still needed: {verdict.get('remaining', '')}")
    lines.append("Commanded actions: " + json.dumps(data.get("executed_actions", [])))
    previous = None
    for index, snap in enumerate(snaps):
        grip = snap["gripper"]
        pos, quat, width = grip["position"], grip["orientation"], grip["width"]
        if len(pos) != 3 or len(quat) != 4:
            raise ValueError("AIRBOT pose requires position xyz and orientation xyzw")
        if not all(math.isfinite(float(v)) for v in [*pos, *quat, width]):
            raise ValueError("AIRBOT rollout contains a nonfinite pose or aperture")
        if float(width) < 0:
            raise ValueError("AIRBOT total jaw opening must be nonnegative")
        lines.append(f"Snapshot {index} ({snap.get('label', '')}): measured EE {pos}, xyzw {quat}")
        if "commanded_width" in grip:
            lines.append(f"Commanded total jaw opening: {grip['commanded_width']:.6f} m")
        if previous is None:
            lines.append(f"Initial total jaw opening: {width:.6f} m")
        else:
            delta = [round(float(pos[i]) - float(previous["position"][i]), 6) for i in range(3)]
            lines.append(f"Measured world displacement xyz: {delta}")
            if abs(float(width) - float(previous["width"])) > 1e-6:
                lines.append(f"gripper_control(width={width:.6f}) [measured total opening]")
        lines.append("Objects: " + json.dumps(snap.get("objects", {})))
        if "hanging" in snap:
            lines.append("Measured ring/rod state: " + json.dumps(snap["hanging"]))
        if "mesh_quality" in snap:
            lines.append("Cumulative mesh quality: " + json.dumps(snap["mesh_quality"]))
        previous = grip
    lines.append("Measured task outcome: " + json.dumps(data.get("outcome", {})))
    image = next((s["screenshot"] for s in reversed(snaps) if s.get("screenshot")), None)
    return {"text": "\n".join(lines), "after_image": str(path.parent / image) if image else None}


def measured_success_gate(rollout_json):
    data = json.loads(Path(rollout_json).read_text())
    if data.get("execution_status") == "rejected":
        return False, "Execution rejected; no completed physical outcome: " + json.dumps(data["execution_error"])
    outcome = data["outcome"]
    # A missing or non-boolean field is a schema error, never physical failure.
    if type(outcome.get("success")) is not bool or type(outcome.get("valid")) is not bool:
        raise ValueError("rollout outcome requires boolean success and valid measurements")
    passed = outcome["success"] and outcome["valid"]
    goal = (
        "lift/rod engagement/rod support/release/settling"
        if data.get("task", {}).get("name") == "clawhauser_hang"
        else "lift/placement/release/settling"
    )
    return passed, f"Measured {goal}: " + json.dumps(outcome)


def _require_pinned_simpact():
    import simpact

    source_root = Path(simpact.__file__).resolve().parent.parent
    revision = None
    try:
        top = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if Path(top).resolve() == source_root:
            revision = subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    if revision is None:
        try:
            direct = importlib.metadata.distribution("simpact").read_text("direct_url.json")
            revision = json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")
        except importlib.metadata.PackageNotFoundError:
            pass
    if revision != SIMPACT_REVISION:
        raise RuntimeError(f"SIMPACT must be installed from pinned revision {SIMPACT_REVISION}; found {revision}")
    return revision


class _RecordedGenerator:
    """Bound malformed output to two calls and retain every raw response."""

    def __init__(self, generate_fn, stage, out_dir, events, proposal_class, *, allowed_types=None):
        self.generate_fn = generate_fn
        self.stage = stage
        self.out_dir = out_dir
        self.events = events
        self.proposal_class = proposal_class
        self.allowed_types = (
            allowed_types if allowed_types is not None else (PROPOSAL_TYPES if stage == "propose" else REFINEMENT_TYPES)
        )

    def __call__(self, contents):
        from PIL import Image
        from simpact.generator.vlm import strip_json_fences

        # Upstream returns lazy PIL images. Load pixels now so their files close
        # even when an injected provider does not inspect the image contents.
        for content in contents:
            if isinstance(content, Image.Image):
                content.load()
        for retry in range(2):
            index = len(self.events)
            event = {"stage": self.stage, "retry": retry}
            self.events.append(event)
            started = time.monotonic()
            try:
                raw = self.generate_fn(contents)
                response_path = self.out_dir / f"response_{index:03d}_{self.stage}.txt"
                response_path.write_text(raw)
                event["response_path"] = str(response_path)
                value = json.loads(strip_json_fences(raw))
                if self.stage == "verify":
                    if not isinstance(value, dict) or type(value.get("success")) is not bool:
                        raise ValueError("verifier success must be a JSON boolean")
                else:
                    plans = value["action_proposals"]
                    event["proposal_count"] = len(plans)
                    expected = INITIAL_CANDIDATES if self.stage == "propose" else 1
                    if len(plans) != expected:
                        raise ValueError(f"{self.stage} returned {len(plans)} proposals; required {expected}")
                    proposals = self.proposal_class.from_dict(value)
                    errors = proposals.validate(allowed_types=self.allowed_types)
                    if errors:
                        raise ValueError("; ".join(errors[:5]))
                    for plan in proposals.action_proposals:
                        if not plan.action_sequence:
                            raise ValueError("empty action_sequence")
                        if any(not math.isfinite(v) for a in plan.action_sequence for v in a.attr_values().values()):
                            raise ValueError("nonfinite action parameter")
                return raw
            except (ValueError, KeyError, TypeError) as error:
                event["error"] = str(error)
                if retry:
                    raise ValueError(f"{self.stage} schema failed after two responses: {error}") from error
            except Exception as error:
                event["error"] = str(error)
                raise
            finally:
                event["wall_time_s"] = time.monotonic() - started
                _write_json(self.out_dir / "vlm_calls.json", self.events)
        raise AssertionError("unreachable")


def run_planning(
    *,
    observation_json,
    instruction,
    out_dir,
    rollout_fn,
    provider,
    model,
    generate_fn=None,
):
    """Run ten proposals, at most three refinements, and one successful-plan replay.

    ``rollout_fn(plan, index, out_dir) -> rollout_json_path`` executes a complete
    action sequence in a fresh simulator process. ``generate_fn`` permits explicit
    provider injection for core offline checks. No provider/model fallback occurs.
    Errors are saved with status='error' and do not become task failures.
    """
    started = time.monotonic()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "error",
        "success": False,
        "planning_success": False,
        "confirmation_success": None,
        "provider": provider,
        "model": model,
        "simpact_revision": SIMPACT_REVISION,
        "instruction": instruction,
        "requested_proposal_count": INITIAL_CANDIDATES,
        "actual_proposal_count": None,
        "max_refinement_iterations": MAX_REFINEMENT_ITERATIONS,
        "refinement_iterations": 0,
        "planning_rollouts_attempted": 0,
        "planning_rollouts_completed": 0,
        "planning_rollouts_rejected": 0,
        "confirmation_rollouts_attempted": 0,
        "confirmation_rollouts_completed": 0,
        "confirmation_rollouts_rejected": 0,
        "invalid_proposals": [],
        "attempts": [],
        "privileged_initial_state": True,
    }
    try:
        observation_path = Path(observation_json).resolve()
        observation = json.loads(observation_path.read_text())
        initial = observation["snapshots"][0]
        result["task"] = observation.get("task", {})
        hanging = result["task"].get("name") == "clawhauser_hang"
        profile = "clawhauser_hang" if hanging else "placement"
        result["planner_profile"] = profile
        proposal_types = REFINEMENT_TYPES if hanging else PROPOSAL_TYPES
        prompt_prefix = "hang_" if hanging else ""
        result["geometry"] = observation.get("geometry", {})
        result["revisions"] = {**observation.get("revisions", {}), "simpact": SIMPACT_REVISION}
        result["initial_state"] = observation.get("initial_state", initial)
        photo = observation_path.parent / initial["screenshot"]
        if not photo.is_file():
            raise ValueError(f"initial scene image is missing: {photo}")
        # Preserve the input observation and make its image independently resolvable.
        for snapshot in observation["snapshots"]:
            if snapshot.get("screenshot"):
                snapshot["screenshot"] = str((observation_path.parent / snapshot["screenshot"]).resolve())
        _write_json(out / "observation.json", observation)
        context = (
            "Known privileged simulation state, not a perception/reconstruction result.\n"
            "World-frame metres, right-handed rotations in radians, quaternion xyzw, total jaw opening.\n"
            + json.dumps({"task": result["task"], "geometry": result["geometry"], "initial_state": initial})
        )
        if provider not in {"google", "codex"} and generate_fn is None:
            raise ValueError(f"unsupported provider {provider!r}; no model/provider substitution is allowed")
        if not model:
            raise ValueError("an explicit model identifier is required")
        if generate_fn is None and provider == "google" and not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is not configured; cannot run the requested Google model")
        result["simpact_revision"] = _require_pinned_simpact()
        from simpact.actions import ProposalSet
        from simpact.generator.optimize_loop import OptimizationLoop, attach_verdict
        from simpact.generator.propose import VLMProposer
        from simpact.generator.regress import RegressOptimizer
        from simpact.generator.verify import TaskVerifier
        from simpact.generator.vlm import gemini_generate

        events = []
        raw_dir = out / "vlm"
        raw_dir.mkdir()

        def recorded(stage):
            if generate_fn is not None:
                call_model = generate_fn
            elif provider == "codex":
                from .providers import make_codex_generate_fn

                call_model = make_codex_generate_fn(model, stage, profile=profile)
            else:

                def call_model(contents):
                    return gemini_generate(contents, model_id=model)

            return _RecordedGenerator(
                call_model,
                stage,
                raw_dir,
                events,
                ProposalSet,
                allowed_types=proposal_types if stage == "propose" else REFINEMENT_TYPES,
            )

        proposals = VLMProposer(
            prompt_template=str(PROMPTS / f"{prompt_prefix}propose.txt"),
            generate_fn=recorded("propose"),
            model_id=model,
        ).propose(instruction, photo, context=context, retries=0, allowed_types=proposal_types)
        result["actual_proposal_count"] = len(proposals.action_proposals)
        proposals.to_json(out / "proposals.json")
        memory = out / "rollouts"
        memory.mkdir()

        def execute(plan, *, confirmation=False):
            phase = "confirmation" if confirmation else "planning"
            key = f"{phase}_rollouts_attempted"
            index = result[key]
            result[key] += 1
            folder = out / "confirmation" if confirmation else memory
            folder.mkdir(exist_ok=True)
            path = Path(rollout_fn(plan, index, folder / f"attempt_{index:03d}")).resolve()
            try:
                parse_airbot_rollout(path)  # Upstream's trace wrapper suppresses parser errors.
                measured_success_gate(path)
                data = json.loads(path.read_text())
            except (ValueError, KeyError, TypeError) as error:
                raise RuntimeError(f"invalid worker rollout schema at {path}: {error}") from error
            for snap in data["snapshots"]:
                if snap.get("screenshot"):
                    snap["screenshot"] = str((path.parent / snap["screenshot"]).resolve())
            data["source_rollout_path"] = str(path)
            mirror = folder / f"rollout_{index:03d}.json"
            _write_json(mirror, data)
            if data.get("execution_status") == "rejected":
                result[f"{phase}_rollouts_rejected"] += 1
                result["invalid_proposals"].append({"index": index, "phase": phase, **data["execution_error"]})
            else:
                result[f"{phase}_rollouts_completed"] += 1
            return mirror

        candidates = []
        for index, plan in enumerate(proposals.action_proposals):
            try:
                candidates.append((plan, execute(plan)))
            except ValueError as error:
                result["invalid_proposals"].append({"index": index, "error": str(error)})
        if not candidates:
            raise ValueError("no initial proposal could be executed; optimization cannot start")
        verifier = TaskVerifier(
            prompt_template=str(PROMPTS / f"{prompt_prefix}verify.txt"),
            generate_fn=recorded("verify"),
            model_id=model,
            success_gate=measured_success_gate,
        )
        regressor = RegressOptimizer(
            prompt_template=str(PROMPTS / f"{prompt_prefix}refine.txt"),
            generate_fn=recorded("refine"),
            model_id=model,
            parse_rollout=parse_airbot_rollout,
        )

        def event(stage, attempt):
            result["attempts"].append(
                {
                    "stage": stage,
                    "kind": attempt.kind,
                    "index": int(attempt.rollout_path.stem.rsplit("_", 1)[1]),
                    "rollout_path": str(attempt.rollout_path),
                    "verdict": attempt.verdict.to_dict(),
                }
            )
            result["refinement_iterations"] = sum(a["kind"] == "refined" for a in result["attempts"])

        loop = OptimizationLoop(
            regressor,
            verifier,
            rollout_fn=lambda plan, index: execute(plan),
            max_iters=MAX_REFINEMENT_ITERATIONS,
            accumulate_dir=memory,
            parse_rollout=parse_airbot_rollout,
        )
        planned = loop.run(candidates, instruction, context, on_event=event)
        result["planning_success"] = planned.success
        result["refinement_iterations"] = planned.iterations
        planned.best_proposalset.to_json(out / "selected_plan.json")
        result["selected_plan_path"] = str(out / "selected_plan.json")
        if planned.success:
            confirmation = execute(planned.best_plan, confirmation=True)
            verdict = verifier.verify(
                confirmation,
                instruction,
                context,
                action_trace=parse_airbot_rollout(confirmation)["text"],
                retries=0,
            )
            attach_verdict(confirmation, verdict)
            result["confirmation_success"] = verdict.success
            result["confirmation_verdict"] = verdict.to_dict()
            result["confirmation_rollout_path"] = str(confirmation)
            result["success"] = verdict.success
        result["status"] = "success" if result["success"] else "failure"
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if result["actual_proposal_count"] is None:
            counts = [
                e["proposal_count"]
                for e in locals().get("events", [])
                if e["stage"] == "propose" and "proposal_count" in e
            ]
            if counts:
                result["actual_proposal_count"] = counts[-1]
        result["wall_time_s"] = time.monotonic() - started
        _write_json(out / "result.json", result)
    return result
