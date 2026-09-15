"""Single-run pipeline logic, shared by scripts/run_pipeline.py and stage7_eval.

Staged observe-act loop (P3 in stage3_policy/CONTRACT_PROPOSAL.md): each attempt
repeatedly calls plan_detailed with the step ids already executed, runs the
returned actions, banks the contiguous successful prefix of action_results,
and re-observes, until the plan is complete or execution fails.

Recovery: if verify() returns replan=True, re-run the staged loop (keeping the
completed step ids) from the fresh observation, up to max_retries (from
configs/default.yaml), then report FAIL. A run succeeds only when the plan
completed, execution reported success AND verify accepts the scene.
"""

from pathlib import Path

import yaml

from common.types import RunResult

_ROOT = Path(__file__).resolve().parents[1]
_FALLBACK_MAX_RETRIES = 2


def _config_max_retries() -> int:
    try:
        cfg = yaml.safe_load((_ROOT / "configs" / "default.yaml").read_text())
        return int(cfg["max_retries"])
    except Exception:
        return _FALLBACK_MAX_RETRIES


def run_once(command: str, seed: int = 0, max_retries: int | None = None) -> RunResult:
    """Run the full pipeline once for a seed: staged planning + verify->replan recovery."""
    from stage1_voice import parse_text
    from stage2_perception import perceive
    from stage3_policy import PlanningError, plan_detailed
    from stage4_bimanual import execute, get_camera_frame, reset_scene
    from stage6_verify import verify

    if max_retries is None:
        max_retries = _config_max_retries()

    log: list[str] = []

    log.append(f"[sim]      stage4_bimanual.reset_scene(seed={seed})")
    sim = reset_scene(seed)

    log.append(f'[voice]    stage1_voice.parse_text("{command}")')
    task = parse_text(command)
    log.append(f"           -> Task with {len(task.steps)} steps:")
    for step in task.steps:
        deps = f" (after {step.depends_on})" if step.depends_on else ""
        log.append(f"              {step.id}. {step.action.value:<12} arm {step.arm}{deps}")

    log.append("[perceive] stage2_perception.perceive(get_camera_frame(sim), sim=sim)")
    scene = perceive(get_camera_frame(sim), sim=sim)
    log.append(f"           -> {len(scene.objects)} objects: {', '.join(scene.objects)}")
    log.append(f"           -> drawers: {dict(scene.drawers)}")

    completed: set[int] = set()  # successfully executed step ids, kept across replans
    attempts = 0
    success = False
    while True:  # verify -> replan recovery loop
        attempts += 1
        planning_refused = False
        execution = None
        plan_complete = False

        while True:  # staged observe-act loop (CONTRACT_PROPOSAL.md P3)
            log.append(
                "[plan]     stage3_policy.plan_detailed(task, scene, "
                f"completed_step_ids={sorted(completed)})  (attempt {attempts})"
            )
            try:
                result = plan_detailed(task, scene, completed_step_ids=completed)
            except PlanningError as exc:
                log.append(f"           -> planning refused ({type(exc).__name__}): {exc}")
                planning_refused = True
                break
            plan_complete = result.complete
            if result.complete:
                log.append(f"           -> {len(result.actions)} executable actions (plan complete)")
            else:
                log.append(
                    f"           -> {len(result.actions)} executable actions now, "
                    f"pending steps {list(result.pending_step_ids)}: {result.blocked_reason}"
                )

            log.append("[execute]  stage4_bimanual.execute(actions, sim)")
            execution = execute(list(result.actions), sim)
            ok = sum(execution.action_results.values())
            log.append(f"           -> {ok}/{len(result.actions)} actions succeeded")
            if execution.error:
                log.append(f"           -> error: {execution.error}")

            newly_completed = 0
            for action in result.actions:  # only the contiguous successful prefix counts
                if not execution.action_results.get(action.step_id):
                    break
                if action.step_id not in completed:
                    newly_completed += 1
                completed.add(action.step_id)

            log.append("[perceive] stage2_perception.perceive(get_camera_frame(sim), sim=sim)  (fresh observation)")
            scene = perceive(get_camera_frame(sim), sim=sim)

            if not execution.success or result.complete:
                break
            if newly_completed == 0:
                # The plan is still incomplete and no step finished, so the next
                # plan_detailed call would see the same state; stop instead of
                # looping forever and let verify decide whether to replan.
                log.append("           -> no progress this stage; leaving the observe-act loop")
                break

        if planning_refused and execution is None:
            # Nothing was executed this attempt, so there is no new scene worth
            # verifying: report FAIL exactly like the pre-staged pipeline did.
            break

        log.append("[verify]   stage6_verify.verify(scene, task)  (scene = last staged observation)")
        verdict = verify(scene, task)
        log.append(f"           -> ok={verdict.ok} replan={verdict.replan} ({verdict.details})")

        if verdict.ok and execution is not None and execution.success and plan_complete:
            success = True
            break
        if verdict.ok and (execution is None or not execution.success):
            log.append("           -> execution reported failure, so verify ok is not counted as success")
        elif verdict.ok:
            log.append("           -> the plan never completed, so verify ok is not counted as success")
        if verdict.replan and attempts < max_retries:
            log.append(
                "           -> replan requested: retrying from the new observation, "
                f"keeping completed steps {sorted(completed)} ({attempts + 1}/{max_retries})"
            )
            continue
        break

    return RunResult(success=success, attempts=attempts, task=task, log=log)
