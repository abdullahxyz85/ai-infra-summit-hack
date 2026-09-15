"""UNIT INTEGRATION tests for common.pipeline.run_once recovery logic.

Every stage is replaced by a controlled double, so these tests show how run_once
wires stage outputs together. They are not physical validation and say nothing
about whether the simulated robot can do the task.
"""

import sys
import types

import pytest

from common.pipeline import run_once
from common.types import Action, ActionType, ExecutionResult, SceneState, Task, TaskStep, VerifyResult

TASK = Task(command="pick mug", steps=[TaskStep(id=1, action=ActionType.PICK, arm="B", object="mug")])


class DoublePlanningError(Exception):
    """Stands in for stage3_policy.PlanningError."""


def _scene(x: float) -> SceneState:
    return SceneState(objects={"mug": (x, 0.18, 0.70)}, drawers={"top_drawer": "open"})


def _plan_result(actions, *, complete=True, pending=()):
    """A PlanResult-shaped double (only the fields run_once reads)."""
    return types.SimpleNamespace(
        actions=tuple(actions),
        complete=complete,
        pending_step_ids=tuple(pending),
        blocked_step_id=pending[0] if pending else None,
        blocked_reason="double: needs a fresh observation" if pending else None,
    )


@pytest.fixture
def install_stages(monkeypatch):
    """Replace every stage module with scripted doubles; returns a record of calls."""

    # run_once now drives stage 3 through plan_detailed (staged observe-act loop,
    # CONTRACT_PROPOSAL.md P3) instead of plan(), so the double fakes plan_detailed.
    # The behaviours the original tests assert are unchanged.
    def install(*, observations, execution_success=(), verdicts=(), plan_error=None, task=TASK, plan_results=None):
        observations, execution_success, verdicts = list(observations), list(execution_success), list(verdicts)
        calls = {"plan_scenes": [], "plan_completed": [], "verify_scenes": [], "executions": 0}

        def plan_detailed(task_, scene, *, completed_step_ids=()):
            calls["plan_scenes"].append(scene)
            calls["plan_completed"].append(set(completed_step_ids))
            if plan_error is not None:
                raise plan_error
            if plan_results is not None:
                scripted = plan_results.pop(0)
                if isinstance(scripted, Exception):
                    raise scripted
                return scripted
            return _plan_result(
                [Action(step_id=1, action=ActionType.PICK, arm="B", object="mug", target_pose=scene.objects["mug"])]
            )

        def execute(actions, sim=None):
            calls["executions"] += 1
            scripted = execution_success.pop(0)  # bool for all actions, or dict step_id -> bool
            per_action = scripted if isinstance(scripted, dict) else {a.step_id: scripted for a in actions}
            action_results = {a.step_id: per_action[a.step_id] for a in actions}
            ok = all(action_results.values())  # mirrors the real executor's success flag
            return ExecutionResult(
                action_results=action_results,
                success=ok,
                final_scene=_scene(0.0),
                error=None if ok else "double: grasp slipped",
            )

        def verify(scene_after, task_):
            calls["verify_scenes"].append(scene_after)
            return verdicts.pop(0)

        doubles = {
            "stage1_voice": {"parse_text": lambda text: task},
            "stage2_perception": {"perceive": lambda image=None, sim=None: observations.pop(0)},
            "stage3_policy": {"plan_detailed": plan_detailed, "PlanningError": DoublePlanningError},
            "stage4_bimanual": {"execute": execute, "get_camera_frame": lambda sim: None, "reset_scene": lambda seed: object()},
            "stage6_verify": {"verify": verify},
        }
        for name, attributes in doubles.items():
            module = types.ModuleType(name)
            for attribute, value in attributes.items():
                setattr(module, attribute, value)
            monkeypatch.setitem(sys.modules, name, module)
        return calls

    return install


def test_retry_plans_from_the_post_execution_observation(install_stages):
    initial, after_first, after_second = _scene(0.06), _scene(0.10), _scene(0.10)
    calls = install_stages(
        observations=[initial, after_first, after_second],
        execution_success=[True, True],
        verdicts=[VerifyResult(ok=False, replan=True), VerifyResult(ok=True)],
    )
    result = run_once("pick mug", max_retries=2)

    assert result.success and result.attempts == 2
    assert calls["plan_scenes"] == [initial, after_first]


def test_failed_execution_is_not_success_even_if_verify_accepts(install_stages):
    install_stages(observations=[_scene(0.06), _scene(0.06)], execution_success=[False], verdicts=[VerifyResult(ok=True)])
    result = run_once("pick mug", max_retries=2)

    assert not result.success and result.attempts == 1
    assert any("not counted as success" in line for line in result.log)


def test_failed_execution_can_recover_through_a_replan(install_stages):
    install_stages(
        observations=[_scene(0.06), _scene(0.07), _scene(0.07)],
        execution_success=[False, True],
        verdicts=[VerifyResult(ok=False, replan=True), VerifyResult(ok=True)],
    )
    result = run_once("pick mug", max_retries=2)
    assert result.success and result.attempts == 2


def test_planning_refusal_reports_fail_without_executing(install_stages):
    calls = install_stages(observations=[_scene(0.06)], plan_error=DoublePlanningError("'mug' is not in the observed scene"))
    result = run_once("pick mug", max_retries=2)

    assert not result.success and result.attempts == 1
    assert calls["executions"] == 0
    assert any("planning refused" in line and "not in the observed scene" in line for line in result.log)


def test_staged_planning_executes_prefix_then_completes(install_stages):
    """P3: an incomplete plan executes its prefix, re-observes, then plans the rest."""
    two_step_task = Task(
        command="pour into held mug",
        steps=[
            TaskStep(id=1, action=ActionType.PICK, arm="B", object="mug"),
            TaskStep(id=2, action=ActionType.POUR, arm="A", source="water_bottle", into="mug", depends_on=[1]),
        ],
    )
    initial, after_pick, after_pour = _scene(0.06), _scene(0.10), _scene(0.12)
    pick = Action(step_id=1, action=ActionType.PICK, arm="B", object="mug")
    pour = Action(step_id=2, action=ActionType.POUR, arm="A", object="water_bottle")
    calls = install_stages(
        task=two_step_task,
        observations=[initial, after_pick, after_pour],
        execution_success=[True, True],
        verdicts=[VerifyResult(ok=True)],
        plan_results=[_plan_result([pick], complete=False, pending=(2,)), _plan_result([pour])],
    )
    result = run_once("pour into held mug", max_retries=2)

    assert result.success and result.attempts == 1
    assert calls["executions"] == 2
    # the second planning stage sees the fresh observation and the completed prefix
    assert calls["plan_scenes"] == [initial, after_pick]
    assert calls["plan_completed"] == [set(), {1}]
    # verify judges the last staged observation, not the initial or mid-loop one
    assert calls["verify_scenes"] == [after_pour]


def test_only_the_contiguous_successful_prefix_is_banked(install_stages):
    """A failed step is not banked, and neither is a success that comes after the failure."""
    three_step_task = Task(
        command="three picks",
        steps=[
            TaskStep(id=1, action=ActionType.PICK, arm="A", object="plate"),
            TaskStep(id=2, action=ActionType.PICK, arm="B", object="mug"),
            TaskStep(id=3, action=ActionType.PICK, arm="A", object="spoon"),
        ],
    )
    acts = [
        Action(step_id=1, action=ActionType.PICK, arm="A", object="plate"),
        Action(step_id=2, action=ActionType.PICK, arm="B", object="mug"),
        Action(step_id=3, action=ActionType.PICK, arm="A", object="spoon"),
    ]
    calls = install_stages(
        task=three_step_task,
        observations=[_scene(0.06), _scene(0.10), _scene(0.12)],
        execution_success=[{1: True, 2: False, 3: True}, True],
        verdicts=[VerifyResult(ok=False, replan=True), VerifyResult(ok=True)],
        plan_results=[_plan_result(acts), _plan_result(acts[1:])],
    )
    result = run_once("three picks", max_retries=2)

    assert result.success and result.attempts == 2
    # step 2 failed, so step 3's success does not count: only step 1 is banked for the replan
    assert calls["plan_completed"] == [set(), {1}]


def test_replan_keeps_completed_step_ids(install_stages):
    """The verify->replan retry must not forget which steps already executed."""
    calls = install_stages(
        observations=[_scene(0.06), _scene(0.10), _scene(0.10)],
        execution_success=[True, True],
        verdicts=[VerifyResult(ok=False, replan=True), VerifyResult(ok=True)],
    )
    result = run_once("pick mug", max_retries=2)

    assert result.success
    assert calls["plan_completed"] == [set(), {1}]


def test_planning_refusal_after_partial_execution_still_verifies(install_stages):
    """A mid-run refusal (e.g. observation contradicts completed steps) reaches verify.

    Only a refusal before anything executed skips verify (there is no new scene);
    after a prefix ran, verify judges the scene and may request a replan.
    """
    two_step_task = Task(
        command="open then pick",
        steps=[
            TaskStep(id=1, action=ActionType.OPEN_DRAWER, arm="A", target="top_drawer"),
            TaskStep(id=2, action=ActionType.PICK, arm="A", object="plate", depends_on=[1]),
        ],
    )
    opener = Action(step_id=1, action=ActionType.OPEN_DRAWER, arm="A", object="top_drawer")
    calls = install_stages(
        task=two_step_task,
        observations=[_scene(0.06), _scene(0.10)],
        execution_success=[True],
        verdicts=[VerifyResult(ok=False, replan=False)],
        plan_results=[
            _plan_result([opener], complete=False, pending=(2,)),
            DoublePlanningError("step 1 is reported complete, but the drawer is observed closed"),
        ],
    )
    result = run_once("open then pick", max_retries=2)

    assert not result.success and result.attempts == 1
    assert calls["executions"] == 1
    assert any("planning refused" in line for line in result.log)
    assert any(line.startswith("[verify]") for line in result.log)


def test_incomplete_plan_without_progress_does_not_hang_or_succeed(install_stages):
    """A plan that stays incomplete while nothing executes must end as FAIL, not loop forever."""
    calls = install_stages(
        observations=[_scene(0.06), _scene(0.06)],
        execution_success=[True],  # executing zero actions still "succeeds"
        verdicts=[VerifyResult(ok=True)],
        plan_results=[_plan_result([], complete=False, pending=(1,))],
    )
    result = run_once("pick mug", max_retries=2)

    assert not result.success and result.attempts == 1
    assert calls["executions"] == 1
    assert any("no progress" in line for line in result.log)


def test_retries_stop_at_max_retries(install_stages):
    calls = install_stages(
        observations=[_scene(0.06), _scene(0.06), _scene(0.06), _scene(0.06)],
        execution_success=[True, True, True],
        verdicts=[VerifyResult(ok=False, replan=True)] * 3,
    )
    result = run_once("pick mug", max_retries=3)

    assert not result.success and result.attempts == 3
    assert calls["executions"] == 3
