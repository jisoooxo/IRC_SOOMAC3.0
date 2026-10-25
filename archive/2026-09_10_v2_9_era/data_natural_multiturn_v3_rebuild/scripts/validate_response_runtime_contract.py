#!/usr/bin/env python3
"""Response 1,225행이 현재 graph/node가 만들 수 있는 구조인지 검사한다."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data_natural_multiturn_v3_rebuild"
STATUS_BY_REASON = {
    "understanding": "clarify",
    "general": "pass",
    "completed_restriction_conflict": "blocked",
    "restriction_conflict": "warning",
    "unsupported": "warning",
    "physical_state": "warning",
    "invalid_candidate": "warning",
    "recommendation_proposed": "pass",
    "recommendation_unavailable": "clarify",
    "confirmation_rejected": "pass",
    "missing_order": "clarify",
    "robot_busy": "blocked",
    "execution_allowed": "pass",
    "state_update_only": "pass",
    "section_transition": "pass",
}
ISSUE_KEYS = {
    "unsupported", "invalid", "protected",
    "restriction_conflicts", "physical_conflicts",
}
ACTION_TYPES = {"turn_applied", "task_completed", "section_transition", "vlm_result"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def task_shape(task: object) -> bool:
    return (
        isinstance(task, dict)
        and isinstance(task.get("class"), str)
        and isinstance(task.get("repeat_count"), int)
        and "amount" not in task
    )


def row_failures(row: dict) -> list[str]:
    model_input = row["input"]
    policy = model_input.get("policy") or {}
    reason = policy.get("reason")
    failures = []
    if reason not in STATUS_BY_REASON:
        failures.append(f"unknown_policy_reason:{reason}")
    elif policy.get("status") != STATUS_BY_REASON[reason]:
        failures.append(f"policy_status:{policy.get('status')}!={STATUS_BY_REASON[reason]}")

    pending = model_input.get("pending")
    prompt = model_input.get("next_prompt")
    prompt_type = prompt.get("type") if isinstance(prompt, dict) else None
    issues = policy.get("issues")
    if isinstance(issues, dict) and any(issues.values()):
        if set(issues) != ISSUE_KEYS:
            failures.append("policy_issue_keys")
        if prompt_type != "validation_issues" or prompt.get("issues") != issues:
            failures.append("validation_issues_prompt")
        for key in ("unsupported", "protected", "invalid", "restriction_blocked"):
            if key not in policy:
                failures.append(f"missing_policy_compat_field:{key}")
    elif reason == "missing_order":
        if prompt_type != "missing_order" or prompt.get("fields") != policy.get("missing", []):
            failures.append("missing_order_prompt")
    elif reason == "section_transition":
        if prompt_type not in {"section_prompt", "execution", None}:
            failures.append(f"section_transition_prompt:{prompt_type}")
    elif isinstance(pending, dict):
        if prompt_type != pending.get("type"):
            failures.append("pending_prompt_type")
    elif prompt is not None:
        failures.append(f"orphan_next_prompt:{prompt_type}")

    if prompt_type == "section_selection":
        failures.append("non_runtime_section_selection")
    if isinstance(pending, dict) and pending.get("type") == "execution":
        source = pending.get("source", "current_update")
        if source == "current_update":
            if "items" in pending or "candidate" in pending:
                failures.append("current_update_pending_extra_fields")
            if not isinstance(prompt, dict) or prompt.get("items") != {}:
                failures.append("current_update_prompt_items")
        elif source == "preselected":
            if not isinstance(pending.get("items"), dict) or not isinstance(pending.get("candidate"), dict):
                failures.append("preselected_pending_shape")

    state = model_input.get("robot_state") or {}
    tasks = list(state.get("task_queue") or []) + list(state.get("completed_tasks") or [])
    if state.get("active_task") is not None:
        tasks.append(state["active_task"])
    if any(not task_shape(task) for task in tasks):
        failures.append("robot_task_shape")
    if state.get("active_task") is not None and state.get("robot_started") is not True:
        failures.append("active_without_robot_started")
    if state.get("completed_tasks") and state.get("robot_started") is not True:
        failures.append("completed_without_robot_started")

    for event in model_input.get("recent_action_history", []):
        event_type = event.get("type") if isinstance(event, dict) else None
        if event_type not in ACTION_TYPES:
            failures.append(f"action_history_type:{event_type}")
        elif event_type == "task_completed" and not task_shape(event.get("task")):
            failures.append("task_completed_event_shape")
        elif event_type == "turn_applied":
            required = {"order_changes", "restriction_added", "restriction_removed", "preference_added", "preference_removed"}
            if not required.issubset(event):
                failures.append("turn_applied_event_shape")

    if reason == "recommendation_unavailable":
        result = model_input.get("recommendation_result")
        if not isinstance(result, dict) or result.get("accepted_fields") != [] or not result.get("rejected_fields"):
            failures.append("recommendation_unavailable_result")
    if reason == "section_transition":
        transition = state.get("section_transition")
        if not isinstance(transition, dict) or transition not in model_input.get("recent_action_history", []):
            failures.append("section_transition_state")
    return failures


def main() -> None:
    rows = read_jsonl(BASE / "intermediate/response_v1_intermediate.jsonl")
    failures = []
    family_failures = Counter()
    for row in rows:
        errors = row_failures(row)
        if errors:
            failures.append({"id": row["id"], "family": row["metadata"].get("scenario_family"), "errors": errors})
            family_failures[row["metadata"].get("scenario_family", "unknown")] += 1

    fixed_families = Counter()
    fixed_rows = 0
    for row in rows:
        if "runtime_contract_mismatch" in row.get("metadata", {}).get("cleanup_error_class", []):
            fixed_rows += 1
            fixed_families[row["metadata"].get("scenario_family", "unknown")] += 1
    report = {
        "status": "PASS" if not failures and len(rows) == 1225 else "FAIL",
        "rows": len(rows),
        "runtime_contract_mismatch_found": fixed_rows,
        "runtime_contract_mismatch_fixed": fixed_rows,
        "runtime_contract_mismatch_remaining": len(failures),
        "fixed_families": dict(fixed_families),
        "failure_families": dict(family_failures),
        "failures": failures[:50],
        "canonical_sources": [
            "src/soomac_irc/soomac_irc/llm_langgraph.py",
            "src/soomac_irc/soomac_irc/llm_langgraph_node.py",
            "src/soomac_irc/soomac_irc/response_model.py",
        ],
    }
    path = BASE / "reports/response_runtime_contract.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
