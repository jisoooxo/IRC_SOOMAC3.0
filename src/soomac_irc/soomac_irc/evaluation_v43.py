"""SOOMAC audit protocol v4.3 semantic evaluator.

The evaluator grades structured meaning, canonical state, policy, and canonicalized
plan events.  It never mutates runtime state and does not decide user intent from
keywords.  Natural-language variants are represented by separate cases sharing the
same semantic oracle.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from soomac_irc.domain import (
    AMOUNTS,
    NOODLE_TYPES,
    RESTRICTION_CATEGORY_ITEMS,
    SAUCES,
    SECTION_ORDER,
    TOPPINGS,
)


_MISSING = object()
_CORE_SESSION_FIELDS = ("order", "preferences", "pending")
_EXECUTION_STARTED = re.compile(r"(?:담기|조리|작업|투입).{0,8}(?:시작|진행)|시작(?:할게|합니다|했)")
_COMPLETION_CLAIM = re.compile(r"(?:완료|다\s*됐|끝났|성공(?:했|입니다)|담았(?:어요|습니다))")
_AMOUNT_CUE = re.compile(r"적게|조금|소량|보통|평범|많이|넉넉|듬뿍|low|normal|high", re.I)
_AMOUNT_PATTERNS = {
    "low": re.compile(r"적(?:게|은)|조금|소량|low", re.I),
    "normal": re.compile(r"보통|평범|normal", re.I),
    "high": re.compile(r"많(?:이|은)|넉넉|듬뿍|푸짐|high", re.I),
}
_INTERNAL_RESPONSE = re.compile(
    r"noodle_portion|noodle_type|toppings\.|policy\.reason|pending(?:\.type)?|"
    r"execution_authorized|starting_now|recommendation_scope"
)


def _claims_execution_started(reply: str) -> bool:
    if not _EXECUTION_STARTED.search(reply):
        return False
    return not bool(re.search(r"(?:할|해|갈)\s*까요\??|\?\s*$", reply))


def _claims_completion(reply: str) -> bool:
    if not _COMPLETION_CLAIM.search(reply):
        return False
    negative = re.search(r"(?:완료된|끝난|담은).{0,12}(?:아직\s*)?(?:없|아니)|아직.{0,12}(?:완료|끝|담).{0,8}(?:않|못)", reply)
    return negative is None


@dataclass(frozen=True)
class Violation:
    code: str
    path: str
    expected: Any = None
    actual: Any = None


@dataclass(frozen=True)
class Grade:
    case_id: str
    passed: bool
    violations: tuple[Violation, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _contains_pattern(expected: Any, actual: Any, path: str) -> list[Violation]:
    """Match a semantic subset without requiring one literal JSON serialization."""
    if isinstance(expected, dict):
        if "$one_of" in expected:
            candidates = expected["$one_of"]
            if any(not _contains_pattern(item, actual, path) for item in candidates):
                return []
            return [Violation("expected_one_of", path, candidates, actual)]
        if "$contains" in expected:
            if not isinstance(actual, list):
                return [Violation("expected_list", path, expected, actual)]
            failures = []
            for index, item in enumerate(expected["$contains"]):
                if not any(not _contains_pattern(item, candidate, path) for candidate in actual):
                    failures.append(Violation("missing_list_member", f"{path}[{index}]", item, actual))
            return failures
        if not isinstance(actual, dict):
            return [Violation("expected_object", path, expected, actual)]
        failures = []
        for key, value in expected.items():
            child = actual.get(key, _MISSING)
            child_path = f"{path}.{key}" if path else key
            if child is _MISSING:
                failures.append(Violation("missing_field", child_path, value, None))
            else:
                failures.extend(_contains_pattern(value, child, child_path))
        return failures
    if expected != actual:
        return [Violation("value_mismatch", path, expected, actual)]
    return []


def _flatten_changes(before: Any, after: Any, prefix: str = "") -> set[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: set[str] = set()
        for key in set(before) | set(after):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                changes.add(path)
            else:
                changes.update(_flatten_changes(before[key], after[key], path))
        return changes
    if before != after:
        return {prefix}
    return set()


def _path_allowed(path: str, allowed: list[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}.") for prefix in allowed)


def _decision_value(decision: dict, sparse_key: str, normalized_key: str) -> Any:
    if sparse_key in decision:
        return decision[sparse_key]
    return decision.get(normalized_key, _MISSING)


def _canonical_decision(decision: dict) -> dict:
    """Expose sparse and normalized Decision outputs through one grading shape."""
    canonical = copy.deepcopy(decision)
    aliases = {
        "order": "order_patch",
        "restrictions": "restriction_options",
        "preferences": "preference_options",
    }
    for sparse, normalized in aliases.items():
        value = _decision_value(decision, sparse, normalized)
        if value is not _MISSING:
            canonical[sparse] = copy.deepcopy(value)
    return canonical


def _decision_meaning(decision: dict) -> dict:
    """Compare sparse/normalized outputs, ignoring only documented no-op defaults."""
    value = _canonical_decision(decision)
    for key in ("order_patch", "restriction_options", "preference_options"):
        value.pop(key, None)
    understanding = value.pop("understanding", None)
    if understanding == "clarify":
        value["clarify"] = True
    for key in ("commit", "skip_current", "clarify"):
        if value.get(key) is False:
            value.pop(key)
    if value.get("confirmation") == "none":
        value.pop("confirmation")
    if value.get("recommendation") == {"action": "none"}:
        value.pop("recommendation")
    if isinstance(value.get("order"), dict):
        value["order"] = {k: v for k, v in value["order"].items() if v is not None and v != {}}
        if not value["order"]:
            value.pop("order")
    for key in ("restrictions", "preferences"):
        if value.get(key) == []:
            value.pop(key)
        elif isinstance(value.get(key), list):
            value[key] = sorted(value[key], key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    return value


def _section_for_plan(plan: dict) -> str | None:
    section = plan.get("section")
    if section is not None:
        return section
    item = plan.get("item")
    if item in NOODLE_TYPES:
        return "noodle"
    if item in {"양파", "버섯"}:
        return "veggie"
    if item in {"소시지", "게살"}:
        return "meat"
    if item in {"치즈", "페퍼론치노"}:
        return "extra"
    if item == "뚜껑":
        return "lid"
    if item in SAUCES:
        return "sauce"
    return None


def _restriction_targets(session: dict) -> set[str]:
    targets: set[str] = set()
    for restriction in (session.get("order") or {}).get("restrictions", []):
        if not isinstance(restriction, dict):
            continue
        target = restriction.get("target")
        if target:
            targets.add(str(target))
            targets.update(RESTRICTION_CATEGORY_ITEMS.get(str(target), ()))
    return targets


def _plan_matches_order(plan: dict, session: dict) -> bool:
    order = session.get("order") or {}
    item = plan.get("item")
    amount = plan.get("amount")
    section = _section_for_plan(plan)
    if section == "noodle":
        if item is not None and item != order.get("noodle_type"):
            return False
        return amount is None or amount == order.get("noodle_portion")
    if section in {"veggie", "meat", "extra"}:
        return item in TOPPINGS and (order.get("toppings") or {}).get(item) == amount
    if section == "sauce":
        return item == order.get("sauce")
    return section == "lid" and item == "뚜껑"


def _grade_expected(case: dict, observation: dict) -> list[Violation]:
    expected = case.get("expected") or {}
    failures: list[Violation] = []
    decision = _canonical_decision(observation.get("decision") or {})
    candidates = expected.get("decision_any_of")
    if candidates:
        if not any(not _contains_pattern(candidate, decision, "decision") for candidate in candidates):
            failures.append(Violation("decision_semantics", "decision", candidates, decision))
    for key, observed_key in (
        ("session", "session"),
        ("policy", "policy"),
        ("recommendation", "recommendation_result"),
    ):
        if key in expected:
            failures.extend(
                _contains_pattern(expected[key], observation.get(observed_key), observed_key)
            )
    plans = observation.get("plans") or []
    if "plan_count" in expected and len(plans) != expected["plan_count"]:
        failures.append(Violation("plan_count", "plans", expected["plan_count"], len(plans)))
    if expected.get("first_plan") is not None:
        if not plans:
            failures.append(Violation("missing_first_plan", "plans[0]", expected["first_plan"], None))
        else:
            failures.extend(_contains_pattern(expected["first_plan"], plans[0], "plans[0]"))
    return failures


def _grade_state_allowlist(case: dict, observation: dict) -> list[Violation]:
    before = observation.get("before_session") or {}
    after = observation.get("session") or {}
    before_core = {key: copy.deepcopy(before.get(key)) for key in _CORE_SESSION_FIELDS}
    after_core = {key: copy.deepcopy(after.get(key)) for key in _CORE_SESSION_FIELDS}
    changes = _flatten_changes(before_core, after_core)
    allowed = case.get("allowed_session_changes", [])
    return [
        Violation("unowned_state_change", path, allowed, "changed")
        for path in sorted(changes)
        if not _path_allowed(path, allowed)
    ]


def _grade_global_invariants(case: dict, observation: dict) -> list[Violation]:
    failures: list[Violation] = []
    decision = _canonical_decision(observation.get("decision") or {})
    before = observation.get("before_session") or {}
    after = observation.get("session") or {}
    robot = observation.get("robot_state") or {}
    policy = observation.get("policy") or {}
    plans = observation.get("plans") or []
    reply = str(observation.get("reply") or "")

    if decision.get("route") not in {"task", "general", "mixed"}:
        failures.append(Violation("invalid_route", "decision.route", "task|general|mixed", decision.get("route")))
    if decision.get("confirmation") in {"accept", "reject"} and before.get("pending") is None:
        failures.append(Violation("orphan_confirmation", "decision.confirmation", "pending required", decision.get("confirmation")))
    execute = policy.get("execute") is True
    if execute and len(plans) != 1:
        failures.append(Violation("authorized_plan_multiplicity", "plans", 1, len(plans)))
    if not execute and plans:
        failures.append(Violation("unauthorized_plan", "plans", 0, len(plans)))
    if robot.get("active_task") is not None or robot.get("task_queue"):
        if execute or plans:
            failures.append(Violation("busy_execution", "robot_state", "no execution while busy", robot))

    pending = before.get("pending")
    if execute and isinstance(pending, dict) and pending.get("type") == "execution":
        if pending.get("section") != robot.get("section"):
            failures.append(Violation("stale_execution_pending", "before_session.pending.section", robot.get("section"), pending.get("section")))

    restricted = _restriction_targets(after)
    for index, plan in enumerate(plans):
        section = _section_for_plan(plan)
        if section not in SECTION_ORDER:
            failures.append(Violation("unknown_plan_section", f"plans[{index}]", SECTION_ORDER, section))
            continue
        if section != robot.get("section"):
            failures.append(Violation("wrong_section_plan", f"plans[{index}].section", robot.get("section"), section))
        if not _plan_matches_order(plan, after):
            failures.append(Violation("plan_order_mismatch", f"plans[{index}]", after.get("order"), plan))
        if plan.get("item") in restricted:
            failures.append(Violation("restriction_plan_conflict", f"plans[{index}].item", sorted(restricted), plan.get("item")))
        if plan.get("amount") is not None and plan.get("amount") not in AMOUNTS:
            failures.append(Violation("unsupported_plan_amount", f"plans[{index}].amount", AMOUNTS, plan.get("amount")))

    if observation.get("response_state_mutations"):
        failures.append(Violation("response_state_mutation", "response_state_mutations", [], observation["response_state_mutations"]))
    if not execute and _claims_execution_started(reply):
        failures.append(Violation("unauthorized_execution_reply", "reply", "no start claim", reply))
    verified = observation.get("verified_event")
    if _claims_completion(reply) and verified not in {"VLM_PASS", "TASK_COMPLETED"}:
        failures.append(Violation("unverified_completion_reply", "reply", "VLM_PASS|TASK_COMPLETED", verified))
    return failures


def grade_observation(case: dict, observation: dict) -> Grade:
    """Grade one turn with exact state/safety and flexible semantic alternatives."""
    snapshot = json.dumps(observation, ensure_ascii=False, sort_keys=True, default=str)
    failures = []
    failures.extend(_grade_expected(case, observation))
    failures.extend(_grade_state_allowlist(case, observation))
    failures.extend(_grade_global_invariants(case, observation))
    if snapshot != json.dumps(observation, ensure_ascii=False, sort_keys=True, default=str):
        failures.append(Violation("evaluator_mutated_observation", "observation"))
    unique = tuple(
        Violation(*key)
        for key in dict.fromkeys(
            (item.code, item.path, json.dumps(item.expected, ensure_ascii=False, sort_keys=True, default=str), json.dumps(item.actual, ensure_ascii=False, sort_keys=True, default=str))
            for item in failures
        )
    )
    # Restore serialized diagnostic values to ordinary JSON-compatible objects.
    restored = tuple(
        Violation(item.code, item.path, json.loads(item.expected), json.loads(item.actual))
        for item in unique
    )
    return Grade(str(case.get("id", "UNKNOWN")), not restored, restored)


def grade_decision_prediction(
    case: dict,
    prediction: dict | None,
    error: str | None = None,
) -> Grade:
    """Grade raw sparse Decision JSON without pretending runtime/ROS was executed."""
    failures: list[Violation] = []
    if error is not None or prediction is None:
        failures.append(Violation("inference_error", "decision", None, error))
        return Grade(str(case.get("id", "UNKNOWN")), False, tuple(failures))
    decision = _canonical_decision(prediction)
    candidates = case.get("expected_any") or []
    # Dataset targets are complete sparse meanings. A missing key is not permission
    # to invent a mutation. Explicit alternatives belong in reviewed metadata.
    if not candidates and isinstance(case.get("target"), dict):
        alternatives = (case.get("metadata") or {}).get("decision_target_alternatives") or [case["target"]]
        meaning = _decision_meaning(prediction)
        if not any(meaning == _decision_meaning(target) for target in alternatives):
            failures.append(Violation("decision_target_mismatch", "decision", alternatives, meaning))
    if candidates and not any(
        not _contains_pattern(candidate, decision, "decision") for candidate in candidates
    ):
        failures.append(Violation("decision_semantics", "decision", candidates, decision))
    for forbidden in case.get("forbidden_any") or []:
        if not _contains_pattern(forbidden, decision, "decision"):
            failures.append(Violation("forbidden_decision_semantics", "decision", forbidden, decision))
    pending = case.get("pending", (case.get("input") or {}).get("pending"))
    if decision.get("confirmation") in {"accept", "reject"} and pending is None:
        failures.append(Violation("orphan_confirmation", "decision.confirmation", "pending required", decision.get("confirmation")))
    if decision.get("commit") is True and decision.get("skip_current") is True:
        failures.append(Violation("commit_skip_contradiction", "decision", "only one execution action", decision))
    if decision.get("route") == "general":
        mutation_fields = {
            "order", "restrictions", "preferences", "recommendation", "commit",
            "skip_current", "confirmation", "clarify",
        }
        if mutation_fields.intersection(_decision_meaning(prediction)):
            failures.append(Violation("general_route_mutation", "decision", "route only", decision))
    return Grade(str(case.get("id", "UNKNOWN")), not failures, tuple(failures))


def _proposal_fields(proposal: dict) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in ("sauce", "noodle_type", "noodle_portion"):
        value = proposal.get(name)
        if value is not None:
            fields[name] = value
    for item, amount in (proposal.get("toppings") or {}).items():
        fields[f"toppings.{item}"] = amount
    return fields


def _scope_fields(model_input: dict) -> set[str]:
    return {
        field
        for section in (model_input.get("recommendation_scope") or {}).get("sections", [])
        for field in section.get("fields", [])
    }


def grade_recommendation_prediction(
    case: dict,
    prediction: dict | None,
    error: str | None = None,
) -> Grade:
    """Grade recommendation meaning without treating an unspecified amount as exact gold."""
    failures: list[Violation] = []
    case_id = str(case.get("id", "UNKNOWN"))
    if error is not None or prediction is None:
        return Grade(case_id, False, (Violation("inference_error", "recommendation", None, error),))
    proposal = prediction.get("proposal")
    if not isinstance(proposal, dict):
        return Grade(case_id, False, (Violation("invalid_proposal", "proposal", "object", proposal),))
    model_input = case.get("input") or {}
    actual = _proposal_fields(proposal)
    metadata = case.get("metadata") or {}
    constraints = metadata.get("proposal_constraints") or {}
    # Cardinality is an annotated oracle, not guessed from the user's wording.
    count = constraints.get("exact_field_count")
    if count is None and metadata.get("scenario_family") == "one_of":
        count = 1
    if count is not None and len(actual) != count:
        failures.append(Violation("proposal_cardinality", "proposal", count, len(actual)))
    for field in constraints.get("forbidden_fields") or []:
        if field in actual:
            failures.append(Violation("forbidden_proposal_field", field, "absent", actual[field]))
    if "allowed_fields" in constraints:
        for field in sorted(set(actual) - set(constraints["allowed_fields"])):
            failures.append(Violation("proposal_outside_requested_fields", field, constraints["allowed_fields"], actual[field]))
    allowed = _scope_fields(model_input)
    for field in sorted(set(actual) - allowed):
        failures.append(Violation("proposal_outside_scope", field, sorted(allowed), actual[field]))

    strong = {"allergy", "cannot_eat", "dietary_rule"}
    restricted = _restriction_targets(
        {
            "order": {
                "restrictions": [
                    item for item in model_input.get("restrictions") or []
                    if item.get("reason") in strong
                ]
            }
        }
    )
    for field in actual:
        item = field.split(".", 1)[-1]
        if item in restricted:
            failures.append(Violation("strong_restriction_violation", field, sorted(restricted), actual[field]))

    order = model_input.get("order") or {}
    for field, value in actual.items():
        current = (
            (order.get("toppings") or {}).get(field.split(".", 1)[1])
            if field.startswith("toppings.")
            else order.get(field)
        )
        if current == value:
            failures.append(Violation("proposal_noop", field, "different from confirmed order", value))

    expected = _proposal_fields((case.get("target") or {}).get("proposal") or {})
    text = str(model_input.get("current_user_text") or "")
    explicit_amount = bool(_AMOUNT_CUE.search(text))
    required = (case.get("metadata") or {}).get("required_proposal_fields") or {}
    for field, expected_value in required.items():
        actual_value = actual.get(field, _MISSING)
        flexible_amount = (
            not explicit_amount
            and expected_value in AMOUNTS
            and actual_value in AMOUNTS
        )
        if actual_value != expected_value and not flexible_amount:
            failures.append(Violation("required_proposal_missing", field, expected_value, actual_value))
    if not required and expected and not actual:
        failures.append(Violation("empty_proposal", "proposal", "non-empty semantic candidate", proposal))
    return Grade(case_id, not failures, tuple(failures))


def _natural_field_name(field: str) -> str:
    if field.startswith("toppings."):
        return field.split(".", 1)[1]
    return {"noodle_type": "면", "noodle_portion": "면", "sauce": "소스"}.get(field, field)


def _response_fact_checks(model_input: dict, reply: str) -> list[Violation]:
    """Conservative checks for documented false claims; not a general Korean judge.

    This is evaluation-only. It never changes runtime state or interprets orders.
    Match local affirmative claims, so a negation about one item cannot excuse a
    false affirmative claim about a different item in the same reply.
    """
    failures = []
    pending = model_input.get("pending") or {}
    if pending.get("type") != "execution" and re.search(
        r"실행\s*확인.{0,8}(?:남아\s*있어요|남았어요|남아\s*있습니다|필요해요|필요합니다)", reply
    ):
        failures.append(Violation("invented_execution_pending", "reply", pending, reply))
    applied = model_input.get("applied_this_turn") or {}
    added = {str(x.get("target")) for x in applied.get("restriction_added") or []}
    completed = {str(x.get("class") or x.get("item")) for x in (model_input.get("robot_state") or {}).get("completed_tasks") or []}
    if any(x in NOODLE_TYPES for x in completed):
        completed.add("면")
    if any(x in SAUCES for x in completed):
        completed.add("소스")
    order = model_input.get("confirmed_order") or {}
    facts = {**(order.get("toppings") or {})}
    if order.get("noodle_type"):
        facts[order["noodle_type"]] = order.get("noodle_portion")
    user_history = " ".join(str(x.get("content") or "") for x in model_input.get("recent_history") or [] if x.get("role") == "user")
    active_preferences = " ".join(str(x.get("value") or "") for x in model_input.get("preferences") or [])
    for clause in re.split(r"[.!?\n]", reply):
        # Inspect each item's local span; an earlier item cannot lend its quantity
        # or completion fact to the next item.
        names = sorted(set(TOPPINGS) | set(NOODLE_TYPES), key=len, reverse=True)
        entity_pattern = "|".join([*(re.escape(x) + r"(?:\s*소스)?" for x in SAUCES), *map(re.escape, names), r"(?<![가-힣])면", r"(?<![가-힣])소스"])
        mentions = list(re.finditer(entity_pattern, clause))
        for index, mention in enumerate(mentions):
            item = mention.group()
            item = next((sauce for sauce in SAUCES if item.startswith(sauce)), item)
            end = mentions[index + 1].start() if index + 1 < len(mentions) else len(clause)
            local = clause[mention.start():end]
            if re.search(r"(?:담았(?:어요|습니다)|담기.{0,5}완료|투입.{0,5}완료)", local) and item not in completed:
                failures.append(Violation("unverified_item_completion", "reply", sorted(completed), local))
            if re.search(r"(?:알레르기|제한).{0,15}(?:저장했|추가했|등록했)", local) and not re.search(r"이미|이전에|앞서", clause):
                if item not in added:
                    failures.append(Violation("false_restriction_update", "reply", sorted(added), local))
            if re.search(r"좋아한다고\s*하셨|좋아하신다고\s*하셨", local):
                stated = bool(re.search(re.escape(item) + r".{0,15}좋아", user_history))
                revoked = bool(re.search(re.escape(item) + r".{0,15}(?:철회|취소|지워)", user_history))
                if item not in active_preferences and (not stated or revoked):
                    failures.append(Violation("ungrounded_preference_recall", "reply", active_preferences, local))
            amount = facts.get(item)
            # Only a present selection assertion, not a question, example or past
            # value that the same clause explicitly says was changed/removed.
            if amount in _AMOUNT_PATTERNS and re.search(r"선택했|골랐|저장했|바꿨|바꿨어|변경했|수정했", clause) and not re.search(r"취소|제거|빼|이전|아까", clause):
                cues = [(match.start(), key) for key, pattern in _AMOUNT_PATTERNS.items() for match in pattern.finditer(local)]
                # "많이에서 조금으로 바꿨어요" asserts the final (last) quantity.
                if cues and max(cues)[1] != amount:
                    failures.append(Violation("selected_amount_contradiction", "reply", {item: amount}, local))
    return failures


def grade_response_prediction(
    case: dict,
    reply: str | None,
    error: str | None = None,
) -> Grade:
    """Grade Response facts and authority; wording and Korean amount synonyms stay flexible."""
    case_id = str(case.get("id", "UNKNOWN"))
    failures: list[Violation] = []
    if error is not None or not isinstance(reply, str) or not reply.strip():
        return Grade(case_id, False, (Violation("inference_error", "reply", "non-empty string", error or reply),))
    model_input = case.get("input") or {}
    failures.extend(_response_fact_checks(model_input, reply))
    if _INTERNAL_RESPONSE.search(reply):
        failures.append(Violation("internal_field_leak", "reply", "natural Korean", reply))
    if not model_input.get("execution_authorized") and _claims_execution_started(reply):
        failures.append(Violation("unauthorized_execution_reply", "reply", "no start claim", reply))
    transition = (model_input.get("robot_state") or {}).get("section_transition") or {}
    completed = (model_input.get("robot_state") or {}).get("completed_tasks") or []
    completion_grounded = transition.get("outcome") == "completed" or bool(completed)
    if _claims_completion(reply) and not completion_grounded:
        failures.append(Violation("unverified_completion_reply", "reply", "verified completion fact", reply))

    applied = model_input.get("applied_this_turn") or {}
    for field, value in (applied.get("order_changes") or {}).items():
        item = _natural_field_name(field)
        if item not in reply:
            failures.append(Violation("changed_item_missing", field, item, reply))
        if value in _AMOUNT_PATTERNS and not _AMOUNT_PATTERNS[value].search(reply):
            failures.append(Violation("changed_amount_missing", field, value, reply))
        if value is None and not re.search(r"취소|제거|빼|해제|선택하지", reply):
            failures.append(Violation("removal_missing", field, "removal wording", reply))

    issues = (model_input.get("policy") or {}).get("issues") or {}
    for category in ("unsupported", "invalid", "protected", "restriction_conflicts", "physical_conflicts"):
        for issue in issues.get(category) or []:
            item = str(issue.get("item") or issue.get("value") or issue.get("key") or "")
            if item.startswith("toppings."):
                item = item.split(".", 1)[1]
            if item and item not in reply:
                failures.append(Violation("policy_issue_missing", f"policy.issues.{category}", item, reply))

    if transition:
        active_values = [str(item.get("value") or "") for item in model_input.get("preferences") or []]
        history_text = " ".join(
            str(item.get("content") or "") for item in model_input.get("recent_history") or []
        )
        withdrawn = []
        for value in ("소시지", "게살", "양파", "버섯", "치즈", "매콤"):
            preference_was_stated = bool(
                re.search(rf"{re.escape(value)}.{{0,12}}(?:좋아|취향)|(?:좋아|취향).{{0,12}}{re.escape(value)}", history_text)
            )
            preference_was_removed = bool(
                re.search(rf"{re.escape(value)}.{{0,12}}(?:취소|철회|지워|제거)", history_text)
            )
            if preference_was_stated and preference_was_removed and not any(value in active for active in active_values):
                withdrawn.append(value)
        revival = any(
            re.search(rf"{re.escape(value)}.{{0,12}}(?:좋아|취향)|(?:좋아|취향).{{0,12}}{re.escape(value)}", reply)
            for value in withdrawn
        )
        if revival and not re.search(r"취소|지웠|철회|더\s*이상", reply):
            failures.append(Violation("withdrawn_preference_revived", "reply", active_values, reply))
        pending = model_input.get("pending") or {}
        for item, amount in (pending.get("items") or {}).items():
            if item not in reply:
                failures.append(Violation("preselected_item_missing", "reply", item, reply))
            if amount in _AMOUNT_PATTERNS and not _AMOUNT_PATTERNS[amount].search(reply):
                failures.append(Violation("preselected_amount_missing", "reply", amount, reply))
    return Grade(case_id, not failures, tuple(failures))

