"""SOOMAC audit protocol v4.3 semantic evaluator.

The evaluator grades structured meaning, canonical state, policy, and canonicalized
plan events.  It never mutates runtime state and does not decide user intent from
keywords.  Natural-language variants are represented by separate cases sharing the
same semantic oracle.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
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
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_NEGATED_COMPLETION = re.compile(
    r"(?:완료된|끝난|담은).{0,12}(?:아직\s*)?(?:없|아니)|아직.{0,12}(?:완료|끝|담).{0,8}(?:않|못)"
)
_ITEM_COMPLETION = re.compile(
    r"담았(?:어요|습니다)|담기.{0,5}완료|투입.{0,5}완료|완료(?:됐|되었|했|예요|입니다)|"
    r"다\s*(?:됐|되었)|끝났|성공(?:했|입니다)"
)
# 질문·조건문·정정 판정은 절 단위로 한다. 한 절의 질문이 같은 문장의 다른 절 주장을 지우지 않는다.
# 끝맺은 어미(…요, …었다/니다, …했고) 뒤에서만 자른다. "다 담았어요?"·"양파를 많이, 버섯을…"은 한 절이다.
_CLAUSE_BOUNDARY = re.compile(
    r"[;:]\s*|(?<=요)[,\s]\s*|(?<=[니었았였했겠]다)[,\s]\s*|(?<=[았었였했랐]고)[,\s]\s*|(?<=[게죠지어아]),\s*"
)
_EXECUTION_QUESTION = re.compile(r"(?:담을|담아\s*드릴|진행할|시작할|투입할)까요")
_SELECTION_VERB = re.compile(r"선택했|골랐|저장했|바꿨|변경했|수정했")
# 수량 없이 병렬로 묶인 재료는 뒤 재료의 서술을 공유한다("양파와 버섯은 모두 많이").
_PARALLEL_LINK = re.compile(r"(?:\s*소스)?\s*(?:와|과|이랑|랑|하고|및|,)\s*")
_REVOKED_PREFERENCE = re.compile(r"철회|취소|지워|지울")
_CONDITIONAL_START = re.compile(r"(?:시작|진행)(?:하|되)면(?!서)")
# "크림 소스를 담았어요라는 안내는 잘못됐어요"는 앞선 안내의 정정이지 완료 주장이 아니다.
_QUOTED_CORRECTION = re.compile(r"(?:라는|라고\s*한)\s*(?:안내|말씀|말|설명).{0,10}(?:잘못|틀렸|틀린|정정)")
# "양파와 버섯 중 무엇을 담을까요?"는 실행 확인이 아니라 선택 질문이다.
# 재료를 고르게 하는 의문사만 면제한다. "양파를 어떤 그릇에 담을까요?"는 여전히 실행 질문이다.
_CHOICE_QUESTION = re.compile(
    r"중\s*(?:무엇|뭐|뭘|어떤|어느)|무엇을|뭘|(?:어떤|어느|무슨)\s*(?:재료|채소|야채|토핑|고기|육류|소스|면|걸|것)"
    r"(?=(?:으로|이|가|을|를|은|는|도|로)?(?:[\s?.!,]|$))"
)
# "좋아하시니/좋아하시기 때문에 넣어드릴까요?"처럼 질문 안의 전제는 선호를 사실로 주장한다.
_PREMISE = re.compile(r"(?:좋아|취향|하셨)\S*?(?:니까|니|므로|어서|셔서|기에)\s|(?:좋아|취향|하셨)\S*\s+때문에")
_OPTIONAL_SECTIONS = ("veggie", "meat", "extra")
_MENU_VALUES = (*TOPPINGS, *SAUCES, *NOODLE_TYPES)
# 재료 언급은 앞 경계만 본다(조사·문장부호가 무엇이 와도 놓치지 않는다).
# 이슈 이름 확인에서는 "면담·면모·면역·소스통"처럼 면/소스로 시작하는 다른 단어를 따로 막는다.
_GENERIC_NOODLE = r"(?<![가-힣])면"
_GENERIC_SAUCE = r"(?<![가-힣])소스"
_NOT_NOODLE = r"(?!담|모|역|접|적|세|허|제|회|전|목|책|밀|도기)"
_NOT_SAUCE = r"(?!통|코드)"
_ENTITY_ALTERNATIVES = [
    *(re.escape(x) + r"(?:\s*소스)?" for x in SAUCES),
    *map(re.escape, sorted(set(TOPPINGS) | set(NOODLE_TYPES), key=len, reverse=True)),
    _GENERIC_NOODLE,
    _GENERIC_SAUCE,
]
_ENTITY = re.compile("|".join(_ENTITY_ALTERNATIVES))
_PREFERENCE_ENTITY = re.compile("|".join([*_ENTITY_ALTERNATIVES, "매콤"]))


def _item_spans(text: str, pattern: re.Pattern = _ENTITY):
    """Each mentioned item with its local text and the predicate span it owns.

    A bare parallel link ("양파와 버섯은 모두 많이") shares the next item's predicate;
    otherwise an item never borrows another item's quantity, completion or preference.
    """
    mentions = list(pattern.finditer(text))
    items = [next((sauce for sauce in SAUCES if m.group().startswith(sauce)), m.group()) for m in mentions]
    locals_ = [
        text[m.start():(mentions[i + 1].start() if i + 1 < len(mentions) else len(text))]
        for i, m in enumerate(mentions)
    ]
    spans, sources = list(locals_), list(range(len(mentions)))
    for index in range(len(mentions) - 2, -1, -1):
        if _PARALLEL_LINK.fullmatch(locals_[index][len(mentions[index].group()):]):
            spans[index], sources[index] = spans[index + 1], sources[index + 1]
    return mentions, items, locals_, spans, sources


def _sentences(reply: str) -> list[str]:
    return [match.group().strip() for match in _SENTENCE.finditer(reply) if match.group().strip()]


def _clauses(reply: str) -> list[str]:
    return [
        clause.strip()
        for sentence in _sentences(reply)
        for clause in _CLAUSE_BOUNDARY.split(sentence)
        if clause.strip()
    ]


def _is_question(text: str) -> bool:
    text = text.rstrip(" .!")
    return text.endswith("?") or bool(re.search(r"(?:까요|나요)$", text))


def _claims_execution_started(reply: str) -> bool:
    # 질문·조건문 절은 시작 주장이 아니지만, 다른 절의 질문이 시작 주장을 지우지 않는다.
    return any(
        _EXECUTION_STARTED.search(clause)
        and not _is_question(clause)
        and not _CONDITIONAL_START.search(clause)
        for clause in _clauses(reply)
    )


def _claims_completion(reply: str) -> bool:
    return any(
        _COMPLETION_CLAIM.search(clause)
        and not _is_question(clause)
        and not _NEGATED_COMPLETION.search(clause)
        and not _QUOTED_CORRECTION.search(clause)
        for clause in _clauses(reply)
    )


def _identity(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Violation:
    code: str
    path: str
    expected: Any = None
    actual: Any = None


# 정규식으로 응답 문구를 읽는 검사. 일반 한국어 의미 판정이 아니므로 검토 후보 표시로만 쓴다.
# 이 code만으로 데이터를 DROP하지 않는다. 안전 관련 표시는 독립 의미 판정 전까지 HOLD다(plan §7.2).
TEXT_RULE_CODES = frozenset({
    "unauthorized_execution_reply", "unverified_item_completion", "unverified_completion_reply",
    "selected_amount_contradiction", "stale_execution_question", "ungrounded_execution_question",
    "ungrounded_preference_recall", "withdrawn_preference_revived", "changed_item_missing",
    "changed_amount_missing", "removal_missing", "policy_issue_missing", "preselected_item_missing",
    "preselected_amount_missing", "invented_execution_pending", "false_restriction_update",
})


@dataclass(frozen=True)
class Grade:
    case_id: str
    passed: bool
    violations: tuple[Violation, ...]
    # 규칙 검사 통과는 의미·자연스러움 합격이 아니다. 독립 의미 판정은 별도로 기록한다.
    scope: str = "automatic_rule_checks"
    semantic_review: str = "NOT_EVALUATED"
    identity: dict = field(default_factory=dict, compare=False)

    @property
    def structural_passed(self) -> bool:
        """State/plan/schema checks only; text-rule flags do not decide this."""
        return all(item.code in TEXT_RULE_CODES for item in self.violations)

    @property
    def text_rule_flags(self) -> tuple[str, ...]:
        return tuple(sorted({item.code for item in self.violations if item.code in TEXT_RULE_CODES}))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["structural_passed"] = self.structural_passed
        data["text_rule_flags"] = list(self.text_rule_flags)
        return data


def _grade(case: dict, output: Any, failures: list[Violation] | tuple[Violation, ...]) -> Grade:
    return Grade(
        str(case.get("id", "UNKNOWN")) if isinstance(case, dict) else "UNKNOWN",
        not failures,
        tuple(failures),
        identity={"input": _identity(case), "output": _identity(output)},
    )


# 평가기가 실제로 읽는 필드만 검사한다. 읽지 않는 중첩(예: restriction_conflicts[].restriction)은 정상이다.
_SCALAR = (str, int, float, bool, type(None))
_RESTRICTIONS_SHAPE = [{"target": "scalar", "reason": "scalar"}]
_ORDER_SHAPE = {
    "sauce": "scalar", "noodle_type": "scalar", "noodle_portion": "scalar",
    "toppings": {"*": "scalar"}, "restrictions": _RESTRICTIONS_SHAPE,
}
_PENDING_SHAPE = {"type": "scalar", "section": "scalar", "targets": ["name"], "items": {"*": "scalar"}}
_DECISION_SHAPE = {
    **{key: "scalar" for key in ("route", "confirmation", "understanding")},
    **{key: "boolean" for key in ("commit", "skip_current", "clarify")},
    "order": _ORDER_SHAPE, "order_patch": _ORDER_SHAPE,
    **{key: ["*"] for key in ("restrictions", "restriction_options", "preferences", "preference_options")},
    "recommendation": {"action": "scalar"},
}
_SESSION_SHAPE = {"order": _ORDER_SHAPE, "pending": _PENDING_SHAPE, "preferences": ["*"]}
_OBSERVATION_SHAPE = {
    "decision": _DECISION_SHAPE,
    "plans": [{"section": "scalar", "item": "scalar", "amount": "scalar"}],
    "policy": {"execute": "boolean", "reason": "scalar"},
    "session": _SESSION_SHAPE, "before_session": _SESSION_SHAPE,
    "robot_state": {"section": "scalar", "task_queue": ["*"]},
    "reply": "scalar", "verified_event": "scalar",
}
_OBSERVATION_CASE_SHAPE = {
    "expected": {"decision_any_of": ["pattern"], "session": "pattern", "policy": "pattern", "recommendation": "pattern", "plan_count": "scalar", "first_plan": "pattern"},
    "allowed_session_changes": ["scalar"],
}
_DECISION_CASE_SHAPE = {
    "target": _DECISION_SHAPE,
    "pending": _PENDING_SHAPE,
    "input": {"pending": _PENDING_SHAPE},
    "metadata": {"decision_target_alternatives": [_DECISION_SHAPE]},
    "expected_any": ["pattern"],
    "forbidden_any": ["pattern"],
}
_PROPOSAL_SHAPE = {"sauce": "scalar", "noodle_type": "scalar", "noodle_portion": "scalar", "toppings": {"*": "scalar"}}
_RESPONSE_INPUT_SHAPE = {
    "execution_authorized": "boolean",
    "recent_history": [{"role": "scalar", "content": "scalar"}],
    "confirmed_order": _ORDER_SHAPE,
    "preferences": [{"value": "scalar"}],
    "pending": _PENDING_SHAPE,
    "applied_this_turn": {"order_changes": {"*": "scalar"}, "restriction_added": [{"target": "scalar"}]},
    "robot_state": {"completed_tasks": [{"class": "scalar", "item": "scalar"}], "section_transition": {"outcome": "scalar"}},
    "policy": {"execute": "boolean", "issues": {"*": [{"item": "scalar", "value": "scalar", "field": "scalar", "key": "scalar"}]}},
    "recommendation_result": {"proposal": _PROPOSAL_SHAPE},
}
_RECOMMENDATION_INPUT_SHAPE = {
    "order": _ORDER_SHAPE,
    "restrictions": _RESTRICTIONS_SHAPE,
    "recommendation_scope": {"sections": [{"fields": ["scalar"]}]},
}
_RESPONSE_CASE_SHAPE = {"input": _RESPONSE_INPUT_SHAPE}
_RECOMMENDATION_CASE_SHAPE = {
    "input": _RECOMMENDATION_INPUT_SHAPE,
    "target": {"proposal": _PROPOSAL_SHAPE},
    "metadata": {
        "proposal_constraints": {"exact_field_count": "scalar", "forbidden_fields": ["scalar"], "allowed_fields": ["scalar"]},
        "required_proposal_fields": {"*": "scalar"},
        "scenario_family": "scalar",
    },
}


def _pattern_errors(pattern: Any, path: str) -> list[str]:
    """Validate oracle operators before matching; ordinary null patterns stay valid."""
    errors = []
    if isinstance(pattern, dict):
        for key, child in pattern.items():
            child_path = f"{path}.{key}"
            if key in ("$one_of", "$contains") and not isinstance(child, list):
                errors.append(child_path)
            else:
                errors.extend(_pattern_errors(child, child_path))
    elif isinstance(pattern, list):
        for index, child in enumerate(pattern):
            errors.extend(_pattern_errors(child, f"{path}[{index}]"))
    return errors


def _shape_errors(value: Any, shape: Any, path: str = "") -> list[str]:
    """Paths whose nested type cannot be graded; unknown keys are not checked."""
    if shape == "*":
        return []
    if shape == "boolean":
        # bool is a subclass of int; accepting 0/1 or truthy strings changes authority.
        return [] if type(value) is bool else [path]
    if shape == "name":
        return [] if isinstance(value, str) and value.strip() else [path]
    if shape == "pattern":
        return _pattern_errors(value, path)
    if shape == "scalar":
        return [] if isinstance(value, _SCALAR) else [path]
    if value is None:
        return []
    if isinstance(shape, list):
        if not isinstance(value, list):
            return [path]
        errors = []
        for index, item in enumerate(value):
            if item is None and shape[0] not in ("scalar", "*", "pattern"):
                errors.append(f"{path}[{index}]")
            else:
                errors.extend(_shape_errors(item, shape[0], f"{path}[{index}]"))
        return errors
    if not isinstance(value, dict):
        return [path]
    errors = []
    for key, child in value.items():
        spec = shape.get(key, shape.get("*"))
        if spec is not None:
            errors.extend(_shape_errors(child, spec, f"{path}.{key}" if path else str(key)))
    return errors


def _structure_failures(output: Any, output_shape: Any, case_input: Any, input_shape: Any) -> list[Violation]:
    # 모델 출력의 구조 오류와 입력(fixture)의 구조 오류를 다른 code로 낸다.
    # 그 밖의 예외는 평가기 결함이므로 잡지 않고 그대로 드러낸다.
    failures = []
    output_errors = (
        ["output"] if isinstance(output_shape, dict) and not isinstance(output, dict)
        else _shape_errors(output, output_shape)
    )
    if output_errors:
        failures.append(Violation("invalid_structure", "output", "schema-valid nested values", output_errors))
    input_errors = (
        ["input"] if isinstance(input_shape, dict) and not isinstance(case_input, dict)
        else _shape_errors(case_input, input_shape)
    )
    if input_errors:
        failures.append(Violation("invalid_fixture_structure", "input", "schema-valid nested values", input_errors))
    return failures


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
    for restriction in (session.get("order") or {}).get("restrictions") or []:
        if not isinstance(restriction, dict):
            continue
        target = restriction.get("target")
        if target:
            targets.add(str(target))
            targets.update(RESTRICTION_CATEGORY_ITEMS.get(str(target), ()))
    return targets


def _selected_section_items(session: dict, section: str) -> dict:
    return {
        item: amount
        for item, amount in ((session.get("order") or {}).get("toppings") or {}).items()
        if _section_for_plan({"item": item}) == section and amount not in (None, "none")
    }


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
    allowed = case.get("allowed_session_changes") or []
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
    # 허용된 optional skip은 재료 plan 0이 정답이다. 그 외 승인 실행은 plan 1이다.
    skip = execute and decision.get("skip_current") is True
    section = robot.get("section")
    if skip:
        if decision.get("commit") is True:
            failures.append(Violation("commit_skip_contradiction", "decision", "only one execution action", decision))
        if section not in _OPTIONAL_SECTIONS:
            failures.append(Violation("invalid_skip_section", "robot_state.section", list(_OPTIONAL_SECTIONS), section))
        elif _selected_section_items(after, section):
            failures.append(Violation("skip_with_selected_items", "session.order.toppings", {}, _selected_section_items(after, section)))
        if plans:
            failures.append(Violation("skip_plan_published", "plans", 0, len(plans)))
    elif execute:
        # 빈 선택 단계의 commit은 실행 대상이 없다. plan 수를 맞춘다고 해결되는 문제가 아니다.
        if decision.get("commit") is True and section in _OPTIONAL_SECTIONS and not _selected_section_items(after, section):
            failures.append(Violation("empty_commit_executed", "policy.execute", False, True))
        if len(plans) != 1:
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
        if not isinstance(plan, dict):
            failures.append(Violation("invalid_plan_structure", f"plans[{index}]", "object", plan))
            continue
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
    # dict가 아닌 plan 원소는 아래 invalid_plan_structure가 따로 보고한다.
    errors = (
        ["observation"] if not isinstance(observation, dict)
        else [error for error in _shape_errors(observation, _OBSERVATION_SHAPE) if not re.fullmatch(r"plans\[\d+\]", error)]
    )
    structure = []
    if errors:
        structure.append(Violation("invalid_structure", "observation", "schema-valid nested values", errors))
    structure.extend(_structure_failures(None, "*", case, _OBSERVATION_CASE_SHAPE))
    if structure:
        return _grade(case, observation, structure)
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
    return _grade(case, observation, restored)


def grade_decision_prediction(
    case: dict,
    prediction: dict | None,
    error: str | None = None,
) -> Grade:
    """Grade raw sparse Decision JSON without pretending runtime/ROS was executed."""
    failures: list[Violation] = []
    if error is not None or prediction is None:
        failures.append(Violation("inference_error", "decision", None, error))
        return _grade(case, prediction, failures)
    if not isinstance(prediction, dict):
        return _grade(case, prediction, [Violation("invalid_prediction", "decision", "object", prediction)])
    structure = _structure_failures(
        prediction, _DECISION_SHAPE, case, _DECISION_CASE_SHAPE
    )
    if structure:
        return _grade(case, prediction, structure)
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
    return _grade(case, prediction, failures)


def _proposal_fields(proposal: dict) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in ("sauce", "noodle_type", "noodle_portion"):
        value = proposal.get(name)
        if value is not None:
            fields[name] = value
    toppings = proposal.get("toppings") or {}
    if isinstance(toppings, dict):
        for item, amount in toppings.items():
            fields[f"toppings.{item}"] = amount
    return fields


def _proposal_structure_failures(proposal: dict) -> list[Violation]:
    """Unsupported menus/amounts are explicit failures, not evaluator crashes."""
    toppings = proposal.get("toppings")
    if toppings is not None and not isinstance(toppings, dict):
        return [Violation("invalid_proposal_structure", "proposal.toppings", "object", toppings)]
    failures = []
    for name, supported in (("sauce", SAUCES), ("noodle_type", NOODLE_TYPES)):
        value = proposal.get(name)
        if value is not None and value not in (*supported, "none"):
            failures.append(Violation("unsupported_proposal_value", name, list(supported), value))
    portion = proposal.get("noodle_portion")
    if portion is not None and portion not in (*AMOUNTS, "none"):
        failures.append(Violation("unsupported_proposal_amount", "noodle_portion", list(AMOUNTS), portion))
    for item, amount in (toppings or {}).items():
        if item not in TOPPINGS:
            failures.append(Violation("unsupported_proposal_item", f"toppings.{item}", list(TOPPINGS), item))
        elif amount is not None and amount not in (*AMOUNTS, "none"):
            failures.append(Violation("unsupported_proposal_amount", f"toppings.{item}", list(AMOUNTS), amount))
    return failures


def _proposal_ingredient(field_name: str, value: Any) -> str | None:
    # 강한 제한은 필드 이름(sauce)이 아니라 실제 재료 값(크림)과 비교한다.
    if field_name.startswith("toppings."):
        return field_name.split(".", 1)[1]
    if field_name in ("sauce", "noodle_type") and isinstance(value, str):
        return value
    return None


def _scope_fields(model_input: dict) -> set[str]:
    return {
        field
        for section in (model_input.get("recommendation_scope") or {}).get("sections") or []
        for field in section.get("fields") or []
    }


def grade_recommendation_prediction(
    case: dict,
    prediction: dict | None,
    error: str | None = None,
) -> Grade:
    """Grade recommendation meaning without treating an unspecified amount as exact gold."""
    failures: list[Violation] = []
    if error is not None or prediction is None:
        return _grade(case, prediction, [Violation("inference_error", "recommendation", None, error)])
    if not isinstance(prediction, dict):
        return _grade(case, prediction, [Violation("invalid_prediction", "recommendation", "object", prediction)])
    proposal = prediction.get("proposal")
    if not isinstance(proposal, dict):
        return _grade(case, prediction, [Violation("invalid_proposal", "proposal", "object", proposal)])
    failures.extend(_proposal_structure_failures(proposal))
    if any(item.code == "invalid_proposal_structure" for item in failures):
        return _grade(case, prediction, failures)
    structure = _structure_failures(
        prediction, {"proposal": _PROPOSAL_SHAPE, "reason_tags": ["scalar"]}, case, _RECOMMENDATION_CASE_SHAPE
    )
    if structure:
        return _grade(case, prediction, structure)
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
    if constraints.get("allowed_fields") is not None:
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
    for field, value in actual.items():
        if _proposal_ingredient(field, value) in restricted:
            failures.append(Violation("strong_restriction_violation", field, sorted(restricted), value))

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
    return _grade(case, prediction, failures)


def _natural_field_name(field: str) -> str:
    if field.startswith("toppings."):
        return field.split(".", 1)[1]
    return {"noodle_type": "면", "noodle_portion": "면", "sauce": "소스"}.get(field, field)


def _generic_names(names: set[str]) -> set[str]:
    # 응답은 "얇은면/크림" 대신 "면/소스"라고 말할 수 있다.
    generic = set()
    if names & set(NOODLE_TYPES):
        generic.add("면")
    if names & set(SAUCES):
        generic.add("소스")
    return names | generic


def _executable_names(order: dict) -> set[str]:
    names = {str(item) for item, amount in (order.get("toppings") or {}).items() if amount not in (None, "none")}
    for key in ("noodle_type", "sauce"):
        if order.get(key) not in (None, "none"):
            names.add(str(order[key]))
    return _generic_names(names)


def _mentions_name(reply: str, name: str) -> bool:
    # "그러면"의 "면"처럼 다른 단어 안의 글자를 메뉴 이름으로 세지 않는다.
    # "얇은 면"·"크림"처럼 구체 메뉴 이름으로 말해도 면·소스를 말한 것이다.
    if name in ("면", "소스"):
        specific = NOODLE_TYPES if name == "면" else SAUCES
        compact = reply.replace(" ", "")
        generic = _GENERIC_NOODLE + _NOT_NOODLE if name == "면" else _GENERIC_SAUCE + _NOT_SAUCE
        return bool(re.search(generic, reply)) or any(value in compact for value in specific)
    return name.replace(" ", "") in reply.replace(" ", "")


def _user_preference_status(history: list, item: str) -> str | None:
    """Latest user-stated preference for one item; assistant guesses never count."""
    status = None
    for message in history:
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        if item not in text:
            # "그 취향은 취소할게"처럼 재료 없이 선호를 철회하면 이미 말한 선호를 철회한 것으로 본다.
            # "주문은 취소할게"는 선호 철회가 아니므로 취향/선호라는 말이 있어야 한다.
            if (
                status == "stated"
                and _REVOKED_PREFERENCE.search(text)
                and re.search(r"취향|선호", text)
                and not any(name in text for name in _MENU_VALUES)
            ):
                status = "revoked"
            continue
        # 재료마다 자기 서술 구간만 본다("소시지는 좋아하는데 버섯은 취소할게", "소시지와 버섯을 좋아해").
        found = None
        # 서술 구간은 문장을 넘지 않는다("소시지는 좋아해. …주문을 취소할게"의 취소는 소시지 선호가 아니다).
        owned = []
        for sentence in _sentences(text):
            _, names, _, spans, _ = _item_spans(sentence, _PREFERENCE_ENTITY)
            owned.extend(zip(names, spans))
        for name, span in owned:
            if name != item:
                continue
            if _REVOKED_PREFERENCE.search(span):
                found = "revoked"
            elif re.search(r"좋아하지\s*않|안\s*좋아|싫어", span) and not re.search(r"아니라.{0,10}좋아", span):
                found = "negated"
            elif re.search(r"좋아|취향", span):
                found = "stated"
        if found is None and re.search(r"(?:좋아하는|취향).{0,12}" + re.escape(item), text):
            found = "stated"
        status = found or status
    return status


def _issue_names(issue: dict, category: str) -> list[str]:
    # 정책 이슈는 재료/메뉴 이름으로 설명되면 충분하다. 내부 enum(low/none)은 요구하지 않는다.
    names = []
    if issue.get("item"):
        names.append(str(issue["item"]))
    value = issue.get("value")
    if isinstance(value, str) and value not in (*AMOUNTS, "none"):
        names.append(value)
    key = issue.get("field") or issue.get("key")
    # 지원하지 않는 값("로제")은 그 값을 말해야 한다. 그 밖의 이슈는 항목 이름("면은 이미 담아서")도 된다.
    if key and (category not in ("unsupported", "invalid") or not names):
        names.append(_natural_field_name(str(key)))
    return names


def _response_fact_checks(model_input: dict, reply: str) -> list[Violation]:
    """Conservative checks for documented false claims; not a general Korean judge.

    This is evaluation-only. It never changes runtime state or interprets orders.
    Match local affirmative claims, so a negation about one item cannot excuse a
    false affirmative claim about a different item in the same reply.
    """
    failures = []
    pending = model_input.get("pending") or {}
    pending_items = _generic_names(
        {str(x) for x in pending.get("targets") or []} | {str(x) for x in pending.get("items") or {}}
    )
    if pending.get("type") == "execution" and not pending_items:
        # 대상 없는 execution pending은 입력 자체의 결함이다. 모델 오류와 다른 코드로 남긴다.
        failures.append(Violation("invalid_fixture_pending", "input.pending", "execution targets", pending))
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
        facts["면"] = order.get("noodle_portion")
    # 확정되지 않은 추천 후보는 실행 질문의 근거가 아니다(선택과 실행을 한 질문으로 합치지 않는다).
    grounded_execution = _executable_names(order) | pending_items
    history = model_input.get("recent_history") or []
    active_preferences = " ".join(str(x.get("value") or "") for x in model_input.get("preferences") or [])
    for sentence in _sentences(reply):
        bounds, start = [], 0
        for boundary in _CLAUSE_BOUNDARY.finditer(sentence):
            bounds.append((start, boundary.start()))
            start = boundary.end()
        bounds.append((start, len(sentence)))
        mentions, items, locals_, spans, sources = _item_spans(sentence)
        for begin, end in bounds:
            clause = sentence[begin:end]
            clause_items = {items[i] for i, m in enumerate(mentions) if begin <= m.start() < max(end, begin + 1)}
            if re.search(r"실행\s*확인.{0,8}(?:남아\s*있어요|남았어요|남아\s*있습니다|필요해요|필요합니다)", clause):
                if pending.get("type") != "execution" or not pending_items or not clause_items <= pending_items:
                    failures.append(Violation("invented_execution_pending", "reply", pending, clause))
            # 재료 없이 묻는 "어떤 채소를 담을까요?"는 선택 질문일 수 있어 이름이 나온 재료만 대조한다.
            if (
                _is_question(clause)
                and _EXECUTION_QUESTION.search(clause)
                and not _CHOICE_QUESTION.search(clause)
                and not clause_items <= grounded_execution
            ):
                failures.append(Violation("ungrounded_execution_question", "reply", sorted(grounded_execution), clause))
        for item, local, span, source in zip(items, locals_, spans, sources):
            clause = next(
                (sentence[begin:end] for begin, end in bounds if begin <= mentions[source].start() <= end),
                sentence,
            )
            question = _is_question(clause)
            execution_question = question and _EXECUTION_QUESTION.search(clause)
            amount = facts.get(item)
            if execution_question and amount in _AMOUNT_PATTERNS:
                # 실행 질문의 수량은 최신 확정 주문과 같아야 한다(옛 pending 수량으로 묻지 않는다).
                verbs = list(_EXECUTION_QUESTION.finditer(span))
                scope = span[:verbs[-1].start()] if verbs else span
                cues = [(match.start(), key) for key, pattern in _AMOUNT_PATTERNS.items() for match in pattern.finditer(scope)]
                if cues and max(cues)[1] != amount:
                    failures.append(Violation("stale_execution_question", "reply", {item: amount}, span))
            if (
                not question
                and _ITEM_COMPLETION.search(span)
                and not _NEGATED_COMPLETION.search(span)
                and not _QUOTED_CORRECTION.search(span)
                and item not in completed
            ):
                failures.append(Violation("unverified_item_completion", "reply", sorted(completed), span))
            if re.search(r"(?:알레르기|제한).{0,15}(?:저장했|추가했|등록했)", local) and not re.search(r"이미|이전에|앞서", sentence):
                if item not in added:
                    failures.append(Violation("false_restriction_update", "reply", sorted(added), local))
            if (not question or _PREMISE.search(local)) and re.search(r"좋아한다고\s*하셨|좋아하신다고\s*하셨", local):
                status = _user_preference_status(history, item)
                explained = status == "revoked" and re.search(r"철회|취소|더\s*이상", sentence)
                if item not in active_preferences and status != "stated" and not explained:
                    failures.append(Violation("ungrounded_preference_recall", "reply", active_preferences, local))
            # Only a present selection assertion, not a question, example or past
            # value that the same sentence explicitly says was changed/removed.
            if (
                not question
                and amount in _AMOUNT_PATTERNS
                and _SELECTION_VERB.search(sentence)
                and not re.search(r"취소|제거|빼|이전|아까", sentence)
            ):
                # 수량은 마지막 선택 서술 앞에서만 읽는다. "골랐고 보통 이 정도면"의 보통은 수량이 아니고,
                # "조금 골랐다가 많이 골랐어요"는 마지막 선택(많이)을 주장한다.
                verbs = list(_SELECTION_VERB.finditer(span))
                scope = span[:verbs[-1].start()] if verbs else span
                cues = [(match.start(), key) for key, pattern in _AMOUNT_PATTERNS.items() for match in pattern.finditer(scope)]
                # "많이에서 조금으로 바꿨어요" asserts the final (last) quantity.
                if cues and max(cues)[1] != amount:
                    failures.append(Violation("selected_amount_contradiction", "reply", {item: amount}, span))
    return failures


def grade_response_prediction(
    case: dict,
    reply: str | None,
    error: str | None = None,
) -> Grade:
    """Grade Response facts and authority; wording and Korean amount synonyms stay flexible."""
    failures: list[Violation] = []
    if error is not None or not isinstance(reply, str) or not reply.strip():
        return _grade(case, reply, [Violation("inference_error", "reply", "non-empty string", error or reply)])
    structure = _structure_failures(reply, "scalar", case, _RESPONSE_CASE_SHAPE)
    if structure:
        return _grade(case, reply, structure)
    model_input = case.get("input") or {}
    failures.extend(_response_fact_checks(model_input, reply))
    if _INTERNAL_RESPONSE.search(reply):
        failures.append(Violation("internal_field_leak", "reply", "natural Korean", reply))
    if model_input.get("execution_authorized") is not True and _claims_execution_started(reply):
        failures.append(Violation("unauthorized_execution_reply", "reply", "no start claim", reply))
    transition = (model_input.get("robot_state") or {}).get("section_transition") or {}
    completed = (model_input.get("robot_state") or {}).get("completed_tasks") or []
    completion_grounded = transition.get("outcome") == "completed" or bool(completed)
    if _claims_completion(reply) and not completion_grounded:
        failures.append(Violation("unverified_completion_reply", "reply", "verified completion fact", reply))

    applied = model_input.get("applied_this_turn") or {}
    for field, value in (applied.get("order_changes") or {}).items():
        item = _natural_field_name(field)
        # "오일로 바꿨어요"처럼 메뉴 값으로 말해도 변경 항목을 말한 것이다.
        spoken = {item, value} if value in _MENU_VALUES else {item}
        if not any(_mentions_name(reply, name) for name in spoken):
            failures.append(Violation("changed_item_missing", field, item, reply))
        if value in _AMOUNT_PATTERNS and not _AMOUNT_PATTERNS[value].search(reply):
            failures.append(Violation("changed_amount_missing", field, value, reply))
        if value is None and not re.search(r"취소|제거|빼|해제|선택하지", reply):
            failures.append(Violation("removal_missing", field, "removal wording", reply))

    issues = (model_input.get("policy") or {}).get("issues") or {}
    for category in ("unsupported", "invalid", "protected", "restriction_conflicts", "physical_conflicts"):
        for issue in issues.get(category) or []:
            if not isinstance(issue, dict):
                continue
            names = _issue_names(issue, category)
            if names and not any(_mentions_name(reply, name) for name in names):
                failures.append(Violation("policy_issue_missing", f"policy.issues.{category}", names[0], reply))

    if transition:
        active_values = [str(item.get("value") or "") for item in model_input.get("preferences") or []]
        # 철회 여부는 사용자 발화의 시간순 마지막 상태로 판단한다. assistant 추측·질문은 근거가 아니다.
        history = model_input.get("recent_history") or []
        withdrawn = [
            value
            for value in ("소시지", "게살", "양파", "버섯", "치즈", "매콤")
            if _user_preference_status(history, value) == "revoked"
            and not any(value in active for active in active_values)
        ]
        revival = any(
            (not _is_question(clause) or _PREMISE.search(clause))
            and re.search(rf"{re.escape(value)}.{{0,12}}(?:좋아|취향)|(?:좋아|취향).{{0,12}}{re.escape(value)}", clause)
            for clause in _clauses(reply)
            for value in withdrawn
        )
        if revival and not re.search(r"취소|지웠|철회|더\s*이상", reply):
            failures.append(Violation("withdrawn_preference_revived", "reply", active_values, reply))
        pending = model_input.get("pending") or {}
        order = model_input.get("confirmed_order") or {}
        for item, amount in (pending.get("items") or {}).items():
            if not any(_mentions_name(reply, name) for name in _generic_names({item})):
                failures.append(Violation("preselected_item_missing", "reply", item, reply))
            # 확인 질문의 수량은 옛 pending 값이 아니라 최신 확정 주문을 따른다.
            latest = order.get("noodle_portion") if item == order.get("noodle_type") else (order.get("toppings") or {}).get(item)
            amount = latest if latest is not None else amount
            if amount in _AMOUNT_PATTERNS and not _AMOUNT_PATTERNS[amount].search(reply):
                failures.append(Violation("preselected_amount_missing", "reply", amount, reply))
    return _grade(case, reply, failures)

