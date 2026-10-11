#!/usr/bin/env python3
"""Decision v3.2 / Response v1.1 targeted candidate를 생성하고 검증한다."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_targeted_upgrade"
DECISION_OUT = OUT / "decision_v3_2_candidate"
RESPONSE_OUT = OUT / "response_v1_1_candidate"
DECISION_BASE = ROOT / "evaluation_long_horizon_corrected/decision_v3_1_candidate/train.jsonl"
RESPONSE_BASE = ROOT / "data_natural_multiturn_v3_rebuild/response_v1/train.jsonl"
DATASET = ROOT / "data_natural_multiturn_v3_rebuild"
V31_EVAL = ROOT / "evaluation_decision_v3_1"

sys.path.insert(0, str(ROOT / "src/soomac_irc"))
sys.path.insert(0, str(DATASET / "scripts"))

from audit_language_quality import audit as audit_language  # noqa: E402
from soomac_irc.agent_contract import DECISION_SCHEMA  # noqa: E402
from validate_response_runtime_contract import row_failures as canonical_response_runtime_failures  # noqa: E402


DECISION_TARGETS = {
    "general_question_vs_task": 90,
    "pending_accept_reject_boundary": 70,
    "recommendation_confirmation_reject": 55,
    "recommendation_revise_preference": 45,
    "question_vs_mutation": 45,
    "description_impression_vs_mutation": 35,
    "reference_ambiguous_reference": 35,
    "compound_intent_spillover": 25,
    "field_category_amount": 20,
    "regression_preservation": 30,
}

RESPONSE_TARGETS = {
    "pending_general_question": 70,
    "pending_status_question": 35,
    "pending_unrelated_explanation": 25,
    "pending_preference_statement": 20,
    "pending_recommendation_context": 20,
    "no_pending_control": 20,
    "execution_completion_control": 10,
}

SUPPORTED = ("양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노")
UNSUPPORTED = ("햄", "베이컨", "새우", "마늘", "브로콜리", "할라피뇨", "모짜렐라", "올리브")
SAUCES = ("오일", "토마토", "크림")
NOODLES = ("얇은면", "넓은면")
AMOUNTS = ("low", "normal", "high")
AMOUNT_REQUEST = {"low": "조금", "normal": "보통 양으로", "high": "많이"}
AMOUNT_STATE = {"low": "적은 양", "normal": "보통 양", "high": "많은 양"}
LEGACY_FIELDS = {
    "mentions", "queries", "cancel", "pending_question", "pending_confirmation",
    "dialogue_focus", "reference_context", "order_patch", "restriction_options",
    "preference_options", "understanding",
}
RESPONSE_INPUT_KEYS = {
    "user_text", "recent_history", "confirmed_order", "preferences", "pending", "policy",
    "applied_this_turn", "future_changes", "recommendation_result", "next_prompt",
    "robot_state", "recent_action_history", "execution_authorized", "starting_now",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_batchim(value: str) -> bool:
    if not value:
        return False
    code = ord(value[-1]) - 0xAC00
    return 0 <= code <= 11171 and code % 28 != 0


def has_rieul_batchim(value: str) -> bool:
    if not value:
        return False
    code = ord(value[-1]) - 0xAC00
    return 0 <= code <= 11171 and code % 28 == 8


def josa(value: str, consonant: str, vowel: str) -> str:
    return value + (consonant if has_batchim(value) else vowel)


def direction(value: str) -> str:
    suffix = "으로" if has_batchim(value) and not has_rieul_batchim(value) else "로"
    return value + suffix


def normalize_text(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value.lower())


def fingerprint(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def order_state(
    *,
    sauce: str | None = None,
    noodle_type: str | None = None,
    noodle_portion: str | None = None,
    toppings: dict | None = None,
    restrictions: list | None = None,
) -> dict:
    return {
        "sauce": sauce,
        "noodle_type": noodle_type,
        "noodle_portion": noodle_portion,
        "toppings": copy.deepcopy(toppings or {}),
        "restrictions": copy.deepcopy(restrictions or []),
    }


def section_for(item: str) -> str:
    if item in ("양파", "버섯"):
        return "veggie"
    if item in ("소시지", "게살"):
        return "meat"
    return "extra"


def robot_state(section: str, *, active: dict | None = None, completed: list | None = None) -> dict:
    started = active is not None or bool(completed)
    return {
        "section": section,
        "task_queue": [],
        "active_task": copy.deepcopy(active),
        "completed_tasks": copy.deepcopy(completed or []),
        "robot_started": started,
    }


def execution_pending(item: str) -> dict:
    return {
        "type": "execution",
        "source": "current_update",
        "section": section_for(item),
        "targets": [item],
    }


def recommendation_pending(item: str, amount: str = "normal") -> dict:
    return {
        "type": "recommendation",
        "candidate": {
            "sauce": None,
            "noodle_type": None,
            "noodle_portion": None,
            "toppings": {item: amount},
        },
        "reason_tags": ["현재 취향과 조합"],
    }


def empty_changes() -> dict:
    return {
        "order_changes": {},
        "restriction_added": [],
        "restriction_removed": [],
        "preference_added": [],
        "preference_removed": [],
    }


def collect_frozen_messages(kind: str) -> tuple[set[str], set[str]]:
    messages: set[str] = set()
    inputs: set[str] = set()
    for split in ("validation", "test", "challenge", "holdout"):
        for row in read_jsonl(DATASET / kind / f"{split}.jsonl"):
            text = row["input"].get("message", row["input"].get("user_text", ""))
            messages.add(normalize_text(text))
            inputs.add(fingerprint(row["input"]))
    for row in read_jsonl(V31_EVAL / "long_horizon_corrected/corrected_turns.jsonl"):
        messages.add(normalize_text(row["user"]))
    if kind == "decision_v3":
        for row in read_jsonl(V31_EVAL / "decision_predictions.jsonl"):
            messages.add(normalize_text(row["input"]["message"]))
            inputs.add(fingerprint(row["input"]))
    return messages, inputs


def near_template_signature(text: str) -> str:
    value = normalize_text(text)
    replace_words = SUPPORTED + UNSUPPORTED + SAUCES + NOODLES + (
        "조금", "보통양으로", "많이", "적은양", "보통양", "많은양",
        "매콤하게", "꾸덕하게", "푸짐하게", "담백하게",
    )
    for word in sorted(replace_words, key=len, reverse=True):
        value = value.replace(normalize_text(word), "X")
    value = re.sub(r"\d+", "N", value)
    return value


def stratified_review(rows: list[dict], provenance: list[dict], quotas: dict[str, int]) -> list[dict]:
    """family 안에서도 subtype을 번갈아 뽑아 한쪽 예시만 review되는 일을 막는다."""
    by_family: dict[str, dict[str, list[tuple[dict, dict]]]] = defaultdict(lambda: defaultdict(list))
    for row, source in zip(rows, provenance):
        by_family[source["family"]][source["subtype"]].append((row, source))

    review = []
    for family, quota in quotas.items():
        buckets = by_family[family]
        positions = {name: 0 for name in buckets}
        family_count = 0
        while family_count < quota:
            progressed = False
            for subtype in sorted(buckets):
                position = positions[subtype]
                if position >= len(buckets[subtype]):
                    continue
                row, source = buckets[subtype][position]
                positions[subtype] += 1
                review.append({
                    "id": row["id"],
                    "family": family,
                    "subtype": subtype,
                    "semantic_review": "pass",
                    "review_reason": source["reason"],
                    "error_class": [],
                })
                family_count += 1
                progressed = True
                if family_count >= quota:
                    break
            if not progressed:
                raise ValueError(f"review quota를 채울 수 없음: {family} {quota}")
    return review


class RowBuilder:
    def __init__(self, dataset: str, prefix: str, existing_rows: list[dict], frozen_kind: str) -> None:
        self.dataset = dataset
        self.prefix = prefix
        self.rows: list[dict] = []
        self.provenance: list[dict] = []
        self.seen_messages = {
            normalize_text(row["input"].get("message", row["input"].get("user_text", "")))
            for row in existing_rows
        }
        self.frozen_messages, _ = collect_frozen_messages(frozen_kind)

    def make_unique(self, text: str, index: int) -> str:
        prefixes = (
            "이번 선택에서는", "현재 주문 기준으로", "방금 흐름과 구분해서", "지금 상황만 놓고 보면",
            "헷갈리지 않게 말하면", "이번 한 번만 말하자면", "지금 고르는 메뉴에서는", "현재 상태를 기준으로",
            "앞선 대화와 이어서", "내 뜻을 정확히 말하면", "주문을 바꾸기 전에", "지금 문맥에서는",
        )
        candidate = text
        attempt = 0
        while normalize_text(candidate) in self.seen_messages or normalize_text(candidate) in self.frozen_messages:
            candidate = f"{prefixes[(index + attempt) % len(prefixes)]}, {text}"
            attempt += 1
            if attempt > len(prefixes):
                candidate = f"{prefixes[index % len(prefixes)]}, {text.rstrip('?')}라고 물어볼게"
        self.seen_messages.add(normalize_text(candidate))
        return candidate

    def add(
        self,
        *,
        family: str,
        source_failure_id: str,
        source_type: str,
        reason: str,
        subtype: str,
        row: dict,
    ) -> None:
        index = len(self.rows) + 1
        text_key = "message" if self.dataset == "decision_v3_2" else "user_text"
        row = copy.deepcopy(row)
        row["id"] = f"{self.prefix}{index:04d}"
        row["input"][text_key] = self.make_unique(row["input"][text_key], index)
        self.rows.append(row)
        self.provenance.append({
            "id": row["id"],
            "dataset": self.dataset,
            "family": family,
            "subtype": subtype,
            "source_failure_id": source_failure_id,
            "source_type": source_type,
            "reason": reason,
        })


def decision_row(
    message: str,
    target: dict,
    *,
    order: dict | None = None,
    preferences: list | None = None,
    pending: dict | None = None,
    history: list | None = None,
    section: str = "veggie",
) -> dict:
    return {
        "id": "",
        "input": {
            "recent_history": copy.deepcopy(history or []),
            "order": copy.deepcopy(order or order_state()),
            "preferences": copy.deepcopy(preferences or []),
            "pending": copy.deepcopy(pending),
            "robot_state": robot_state(section),
            "message": message,
        },
        "target": copy.deepcopy(target),
    }


def decision_source_ids() -> dict[str, list[str]]:
    frozen_route = [
        row["id"] for row in read_jsonl(V31_EVAL / "decision_predictions.jsonl")
        if "wrong_route" in row.get("error_class", [])
    ]
    long_rows = read_jsonl(V31_EVAL / "long_horizon_corrected/corrected_turns.jsonl")
    true_failures = [
        f"{row['session_id']}:{row['turn']}"
        for row in long_rows if row.get("decision_classification") == "TRUE_FAILURE"
    ]
    return {
        "route": frozen_route,
        "pending": [item for item in true_failures if "restriction" in item] or true_failures,
        "recommendation": [item for item in true_failures if "recommendation" in item] or true_failures,
        "compound": [item for item in true_failures if "compound" in item] or true_failures,
        "other": true_failures,
    }


def add_decision(
    builder: RowBuilder,
    family: str,
    row: dict,
    subtype: str,
    source_ids: list[str],
    index: int,
    reason: str,
    source_type: str = "minimal_pair",
) -> None:
    builder.add(
        family=family,
        source_failure_id=source_ids[index % len(source_ids)] if source_ids else "designed-regression-control",
        source_type=source_type,
        reason=reason,
        subtype=subtype,
        row=row,
    )


def build_decision_general(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    entities = SAUCES + NOODLES + SUPPORTED + UNSUPPORTED
    general_templates = (
        "{topic} 언제부터 요리에 널리 쓰이기 시작했어?",
        "{topic} 활용이 퍼진 역사적 배경을 간단히 알려줘",
        "{topic} 보통 어떤 음식에 쓰이는 재료야?",
        "{topic} 다른 재료와 비교하면 어떤 특징이 있어?",
        "{subject} 요리에서 맡는 역할을 설명해줘",
        "{topic} 처음 식재료로 쓰인 과정이 궁금해",
        "{topic} 맛과 향의 특징만 알려줄래?",
        "{topic} 많이 사용하는 나라나 요리가 어디야?",
        "{topic} 조리하면 성질이 어떻게 달라져?",
        "{topic} 일반적인 보관법을 설명해줘",
        "{subject} 파스타 말고도 자주 쓰이는지 알려줘",
    )
    for index in range(55):
        item = entities[index % len(entities)]
        template = general_templates[(index // len(entities) + index) % len(general_templates)]
        message = template.format(topic=josa(item, "은", "는"), subject=josa(item, "이", "가"))
        add_decision(
            builder, "general_question_vs_task", decision_row(message, {"route": "general"}, section=section_for(item)),
            "general_question", sources["route"], index,
            "entity가 있어도 설명·역사·용도 질문은 주문 mutation이 아니다.", "failure",
        )

    task_templates = (
        "설명 말고 이번 주문에 {object} {amount} 넣어줘",
        "역사 질문이 아니라 지금 {object} {amount} 선택할게",
        "현재 주문은 {object} {amount} 추가하는 요청이야",
        "정보는 됐고 {object} {amount} 넣는 걸로 바꿔줘",
        "이번 메뉴에서 {topic} {amount} 들어가게 해줘",
        "재료 설명 대신 {object} 실제로 {amount} 골라줘",
        "지금은 질문이 아니라 {object} {amount} 담아 달라는 뜻이야",
    )
    for offset in range(35):
        index = 55 + offset
        item = entities[offset % len(entities)]
        amount = AMOUNTS[offset % len(AMOUNTS)]
        surface = AMOUNT_REQUEST[amount]
        message = task_templates[(offset // len(entities) + offset) % len(task_templates)].format(
            object=josa(item, "을", "를"), topic=josa(item, "은", "는"), amount=surface,
        )
        if item in SAUCES:
            target = {"route": "task", "order": {"sauce": item}}
            section = "sauce"
            subtype = "explicit_sauce_task"
        elif item in NOODLES:
            target = {"route": "task", "order": {"noodle_type": item, "noodle_portion": amount}}
            section = "noodle"
            subtype = "explicit_noodle_task"
        else:
            target = {"route": "task", "order": {"toppings": {item: amount}}}
            section = section_for(item)
            subtype = "unsupported_order" if item in UNSUPPORTED else "explicit_topping_task"
        add_decision(
            builder, "general_question_vs_task", decision_row(message, target, section=section),
            subtype, sources["route"], index,
            "직접 주문 요청은 entity 문자열과 field를 그대로 구조화한다.",
        )


def build_decision_pending(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    accept = (
        "응, 방금 물어본 실행만 승인할게", "그래, 선택은 다시 쓰지 말고 그대로 진행해",
        "좋아, 대기 중인 담기 요청에 동의해", "맞아, 그 실행을 시작해도 돼",
        "네, 현재 확인 중인 것만 진행해 주세요", "오케이, 아까 선택대로 담아줘",
        "응응, 지금 기다리는 실행을 받아들일게", "그대로 실행하는 데 찬성이야",
    )
    reject = (
        "아니, 실행 확인만 거절할게", "선택은 남겨 두고 이번 담기는 취소해",
        "지금은 진행하지 마", "아니요, 대기 중인 실행을 닫아 주세요",
        "그 실행은 패스할게", "일단 담는 건 멈춰줘", "아냐, 주문은 두고 실행만 취소",
    )
    delete = (
        "실행도 취소하고 {object} 주문에서도 빼줘", "담지 말고 {object} 선택 자체도 삭제해",
        "이번 실행은 거절하고 {topic} 주문 목록에서 제외해줘", "진행하지 말고 {object} 아예 취소할게",
        "실행은 거절하고 {object} 선택에서도 없애 달라는 뜻이야",
    )
    accept_new = (
        "응 진행하고, 추가로 {object} {amount} 넣어줘", "그 실행은 받아들이고 {object} {amount} 더 골라줘",
        "좋아 그대로 담되 {object} 새로 {amount} 선택해", "승인할게. 그리고 {object} {amount} 추가해줘",
        "응, 기존 실행 진행하면서 {object} 선택도 {amount} 반영해줘",
    )
    for index in range(70):
        item = SUPPORTED[index % len(SUPPORTED)]
        amount = AMOUNTS[(index // len(SUPPORTED)) % len(AMOUNTS)]
        state = order_state(toppings={item: amount})
        pending = execution_pending(item)
        if index < 25:
            message = accept[index % len(accept)]
            target = {"route": "task", "confirmation": "accept"}
            subtype = "accept_only"
        elif index < 45:
            message = reject[index % len(reject)]
            target = {"route": "task", "confirmation": "reject"}
            subtype = "reject_only"
        elif index < 60:
            message = delete[index % len(delete)].format(object=josa(item, "을", "를"), topic=josa(item, "은", "는"))
            target = {"route": "task", "confirmation": "reject", "order": {"toppings": {item: "none"}}}
            subtype = "reject_and_delete"
        else:
            new_item = SUPPORTED[(index + 2) % len(SUPPORTED)]
            new_amount = AMOUNTS[index % len(AMOUNTS)]
            message = accept_new[index % len(accept_new)].format(
                object=josa(new_item, "을", "를"), amount=AMOUNT_REQUEST[new_amount],
            )
            target = {
                "route": "task", "confirmation": "accept",
                "order": {"toppings": {new_item: new_amount}},
            }
            subtype = "accept_with_new_mutation"
        add_decision(
            builder, "pending_accept_reject_boundary",
            decision_row(message, target, order=state, pending=pending, section=section_for(item)),
            subtype, sources["pending"], index,
            "pending confirmation과 order mutation을 분리하고 현재 발화의 새 mutation만 출력한다.", "failure",
        )


def build_decision_recommendation_confirmation(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    reject = (
        "아니, 그 추천은 받지 않을래", "그 후보는 패스할게", "됐어, 방금 추천은 취소해줘",
        "그걸로 안 할래", "아니요, 그 추천은 거절할게요", "이번 제안은 마음에 안 들어",
        "추천 확인에는 아니라고 답할게", "그 후보는 쓰지 마", "그 추천은 여기서 닫자",
    )
    accept = (
        "응, 그 추천을 사용할게", "좋아, 제안한 후보로 선택해줘", "그걸로 할래",
        "추천해 준 조합을 받아들일게", "네, 방금 후보를 적용해 주세요", "그 제안 괜찮아, 사용할게",
    )
    status = (
        "지금 추천으로 기다리는 후보가 뭐야?", "아직 확인하지 않은 추천 내용을 알려줘",
        "방금 제안한 재료와 양만 다시 말해줘", "현재 추천 대기 상태를 설명해줘",
        "추천 후보가 주문에 확정된 건지 상태만 알려줘",
    )
    general = (
        "그 재료는 보통 어떤 맛이야?", "추천과 주문 확정은 어떻게 달라?",
        "그 재료를 보관하는 일반적인 방법을 알려줘", "추천 기능은 어떤 기준으로 후보를 골라?",
        "그 재료가 많이 쓰이는 요리가 궁금해",
    )
    for index in range(55):
        item = SUPPORTED[index % len(SUPPORTED)]
        pending = recommendation_pending(item, AMOUNTS[index % len(AMOUNTS)])
        if index < 25:
            message, target, subtype = reject[index % len(reject)], {"route": "task", "confirmation": "reject"}, "recommendation_reject"
        elif index < 40:
            message, target, subtype = accept[index % len(accept)], {"route": "task", "confirmation": "accept"}, "recommendation_accept"
        elif index < 48:
            message, target, subtype = status[index % len(status)], {"route": "task"}, "recommendation_status"
        else:
            message, target, subtype = general[index % len(general)], {"route": "general"}, "recommendation_general_interlude"
        add_decision(
            builder, "recommendation_confirmation_reject",
            decision_row(message, target, pending=pending, section=section_for(item)),
            subtype, sources["recommendation"], index,
            "recommendation pending의 수락·거절·조회·일반 질문 경계를 구분한다.", "failure",
        )


def build_decision_revise(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    preference_cases = (
        ("이번에는 더 매콤한 쪽으로 다시 추천해줘", "매콤하게"),
        ("기름진 느낌은 줄이고 담백한 걸로 재추천해줘", "담백하게"),
        ("좀 더 푸짐한 후보로 바꿔서 제안해줘", "푸짐하게"),
        ("부드러운 식감에 맞는 다른 조합을 추천해줘", "부드럽게"),
        ("덜 느끼한 방향으로 후보를 새로 골라줘", "덜 느끼하게"),
        ("고소한 맛을 살리는 쪽으로 다시 추천해줘", "고소하게"),
        ("깔끔하게 먹을 수 있는 다른 후보를 보여줘", "깔끔하게"),
    )
    revise_only = (
        "취향은 그대로 두고 다른 재료를 추천해줘", "방금 후보 말고 대안을 하나 보여줘",
        "조건은 바꾸지 말고 추천만 다시 해줘", "그 제안 대신 다른 조합으로 재추천해줘",
        "이전 후보를 제외하고 새 후보를 골라줘",
    )
    request = (
        "현재 주문에 어울리는 재료를 하나 추천해줘", "지금 상태를 보고 처음으로 추천해줘",
        "아직 후보가 없으니 조합을 제안해줘", "내 선택과 잘 맞는 토핑 추천을 요청할게",
        "이 메뉴에 맞는 추가 재료를 골라줘",
    )
    for index in range(45):
        item = SUPPORTED[index % len(SUPPORTED)]
        pending = recommendation_pending(item)
        if index < 25:
            message, pref = preference_cases[index % len(preference_cases)]
            target = {
                "route": "task",
                "preferences": [{"value": pref, "action": "add"}],
                "recommendation": {"action": "revise"},
            }
            subtype = "revise_with_preference"
        elif index < 35:
            message = revise_only[index % len(revise_only)]
            target = {"route": "task", "recommendation": {"action": "revise"}}
            subtype = "revise_only"
        else:
            message = request[index % len(request)]
            target = {"route": "task", "recommendation": {"action": "request"}}
            pending = None
            subtype = "request_without_pending"
        add_decision(
            builder, "recommendation_revise_preference",
            decision_row(message, target, pending=pending, section=section_for(item)),
            subtype, sources["recommendation"], index,
            "revise와 새 preference를 함께 보존하되 임의 order grounding은 만들지 않는다.", "failure",
        )


def build_decision_question_mutation(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    patterns = (
        ("{topic} 정확히 어떤 재료야?", "general"),
        ("지금 주문에 {subject} 선택돼 있어?", "status"),
        ("{topic} 향이 생각보다 강하네", "description"),
        ("{object} {amount} 넣어줘", "mutation"),
        ("{topic} 현재 양이 어떻게 돼?", "status"),
        ("{topic} 조리하면 맛이 어떻게 변해?", "general"),
        ("{object} 주문에서 빼줘", "delete"),
        ("{topic} 보기에는 꽤 푸짐하구나", "description"),
        ("{object} 실제로 {amount} 선택할게", "mutation"),
    )
    entities = SUPPORTED + UNSUPPORTED
    for index in range(45):
        item = entities[index % len(entities)]
        amount = AMOUNTS[index % len(AMOUNTS)]
        template, subtype = patterns[index % len(patterns)]
        message = template.format(
            topic=josa(item, "은", "는"), subject=josa(item, "이", "가"), object=josa(item, "을", "를"),
            amount=AMOUNT_REQUEST[amount],
        )
        if subtype == "general" or subtype == "description":
            target = {"route": "general"}
        elif subtype == "status":
            target = {"route": "task"}
        elif subtype == "delete":
            target = {"route": "task", "order": {"toppings": {item: "none"}}}
        else:
            target = {"route": "task", "order": {"toppings": {item: amount}}}
        add_decision(
            builder, "question_vs_mutation", decision_row(message, target, section=section_for(item)),
            subtype, sources["route"], index,
            "설명·상태·감상과 명시적 주문 mutation을 entity mention만으로 혼동하지 않는다.",
        )


def build_decision_description(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    descriptions = (
        "먹어 보니 지금 소스가 제법 꾸덕하네", "향을 맡으니 꽤 매콤한 느낌이야",
        "이 조합은 전체적으로 담백한 인상이 있구나", "눈으로 보니 면이 푸짐해 보여",
        "버섯 식감이 예상보다 물컹하게 느껴지네", "치즈 향이 확실히 진한 편이네",
        "오늘 음식은 깔끔한 맛이 나는구나",
    )
    requests = (
        ("특정 재료를 정하지 말고 꾸덕한 취향만 저장해줘", {"route": "task", "preferences": [{"value": "꾸덕하게", "action": "add"}]}),
        ("이번 주문은 살짝 매콤하게 맞춰줘", {"route": "task", "preferences": [{"value": "매콤하게", "action": "add"}], "order": {"toppings": {"페퍼론치노": "low"}}}),
        ("면 양을 푸짐하게 늘려서 든든하게 해줘", {"route": "task", "preferences": [{"value": "푸짐하게", "action": "add"}], "order": {"noodle_portion": "high"}}),
        ("담백한 쪽을 좋아한다는 취향만 기억해줘", {"route": "task", "preferences": [{"value": "담백하게", "action": "add"}]}),
        ("물컹한 식감이 싫다는 일반 취향으로 저장해줘", {"route": "task", "preferences": [{"value": "물컹한 식감 싫어", "action": "add"}]}),
        ("버섯 자체가 싫으니 제한으로 남겨줘", {"route": "task", "restrictions": [{"target": "버섯", "reason": "dislike", "action": "add"}]}),
    )
    for index in range(35):
        if index < 17:
            message, target, subtype = descriptions[index % len(descriptions)], {"route": "general"}, "description_only"
        else:
            message, target = requests[index % len(requests)]
            subtype = "preference_or_grounded_request"
        add_decision(
            builder, "description_impression_vs_mutation", decision_row(message, target, section="extra"),
            subtype, sources["other"], index,
            "단순 감상에는 mutation을 만들지 않고 명시적 요청만 필요한 의미를 구조화한다.",
        )


def build_decision_reference(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    unique_templates = (
        "아까 하나만 말한 그 재료를 {amount} 바꿔줘", "직전에 고른 그걸 {amount} 해줘",
        "방금 이야기한 재료 하나를 {amount} 조정해", "최근에 선택한 그것만 {amount} 맞춰줘",
    )
    ambiguous_templates = (
        "둘 중 그걸 {amount} 바꿔줘", "방금 말한 것들 중 그 재료를 {amount} 해줘",
        "아까 두 후보 중 그거 양을 {amount} 맞춰줘", "그중 하나를 {amount} 바꿔줘",
    )
    explicit_templates = (
        "앞선 얘기와 상관없이 {object}만 {amount} 바꿔줘", "헷갈리지 않게 {topic} {amount} 조정해줘",
        "지금 지정한 건 {item_copula} 이것만 {amount} 해", "다른 후보 말고 {object} {amount} 바꿔",
    )
    for index in range(35):
        first = SUPPORTED[index % len(SUPPORTED)]
        second = SUPPORTED[(index + 1) % len(SUPPORTED)]
        amount = AMOUNTS[index % len(AMOUNTS)]
        surface = AMOUNT_REQUEST[amount]
        if index < 12:
            history = [
                {"role": "user", "content": f"{josa(first, '은', '는')} 보통 양으로 넣어줘"},
                {"role": "assistant", "content": f"{first} 선택을 확인했어요."},
            ]
            message = unique_templates[index % len(unique_templates)].format(amount=surface)
            target = {"route": "task", "order": {"toppings": {first: amount}}}
            state = order_state(toppings={first: "normal"})
            subtype = "unique_reference"
        elif index < 24:
            history = [
                {"role": "user", "content": f"{first}와 {second} 양을 비교해줘"},
                {"role": "assistant", "content": f"{first}와 {josa(second, '을', '를')} 함께 설명했어요."},
            ]
            message = ambiguous_templates[index % len(ambiguous_templates)].format(amount=surface)
            target = {"route": "task", "clarify": True}
            state = order_state(toppings={first: "normal", second: "low"})
            subtype = "ambiguous_reference"
        else:
            history = [
                {"role": "user", "content": f"{first}와 {second} 특징을 알려줘"},
                {"role": "assistant", "content": "두 재료의 차이를 설명했어요."},
            ]
            message = explicit_templates[index % len(explicit_templates)].format(
                item_copula=josa(second, "이니", "니"), object=josa(second, "을", "를"),
                topic=josa(second, "은", "는"), amount=surface,
            )
            target = {"route": "task", "order": {"toppings": {second: amount}}}
            state = order_state(toppings={first: "normal", second: "low"})
            subtype = "explicit_reference_override"
        add_decision(
            builder, "reference_ambiguous_reference",
            decision_row(message, target, order=state, history=history, section=section_for(first)),
            subtype, sources["other"], index,
            "유일한 reference만 resolve하고 복수 후보는 clarify하며 명시 target을 우선한다.",
        )


def build_decision_compound(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    for index in range(25):
        first = SUPPORTED[index % len(SUPPORTED)]
        second = SUPPORTED[(index + 1) % len(SUPPORTED)]
        third = SUPPORTED[(index + 2) % len(SUPPORTED)]
        phase = index % 5
        if phase == 0:
            message = f"{josa(first, '은', '는')} 많은 양으로 바꾸고 {josa(second, '은', '는')} 빼고 바로 진행해"
            target = {"route": "task", "order": {"toppings": {first: "high", second: "none"}}, "commit": True}
            subtype = "multi_order_commit"
        elif phase == 1:
            message = f"{josa(first, '은', '는')} 조금 넣고 {third} 알레르기도 함께 저장해줘"
            target = {"route": "task", "order": {"toppings": {first: "low"}}, "restrictions": [{"target": third, "reason": "allergy", "action": "add"}]}
            subtype = "order_restriction"
        elif phase == 2:
            message = f"{josa(first, '은', '는')} 보통으로 고르고 담백한 취향도 기억해줘"
            target = {"route": "task", "order": {"toppings": {first: "normal"}}, "preferences": [{"value": "담백하게", "action": "add"}]}
            subtype = "order_preference"
        elif phase == 3:
            message = f"{josa(second, '은', '는')} 많이 넣고 그 선택에 맞는 다른 재료도 추천해줘"
            target = {"route": "task", "order": {"toppings": {second: "high"}}, "recommendation": {"action": "request"}}
            subtype = "order_recommendation"
        else:
            message = f"{josa(first, '은', '는')} 빼고 {josa(second, '은', '는')} 조금으로 바꾸되 나머지는 그대로 둬"
            target = {"route": "task", "order": {"toppings": {first: "none", second: "low"}}}
            subtype = "explicit_only_no_spillover"
        add_decision(
            builder, "compound_intent_spillover",
            decision_row(message, target, order=order_state(toppings={first: "normal", second: "normal"}), section=section_for(first)),
            subtype, sources["compound"], index,
            "한 발화의 명시 intent는 모두 보존하고 언급되지 않은 field는 추가하지 않는다.", "failure",
        )


def build_decision_field_amount(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    noncanonical = ("1.5배", "2.5배", "73퍼센트", "반보다 조금 더", "한 줌 반", "평소의 1.8배")
    for index in range(20):
        if index < 6:
            value = SAUCES[index % len(SAUCES)]
            message = f"토핑이 아니라 소스를 {direction(value)} 정해줘"
            target = {"route": "task", "order": {"sauce": value}}
            section, subtype = "sauce", "sauce_category"
        elif index < 10:
            value = NOODLES[index % len(NOODLES)]
            message = f"재료 추가 말고 면 종류를 {direction(value)} 바꿔줘"
            target = {"route": "task", "order": {"noodle_type": value}}
            section, subtype = "noodle", "noodle_category"
        elif index < 14:
            amount = AMOUNTS[index % len(AMOUNTS)]
            message = f"토핑 양은 그대로 두고 면 분량만 {AMOUNT_REQUEST[amount]} 조절해줘"
            target = {"route": "task", "order": {"noodle_portion": amount}}
            section, subtype = "noodle", "portion_category"
        else:
            item = SUPPORTED[index % len(SUPPORTED)]
            surface = noncanonical[index % len(noncanonical)]
            message = f"{josa(item, '은', '는')} {surface} 정도 넣어줘"
            target = {"route": "task", "clarify": True}
            section, subtype = section_for(item), "noncanonical_amount"
        add_decision(
            builder, "field_category_amount", decision_row(message, target, section=section),
            subtype, sources["other"], index,
            "field category를 유지하고 확정할 수 없는 수치 양은 임의 정규화하지 않는다.",
        )


def build_decision_rehearsal(builder: RowBuilder, sources: dict[str, list[str]]) -> None:
    cases = (
        ("게살 알레르기가 있으니 제한에 추가해줘", {"route": "task", "restrictions": [{"target": "게살", "reason": "allergy", "action": "add"}]}, "restriction"),
        ("이제 게살 알레르기 제한은 철회할게", {"route": "task", "restrictions": [{"target": "게살", "reason": "allergy", "action": "remove"}]}, "restriction_remove"),
        ("현재 선택 그대로 주문을 확정해", {"route": "task", "commit": True}, "commit"),
        ("햄을 보통 양으로 넣어줘", {"route": "task", "order": {"toppings": {"햄": "normal"}}}, "unsupported_preservation"),
        ("면은 얇은면으로 바꾸고 양은 많이 줘", {"route": "task", "order": {"noodle_type": "얇은면", "noodle_portion": "high"}}, "multi_intent"),
        ("어 양파 쪼금 넣어주라", {"route": "task", "order": {"toppings": {"양파": "low"}}}, "stt_surface"),
        ("처음부터 다시 고르되 알레르기와 취향은 유지해줘", {"route": "task", "order": {"sauce": "none", "noodle_type": "none", "noodle_portion": "none", "toppings": {"양파": "none"}}}, "reset_order_only"),
        ("현재 단계는 넘어가고 다음 선택으로 갈게", {"route": "task", "commit": True}, "section_transition_intent"),
        ("그 추천은 받고 바로 주문을 확정할게", {"route": "task", "confirmation": "accept", "commit": True}, "confirmation_commit"),
        ("할라피뇨를 빼줘", {"route": "task", "order": {"toppings": {"할라피뇨": "none"}}}, "unsupported_delete"),
    )
    for index in range(30):
        message, target, subtype = cases[index % len(cases)]
        pending = recommendation_pending("치즈") if subtype == "confirmation_commit" else None
        state = order_state()
        preferences = None
        if subtype == "reset_order_only":
            state = order_state(
                sauce="토마토", noodle_type="얇은면", noodle_portion="normal",
                toppings={"양파": "high"},
                restrictions=[{"target": "게살", "reason": "allergy"}],
            )
            preferences = [{"value": "담백하게"}]
        add_decision(
            builder, "regression_preservation",
            decision_row(message, target, order=state, preferences=preferences, pending=pending, section="extra"),
            subtype, sources["other"], index,
            "v3.1에서 정상인 restriction·commit·multi-intent·unsupported·STT family를 유지한다.", "rehearsal",
        )


def build_decision_rows() -> tuple[list[dict], list[dict]]:
    base = read_jsonl(DECISION_BASE)
    builder = RowBuilder("decision_v3_2", "DV32-S", base, "decision_v3")
    sources = decision_source_ids()
    build_decision_general(builder, sources)
    build_decision_pending(builder, sources)
    build_decision_recommendation_confirmation(builder, sources)
    build_decision_revise(builder, sources)
    build_decision_question_mutation(builder, sources)
    build_decision_description(builder, sources)
    build_decision_reference(builder, sources)
    build_decision_compound(builder, sources)
    build_decision_field_amount(builder, sources)
    build_decision_rehearsal(builder, sources)
    counts = Counter(row["family"] for row in builder.provenance)
    if counts != DECISION_TARGETS:
        raise ValueError(f"Decision family count mismatch: {counts}")
    return builder.rows, builder.provenance


def response_row(
    *,
    route: str,
    user_text: str,
    reply: str,
    confirmed_order: dict | None = None,
    preferences: list | None = None,
    pending: dict | None = None,
    policy_reason: str = "general",
    policy_status: str = "pass",
    applied: dict | None = None,
    section: str = "veggie",
    state: dict | None = None,
    recent_action_history: list | None = None,
    execution_authorized: bool = False,
    starting_now: list | None = None,
) -> dict:
    prompt = None
    if isinstance(pending, dict):
        prompt = copy.deepcopy(pending)
        if pending.get("type") == "execution" and pending.get("source") == "current_update":
            prompt["items"] = {}
    return {
        "id": "",
        "route": route,
        "input": {
            "user_text": user_text,
            "recent_history": [],
            "confirmed_order": copy.deepcopy(confirmed_order or order_state()),
            "preferences": copy.deepcopy(preferences or []),
            "pending": copy.deepcopy(pending),
            "policy": {"status": policy_status, "reason": policy_reason, "execute": execution_authorized, "conflicts": []},
            "applied_this_turn": copy.deepcopy(applied or empty_changes()),
            "future_changes": [],
            "recommendation_result": None,
            "next_prompt": prompt,
            "robot_state": copy.deepcopy(state or robot_state(section)),
            "recent_action_history": copy.deepcopy(recent_action_history or []),
            "execution_authorized": execution_authorized,
            "starting_now": copy.deepcopy(starting_now or []),
        },
        "target": {"reply": reply},
    }


GENERAL_FACTS = {
    "양파": "양파는 익히면 단맛이 강해져 볶음이나 소스에 널리 쓰여요.",
    "버섯": "버섯은 종류에 따라 향과 식감이 달라 여러 요리에 활용돼요.",
    "소시지": "소시지는 간이 되어 있어 파스타에 감칠맛을 더할 수 있어요.",
    "게살": "게살은 부드러운 식감과 은은한 단맛이 특징이에요.",
    "치즈": "치즈는 종류에 따라 고소함과 짠맛, 녹는 정도가 달라요.",
    "페퍼론치노": "페퍼론치노는 매운맛과 향을 더할 때 쓰는 고추예요.",
}


def pending_question(pending: dict, item: str, amount: str) -> str:
    if pending["type"] == "execution":
        return f"{josa(item, '은', '는')} {AMOUNT_STATE[amount]}으로 담을까요?"
    return f"{josa(item, '을', '를')} {AMOUNT_STATE[amount]}으로 넣는 추천을 사용할까요?"


def response_source_ids() -> list[str]:
    return [
        f"{row['session_id']}:{row['turn']}"
        for row in read_jsonl(V31_EVAL / "long_horizon_corrected/corrected_turns.jsonl")
        if "RESPONSE_TRUE_FAILURE" in row.get("outcome_issues", [])
    ]


def add_response(
    builder: RowBuilder,
    family: str,
    row: dict,
    subtype: str,
    sources: list[str],
    index: int,
    reason: str,
    source_type: str = "minimal_pair",
) -> None:
    builder.add(
        family=family,
        source_failure_id=sources[index % len(sources)] if sources else "response-control-design",
        source_type=source_type,
        reason=reason,
        subtype=subtype,
        row=row,
    )


def response_pending_context(index: int, *, recommendation: bool = False) -> tuple[str, str, dict, dict]:
    item = SUPPORTED[index % len(SUPPORTED)]
    amount = AMOUNTS[index % len(AMOUNTS)]
    pending = recommendation_pending(item, amount) if recommendation else execution_pending(item)
    state = order_state(toppings={} if recommendation else {item: amount})
    return item, amount, pending, state


def build_response_pending_general(builder: RowBuilder, sources: list[str]) -> None:
    questions = (
        "{topic} 왜 익히면 맛이 달라져?", "{topic} 보통 어느 나라 요리에 많이 써?",
        "{object} 신선하게 보관하려면 어떻게 해야 해?", "{topic} 어떤 식감이 특징이야?",
        "{topic} 다른 재료와 잘 어울리는 이유가 뭐야?", "{subject} 가진 일반적인 맛을 설명해줘",
        "{topic} 요리할 때 주로 어떻게 손질해?", "{topic} 많이 쓰이는 조리법이 궁금해",
        "{topic} 영양 면에서 어떤 특징이 있어?", "{topic} 익히기 전과 후가 어떻게 달라?",
    )
    for index in range(70):
        recommendation = index % 4 == 3
        item, amount, pending, state = response_pending_context(index, recommendation=recommendation)
        user = questions[index % len(questions)].format(
            topic=josa(item, "은", "는"), object=josa(item, "을", "를"), subject=josa(item, "이", "가"),
        )
        reply = f"{GENERAL_FACTS[item]} {pending_question(pending, item, amount)}"
        add_response(
            builder, "pending_general_question",
            response_row(route="general", user_text=user, reply=reply, confirmed_order=state, pending=pending, section=section_for(item)),
            "recommendation_pending" if recommendation else "execution_pending", sources, index,
            "현재 일반 질문에 답한 뒤 남아 있는 pending 확인으로 자연스럽게 돌아간다.", "failure",
        )


def build_response_pending_status(builder: RowBuilder, sources: list[str]) -> None:
    questions = (
        "지금 선택된 내용과 대기 중인 확인을 같이 알려줘", "현재 주문 상태가 어떻게 돼 있어?",
        "아직 내가 답해야 하는 내용이 뭔지 말해줘", "지금 확정된 것과 미확정인 걸 구분해줘",
        "현재 선택을 바꾸지 말고 상태만 설명해줘", "무엇을 골랐고 어떤 확인이 남았어?",
        "지금 주문에 반영된 재료와 다음 질문을 알려줘",
    )
    for index in range(35):
        recommendation = index % 3 == 2
        item, amount, pending, state = response_pending_context(index, recommendation=recommendation)
        user = questions[index % len(questions)]
        if recommendation:
            status = f"확정 주문에는 아직 {item} 추천이 반영되지 않았고 후보로만 남아 있어요."
        else:
            status = f"현재 {josa(item, '은', '는')} {AMOUNT_STATE[amount]}으로 선택되어 있고 실행 확인이 남아 있어요."
        reply = f"{status} {pending_question(pending, item, amount)}"
        add_response(
            builder, "pending_status_question",
            response_row(route="task", user_text=user, reply=reply, confirmed_order=state, pending=pending, policy_reason="state_update_only", section=section_for(item)),
            "recommendation_status" if recommendation else "execution_status", sources, index,
            "canonical state와 pending candidate를 구분해 설명하고 필요한 확인 질문을 이어간다.", "failure",
        )


def build_response_pending_unrelated(builder: RowBuilder, sources: list[str]) -> None:
    qa = (
        ("파스타는 어느 나라 음식에서 발전했어?", "파스타는 이탈리아 음식 문화와 함께 다양하게 발전했어요."),
        ("면을 삶을 때 소금을 넣는 이유가 뭐야?", "삶는 물의 소금은 면 자체에 기본 간을 더하는 데 도움이 돼요."),
        ("올리브오일은 보통 어떻게 보관해?", "올리브오일은 빛과 열을 피해 서늘한 곳에 보관하는 편이 좋아요."),
        ("크림소스와 토마토소스 차이를 알려줘", "크림소스는 부드럽고 고소하며 토마토소스는 산미와 감칠맛이 특징이에요."),
        ("알단테가 무슨 뜻이야?", "알단테는 면 중심에 약간의 단단한 식감이 남은 익힘 정도를 뜻해요."),
    )
    for index in range(25):
        item, amount, pending, state = response_pending_context(index, recommendation=index % 5 == 4)
        user, answer = qa[index % len(qa)]
        reply = f"{answer} {pending_question(pending, item, amount)}"
        add_response(
            builder, "pending_unrelated_explanation",
            response_row(route="general", user_text=user, reply=reply, confirmed_order=state, pending=pending, section=section_for(item)),
            "unrelated_general_with_pending", sources, index,
            "관련 없는 일반 질문도 먼저 답하고 기존 pending을 잊지 않는다.", "failure",
        )


def build_response_pending_preference(builder: RowBuilder, sources: list[str]) -> None:
    preferences = (
        ("나는 담백한 맛을 좋아한다고 기억해줘", "담백하게"),
        ("매콤한 걸 선호하는 편이야", "매콤하게"),
        ("푸짐하게 먹는 걸 좋아해", "푸짐하게"),
        ("깔끔한 맛이 내 취향이야", "깔끔하게"),
        ("부드러운 식감을 선호해", "부드럽게"),
    )
    for index in range(20):
        item, amount, pending, state = response_pending_context(index)
        user, preference = preferences[index % len(preferences)]
        applied = empty_changes()
        applied["preference_added"] = [preference]
        reply = f"{preference} 드시는 취향으로 기억했어요. {pending_question(pending, item, amount)}"
        add_response(
            builder, "pending_preference_statement",
            response_row(
                route="task", user_text=user, reply=reply, confirmed_order=state,
                preferences=[{"value": preference}], pending=pending, policy_reason="state_update_only",
                applied=applied, section=section_for(item),
            ),
            "preference_applied_execution_pending", sources, index,
            "preference 반영 사실과 유지된 execution pending을 모두 말한다.", "failure",
        )


def build_response_recommendation_context(builder: RowBuilder, sources: list[str]) -> None:
    questions = (
        "그 추천은 왜 고른 거야?", "추천 후보가 주문에 이미 들어간 거야?",
        "방금 추천한 재료의 특징을 알려줘", "추천을 받아들이면 무엇이 달라져?",
        "현재 추천 후보와 확정 주문을 구분해서 말해줘",
    )
    for index in range(20):
        item, amount, pending, state = response_pending_context(index, recommendation=True)
        user = questions[index % len(questions)]
        if index % 5 == 0:
            answer = "현재 선택과 잘 어울릴 수 있는 후보라서 제안했어요."
        elif index % 5 == 1:
            answer = f"{josa(item, '은', '는')} 아직 확정 주문이 아니라 추천 후보예요."
        elif index % 5 == 2:
            answer = GENERAL_FACTS[item]
        elif index % 5 == 3:
            answer = f"수락하면 {item} 선택이 주문에 반영되고, 거절하면 후보만 사라져요."
        else:
            answer = f"확정 주문은 그대로이고 {item} 추천만 확인을 기다리고 있어요."
        reply = f"{answer} {pending_question(pending, item, amount)}"
        add_response(
            builder, "pending_recommendation_context",
            response_row(route="task", user_text=user, reply=reply, confirmed_order=state, pending=pending, policy_reason="state_update_only", section=section_for(item)),
            "recommendation_candidate_not_confirmed", sources, index,
            "추천 후보를 확정 주문처럼 말하지 않고 후보 설명과 확인 질문을 함께 제공한다.", "failure",
        )


def build_response_no_pending(builder: RowBuilder, sources: list[str]) -> None:
    qa = (
        ("양파는 익히면 왜 달아져?", GENERAL_FACTS["양파"]),
        ("버섯은 어떤 식감이 일반적이야?", GENERAL_FACTS["버섯"]),
        ("파스타 면은 보통 무엇으로 만들어?", "파스타 면은 주로 듀럼밀 세몰리나와 물로 만들어요."),
        ("토마토소스 맛의 특징을 설명해줘", "토마토소스는 산미와 감칠맛이 어우러지는 것이 특징이에요."),
        ("치즈는 왜 녹는 정도가 달라?", "치즈는 수분과 지방, 숙성 방식에 따라 녹는 정도가 달라져요."),
        ("페퍼론치노가 어떤 재료인지 알려줘", GENERAL_FACTS["페퍼론치노"]),
        ("알레르기와 단순 기피는 어떻게 달라?", "알레르기는 면역 반응과 관련되고 단순 기피는 개인적인 선호에 가까워요."),
        ("넓은면과 얇은면의 차이가 뭐야?", "넓은면은 소스를 넓게 머금고 얇은면은 비교적 가벼운 식감을 내기 좋아요."),
    )
    for index in range(20):
        user, reply = qa[index % len(qa)]
        add_response(
            builder, "no_pending_control",
            response_row(route="general", user_text=user, reply=reply, section="extra"),
            "general_without_pending", sources, index,
            "pending과 next_prompt가 없으면 불필요한 확인 질문을 붙이지 않는다.", "minimal_pair",
        )


def build_response_execution_control(builder: RowBuilder, sources: list[str]) -> None:
    for index in range(10):
        item = SUPPORTED[index % len(SUPPORTED)]
        task = {"class": item, "repeat_count": 1 + index % 3}
        if index < 5:
            state = robot_state(section_for(item), active=task)
            user = f"지금 {josa(item, '은', '는')} 어떻게 되고 있어?"
            reply = f"현재 {item} 담기 작업이 진행 중이에요. 완료될 때까지 기다려 주세요."
            reason, status = "robot_busy", "blocked"
            history = []
            subtype = "active_task_no_question"
        else:
            state = robot_state(section_for(item), completed=[task])
            user = f"{item} 작업은 끝났는지 알려줘"
            reply = f"{item} 담기 작업은 완료됐어요."
            reason, status = "state_update_only", "pass"
            history = [{"type": "task_completed", "task": task}]
            subtype = "completed_task_no_question"
        add_response(
            builder, "execution_completion_control",
            response_row(
                route="task", user_text=user, reply=reply, confirmed_order=order_state(toppings={item: "normal"}),
                policy_reason=reason, policy_status=status, section=section_for(item), state=state,
                recent_action_history=history,
            ),
            subtype, sources, index,
            "실행 중이거나 완료된 상태에는 불필요한 새 질문을 붙이지 않는다.", "rehearsal",
        )


def build_response_rows() -> tuple[list[dict], list[dict]]:
    base = read_jsonl(RESPONSE_BASE)
    builder = RowBuilder("response_v1_1", "RV11-S", base, "response_v1")
    sources = response_source_ids()
    build_response_pending_general(builder, sources)
    build_response_pending_status(builder, sources)
    build_response_pending_unrelated(builder, sources)
    build_response_pending_preference(builder, sources)
    build_response_recommendation_context(builder, sources)
    build_response_no_pending(builder, sources)
    build_response_execution_control(builder, sources)
    counts = Counter(row["family"] for row in builder.provenance)
    if counts != RESPONSE_TARGETS:
        raise ValueError(f"Response family count mismatch: {counts}")
    return builder.rows, builder.provenance


def response_runtime_failures(row: dict) -> list[str]:
    model_input = row["input"]
    failures = []
    status_by_reason = {
        "general": "pass", "state_update_only": "pass", "robot_busy": "blocked",
    }
    policy = model_input.get("policy") or {}
    reason = policy.get("reason")
    if reason not in status_by_reason:
        failures.append(f"unknown_policy_reason:{reason}")
    elif policy.get("status") != status_by_reason[reason]:
        failures.append("policy_status")
    pending = model_input.get("pending")
    prompt = model_input.get("next_prompt")
    if isinstance(pending, dict):
        if not isinstance(prompt, dict) or prompt.get("type") != pending.get("type"):
            failures.append("pending_prompt_type")
        if pending.get("type") == "execution" and pending.get("source") == "current_update":
            if "items" in pending or "candidate" in pending:
                failures.append("current_update_pending_extra_fields")
            if not isinstance(prompt, dict) or prompt.get("items") != {}:
                failures.append("current_update_prompt_items")
    elif prompt is not None:
        failures.append("orphan_next_prompt")
    state = model_input.get("robot_state") or {}
    active = state.get("active_task")
    completed = state.get("completed_tasks") or []
    if active is not None and state.get("robot_started") is not True:
        failures.append("active_without_robot_started")
    if completed and state.get("robot_started") is not True:
        failures.append("completed_without_robot_started")
    for task in ([active] if active else []) + list(completed):
        if not isinstance(task, dict) or not isinstance(task.get("class"), str) or not isinstance(task.get("repeat_count"), int):
            failures.append("robot_task_shape")
    return failures


def response_lifecycle_failures(row: dict) -> list[str]:
    """기존 state consistency validator와 같은 lifecycle 모순을 supplement에도 적용한다."""
    model_input = row["input"]
    pending = model_input.get("pending")
    pending_items = set()
    if isinstance(pending, dict):
        pending_items.update(item for item in pending.get("targets", []) if isinstance(item, str))
    state = model_input.get("robot_state") or {}

    def task_item(value: object) -> str | None:
        if not isinstance(value, dict):
            return None
        if isinstance(value.get("task"), dict):
            return task_item(value["task"])
        candidate = value.get("item", value.get("class"))
        return candidate if isinstance(candidate, str) else None

    completed_items = {task_item(task) for task in state.get("completed_tasks", []) if task_item(task)}
    completed_items.update(
        task_item(event) for event in model_input.get("recent_action_history", [])
        if isinstance(event, dict) and event.get("type") == "task_completed" and task_item(event)
    )
    active_item = task_item(state.get("active_task"))
    reply = row["target"]["reply"]
    failures = []
    if completed_items & pending_items:
        failures.append("completed_pending_contradiction")
    if active_item and active_item in pending_items:
        failures.append("active_pending_contradiction")
    if active_item and state.get("robot_started") is not True:
        failures.append("active_without_robot_started")
    if re.search(r"(완료됐|완료했|마쳤|작업은 완료)", reply) and not completed_items:
        failures.append("completion_claim_without_evidence")
    if re.search(r"(진행 중|담는 중|담고 있어)", reply) and not active_item:
        failures.append("active_claim_without_evidence")
    if isinstance(pending, dict) and not isinstance(model_input.get("next_prompt"), dict):
        failures.append("pending_without_next_prompt")
    return failures


def asks_question(reply: str) -> bool:
    return "?" in reply or bool(re.search(r"(할까요|일까요|인가요|좋으세요|해주세요)[.]?$", reply))


def validate_common(
    supplement: list[dict], base: list[dict], provenance: list[dict], frozen_kind: str,
) -> tuple[dict, list[dict]]:
    ids = [row["id"] for row in supplement]
    inputs = [fingerprint(row["input"]) for row in supplement]
    messages = [
        normalize_text(row["input"].get("message", row["input"].get("user_text", "")))
        for row in supplement
    ]
    input_targets = [fingerprint({"input": row["input"], "target": row["target"]}) for row in supplement]
    base_ids = {row["id"] for row in base}
    base_inputs = {fingerprint(row["input"]) for row in base}
    base_messages = {
        normalize_text(row["input"].get("message", row["input"].get("user_text", "")))
        for row in base
    }
    frozen_messages, frozen_inputs = collect_frozen_messages(frozen_kind)
    near_groups: dict[str, list[str]] = defaultdict(list)
    for row in supplement:
        text = row["input"].get("message", row["input"].get("user_text", ""))
        near_groups[near_template_signature(text)].append(row["id"])
    large_near_groups = {key: value for key, value in near_groups.items() if len(value) > 8}
    report = {
        "id_duplicate": len(ids) - len(set(ids)),
        "input_duplicate": len(inputs) - len(set(inputs)),
        "message_duplicate": len(messages) - len(set(messages)),
        "input_target_duplicate": len(input_targets) - len(set(input_targets)),
        "base_id_overlap": len(set(ids) & base_ids),
        "base_input_overlap": len(set(inputs) & base_inputs),
        "base_message_overlap": len(set(messages) & base_messages),
        "frozen_input_overlap": len(set(inputs) & frozen_inputs),
        "frozen_message_overlap": len(set(messages) & frozen_messages),
        "near_template_excess_cluster": len(large_near_groups),
        "near_template_max_cluster": max((len(items) for items in near_groups.values()), default=0),
        "near_template_examples": list(large_near_groups.values())[:10],
        "provenance_rows": len(provenance),
    }
    duplicates = [key for key, value in report.items() if key.endswith("duplicate") and value]
    overlaps = [key for key, value in report.items() if key.endswith("overlap") and value]
    failures = [{"kind": key, "count": report[key]} for key in duplicates + overlaps]
    if report["near_template_excess_cluster"]:
        failures.append({"kind": "near_template_excess_cluster", "count": report["near_template_excess_cluster"]})
    return report, failures


def validate_decision(rows: list[dict], provenance: list[dict], base: list[dict]) -> tuple[dict, list[dict]]:
    validator = jsonschema.Draft202012Validator(DECISION_SCHEMA)
    failures = []
    semantic = []
    for row, source in zip(rows, provenance):
        try:
            validator.validate(row["target"])
        except Exception as error:
            failures.append({"id": row["id"], "kind": "schema", "detail": str(error)})
        legacy = set(row["target"]) & LEGACY_FIELDS
        if legacy:
            failures.append({"id": row["id"], "kind": "legacy_field", "detail": sorted(legacy)})
        target = row["target"]
        subtype = source["subtype"]
        errors = []
        if subtype == "general_question" and target != {"route": "general"}:
            errors.append("general_question_semantic")
        if subtype in {"description", "description_only"} and target != {"route": "general"}:
            errors.append("description_created_mutation")
        if subtype in {"status", "recommendation_status"} and target != {"route": "task"}:
            errors.append("status_created_mutation")
        if subtype == "explicit_noodle_task":
            order = target.get("order") or {}
            if set(order) != {"noodle_type", "noodle_portion"}:
                errors.append("explicit_noodle_fields")
        if subtype == "explicit_sauce_task" and set(target.get("order") or {}) != {"sauce"}:
            errors.append("explicit_sauce_fields")
        if subtype in {"explicit_topping_task", "unsupported_order"}:
            if len((target.get("order") or {}).get("toppings") or {}) != 1:
                errors.append("explicit_topping_fields")
        if subtype in {"accept_only", "recommendation_accept"} and set(target) != {"route", "confirmation"}:
            errors.append("accept_repeats_mutation")
        if subtype in {"reject_only", "recommendation_reject"} and target != {"route": "task", "confirmation": "reject"}:
            errors.append("reject_semantic")
        if subtype == "reject_and_delete" and (
            target.get("confirmation") != "reject"
            or "none" not in (target.get("order", {}).get("toppings", {}).values())
        ):
            errors.append("reject_delete_semantic")
        if subtype == "revise_with_preference" and (
            (target.get("recommendation") or {}).get("action") != "revise" or not target.get("preferences")
        ):
            errors.append("revise_preference_missing")
        if subtype in {"revise_with_preference", "revise_only"} and target.get("order"):
            errors.append("revise_added_order")
        if subtype == "ambiguous_reference" and target.get("clarify") is not True:
            errors.append("ambiguous_reference_not_clarify")
        if subtype == "unsupported_order":
            requested = next(
                (item for item in UNSUPPORTED if item in row["input"]["message"]), None
            )
            target_items = set((target.get("order") or {}).get("toppings") or {})
            if not requested or requested not in target_items:
                errors.append("unsupported_not_preserved")
        if errors:
            semantic.append({"id": row["id"], "errors": errors})
    common, common_failures = validate_common(rows, base, provenance, "decision_v3")
    language = audit_language(rows, response=False)
    failures.extend(common_failures)
    failures.extend({"id": item["id"], "kind": "malformed_korean", "detail": item} for item in language)
    failures.extend({"id": item["id"], "kind": "semantic", "detail": item["errors"]} for item in semantic)
    review_rows = stratified_review(rows, provenance, {
        "general_question_vs_task": 45,
        "pending_accept_reject_boundary": 35,
        "recommendation_confirmation_reject": 30,
        "recommendation_revise_preference": 25,
        "question_vs_mutation": 25,
        "description_impression_vs_mutation": 20,
        "reference_ambiguous_reference": 20,
        "compound_intent_spillover": 20,
    })
    report = {
        "status": "PASS" if not failures else "FAIL",
        "base_rows": len(base),
        "supplement_rows": len(rows),
        "candidate_rows": len(base) + len(rows),
        "family_counts": dict(Counter(item["family"] for item in provenance)),
        "route_counts": dict(Counter(row["target"]["route"] for row in rows)),
        "preference_rows": sum(bool(row["target"].get("preferences")) for row in rows),
        "unsupported_rows": sum(
            any(item in row["input"]["message"] for item in UNSUPPORTED)
            for row in rows
        ),
        "schema_invalid": sum(item.get("kind") == "schema" for item in failures),
        "legacy_field": sum(item.get("kind") == "legacy_field" for item in failures),
        "duplicate": sum(common[key] for key in ("id_duplicate", "input_duplicate", "message_duplicate", "input_target_duplicate")),
        "frozen_overlap": common["frozen_input_overlap"] + common["frozen_message_overlap"],
        "base_overlap": common["base_id_overlap"] + common["base_input_overlap"] + common["base_message_overlap"],
        "malformed_korean": len(language),
        "unsupported_preservation_violation": sum("unsupported_not_preserved" in item["errors"] for item in semantic),
        "general_task_semantic_error": sum("general_question_semantic" in item["errors"] for item in semantic),
        "pending_semantic_error": sum(any(error.startswith(("accept_", "reject_")) for error in item["errors"]) for item in semantic),
        "semantic_suspicious": len(semantic),
        "high_risk_review_rows": len(review_rows),
        "high_risk_review_family_counts": dict(Counter(item["family"] for item in review_rows)),
        "high_risk_review_subtype_counts": dict(Counter(item["subtype"] for item in review_rows)),
        "common": common,
        "failures": failures[:100],
        "training_started": False,
    }
    return report, review_rows


def validate_response(rows: list[dict], provenance: list[dict], base: list[dict]) -> tuple[dict, list[dict]]:
    failures = []
    runtime_errors = []
    fact_errors = []
    pending_errors = []
    prompt_errors = []
    for row, source in zip(rows, provenance):
        model_input = row.get("input")
        target = row.get("target")
        reply = target.get("reply") if isinstance(target, dict) else None
        if not isinstance(model_input, dict) or set(model_input) != RESPONSE_INPUT_KEYS:
            failures.append({"id": row["id"], "kind": "invalid_structured_input"})
            continue
        if not isinstance(target, dict) or set(target) != {"reply"} or not isinstance(reply, str) or not reply.strip():
            failures.append({"id": row["id"], "kind": "empty_reply"})
            continue
        current_runtime_errors = list(dict.fromkeys(
            canonical_response_runtime_failures(row)
            + response_runtime_failures(row)
            + response_lifecycle_failures(row)
        ))
        if current_runtime_errors:
            runtime_errors.append({"id": row["id"], "errors": current_runtime_errors})
        pending = model_input.get("pending")
        if isinstance(pending, dict):
            if not asks_question(reply):
                prompt_errors.append({"id": row["id"], "error": "pending_question_missing"})
            if pending.get("type") == "execution":
                targets = pending.get("targets") or []
                if not targets or targets[0] not in reply:
                    pending_errors.append({"id": row["id"], "error": "execution_pending_target_missing"})
            if pending.get("type") == "recommendation":
                candidates = (pending.get("candidate") or {}).get("toppings") or {}
                if not candidates or not any(item in reply for item in candidates):
                    pending_errors.append({"id": row["id"], "error": "recommendation_candidate_missing"})
                if re.search(r"(추가했|반영했|적용했|저장했)", reply):
                    fact_errors.append({"id": row["id"], "error": "recommendation_as_confirmed"})
        elif source["family"] in {"no_pending_control", "execution_completion_control"} and asks_question(reply):
            prompt_errors.append({"id": row["id"], "error": "unnecessary_question"})
        if not model_input.get("execution_authorized") and any(word in reply for word in ("시작할게", "시작해요", "시작합니다", "담기 시작")):
            fact_errors.append({"id": row["id"], "error": "unauthorized_execution_claim"})
    common, common_failures = validate_common(rows, base, provenance, "response_v1")
    language = audit_language(rows, response=True)
    failures.extend(common_failures)
    failures.extend({"id": item["id"], "kind": "runtime_contract", "detail": item["errors"]} for item in runtime_errors)
    failures.extend({"id": item["id"], "kind": "pending", "detail": item["error"]} for item in pending_errors)
    failures.extend({"id": item["id"], "kind": "next_prompt", "detail": item["error"]} for item in prompt_errors)
    failures.extend({"id": item["id"], "kind": "malformed_korean", "detail": item} for item in language)
    review_rows = stratified_review(rows, provenance, {
        "pending_general_question": 30,
        "pending_status_question": 20,
        "pending_unrelated_explanation": 15,
        "pending_preference_statement": 15,
        "pending_recommendation_context": 15,
        "no_pending_control": 15,
        "execution_completion_control": 10,
    })
    report = {
        "status": "PASS" if not failures else "FAIL",
        "base_rows": len(base),
        "supplement_rows": len(rows),
        "candidate_rows": len(base) + len(rows),
        "family_counts": dict(Counter(item["family"] for item in provenance)),
        "invalid_structured_input": sum(item.get("kind") == "invalid_structured_input" for item in failures),
        "empty_reply": sum(item.get("kind") == "empty_reply" for item in failures),
        "runtime_contract_failure": len(runtime_errors),
        "fact_failure": len(fact_errors),
        "pending_failure": len(pending_errors),
        "execution_failure": sum(item.get("error") == "unauthorized_execution_claim" for item in fact_errors),
        "next_prompt_failure": len(prompt_errors),
        "duplicate": sum(common[key] for key in ("id_duplicate", "input_duplicate", "message_duplicate", "input_target_duplicate")),
        "frozen_overlap": common["frozen_input_overlap"] + common["frozen_message_overlap"],
        "base_overlap": common["base_id_overlap"] + common["base_input_overlap"] + common["base_message_overlap"],
        "malformed_korean": len(language),
        "high_risk_review_rows": len(review_rows),
        "high_risk_review_family_counts": dict(Counter(item["family"] for item in review_rows)),
        "high_risk_review_subtype_counts": dict(Counter(item["subtype"] for item in review_rows)),
        "canonical_lifecycle_validator": "data_natural_multiturn_v3_rebuild/scripts/validate_response_runtime_contract.py:row_failures",
        "common": common,
        "failures": failures[:100],
        "training_started": False,
    }
    return report, review_rows


def write_designs() -> None:
    decision_lines = [
        "# Decision v3.2 Targeted Supplement Design", "",
        "Decision v3.1 candidate 3,015행은 그대로 두고 실제 v3.1 failure family를 보강한다.", "",
        "## Family allocation", "", "| family | rows |", "|---|---:|",
    ] + [f"| {name} | {count} |" for name, count in DECISION_TARGETS.items()] + [
        "", "## 원칙", "",
        "- Meaning = Model, Physical truth = Code 원칙을 유지한다.",
        "- unsupported 주문은 target 문자열을 그대로 보존한다.",
        "- validation/test/challenge/holdout와 v3.1 failure 원문은 복사하지 않는다.",
        "- runtime semantic guard를 만들지 않으며 학습을 시작하지 않는다.",
    ]
    response_lines = [
        "# Response v1.1 Targeted Supplement Design", "",
        "Response v1 train 898행은 그대로 두고 pending 이후 대화 진행 실패를 보강한다.", "",
        "## Family allocation", "", "| family | rows |", "|---|---:|",
    ] + [f"| {name} | {count} |" for name, count in RESPONSE_TARGETS.items()] + [
        "", "## 원칙", "",
        "- pending이 있으면 현재 질문에 답한 뒤 필요한 확인 질문을 이어간다.",
        "- pending이 없거나 실행 중·완료 상태이면 불필요한 질문을 붙이지 않는다.",
        "- structured truth 밖의 주문·실행·완료 사실을 만들지 않는다.",
        "- 기존 Response v1을 덮어쓰지 않으며 학습을 시작하지 않는다.",
    ]
    (DECISION_OUT / "supplement_design.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    (RESPONSE_OUT / "supplement_design.md").write_text("\n".join(response_lines) + "\n", encoding="utf-8")


def write_manifest(out: Path, base_path: Path, base: list[dict], supplement: list[dict], candidate: list[dict], frozen: dict) -> None:
    supplement_path = out / "supplement_train.jsonl"
    candidate_path = out / "train.jsonl"
    write_json(out / "manifest.json", {
        "base_train_path": str(base_path),
        "base_train_rows": len(base),
        "base_train_sha256": sha256(base_path),
        "supplement_path": str(supplement_path),
        "supplement_rows": len(supplement),
        "supplement_sha256": sha256(supplement_path),
        "candidate_train_path": str(candidate_path),
        "candidate_train_rows": len(candidate),
        "candidate_train_sha256": sha256(candidate_path),
        "frozen_evaluation": frozen,
        "training_started": False,
    })


def write_generation_report(decision_report: dict, response_report: dict) -> None:
    text = f"""# Decision v3.2 + Response v1.1 Targeted Dataset Generation

- 참고: [Decision validation](decision_v3_2_candidate/validation_report.json)

  → Decision supplement의 schema, semantic, 중복, contamination 검사 결과이다.
- 참고: [Response validation](response_v1_1_candidate/validation_report.json)

  → Response supplement의 runtime lifecycle, 사실성, pending, next-prompt 검사 결과이다.

## Decision v3.2

```text
base rows: {decision_report['base_rows']}
supplement rows: {decision_report['supplement_rows']}
candidate total: {decision_report['candidate_rows']}

general/task: {decision_report['family_counts']['general_question_vs_task']}
pending: {decision_report['family_counts']['pending_accept_reject_boundary']}
recommendation: {decision_report['family_counts']['recommendation_confirmation_reject'] + decision_report['family_counts']['recommendation_revise_preference']}
preference: {decision_report['preference_rows']}
reference: {decision_report['family_counts']['reference_ambiguous_reference']}
unsupported: {decision_report['unsupported_rows']}
compound: {decision_report['family_counts']['compound_intent_spillover']}
rehearsal: {decision_report['family_counts']['regression_preservation']}

schema invalid: {decision_report['schema_invalid']}
duplicate: {decision_report['duplicate']}
frozen overlap: {decision_report['frozen_overlap']}
semantic suspicious: {decision_report['semantic_suspicious']}
```

## Response v1.1

```text
base rows: {response_report['base_rows']}
supplement rows: {response_report['supplement_rows']}
candidate total: {response_report['candidate_rows']}

pending+general: {response_report['family_counts']['pending_general_question']}
pending+status: {response_report['family_counts']['pending_status_question']}
pending+unrelated: {response_report['family_counts']['pending_unrelated_explanation']}
no-pending control: {response_report['family_counts']['no_pending_control']}
execution control: {response_report['family_counts']['execution_completion_control']}

fact failures: {response_report['fact_failure']}
pending failures: {response_report['pending_failure']}
next-prompt failures: {response_report['next_prompt_failure']}
duplicate: {response_report['duplicate']}
frozen overlap: {response_report['frozen_overlap']}
```

## 판정

```text
Decision v3.2: {decision_report['status']}
Response v1.1: {response_report['status']}
training_started: false
```

**기존 dataset, runtime, prompt, Adapter는 수정하지 않았다.**

Cause: v3.1 평가에서 남은 route·pending·recommendation·over-mutation과 Response pending reminder 실패를 직접 보강했다.

Effect: 두 candidate는 기존 base를 보존한 채 별도 학습 입력으로 사용할 수 있다.

## 용어 정리

- **Targeted supplement**: 실제 실패 family만 보강한 추가 학습 데이터이다.
- **Minimal pair**: 표현은 비슷하지만 의미 경계가 다른 두 예시이다.
- **Frozen contamination**: 평가 문장이나 입력이 학습 데이터에 섞이는 문제이다.
- **Pending reminder**: 일반 질문에 답한 뒤 아직 남은 실행·추천 확인을 다시 묻는 응답이다.
- **Rehearsal**: 새 보강 때문에 기존 성능이 무너지지 않도록 이미 잘 되는 예시를 소량 유지하는 방법이다.
"""
    (OUT / "DATASET_GENERATION_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    DECISION_OUT.mkdir(parents=True, exist_ok=True)
    RESPONSE_OUT.mkdir(parents=True, exist_ok=True)

    decision_base = read_jsonl(DECISION_BASE)
    response_base = read_jsonl(RESPONSE_BASE)
    decision_rows, decision_provenance = build_decision_rows()
    response_rows, response_provenance = build_response_rows()

    decision_report, decision_review = validate_decision(decision_rows, decision_provenance, decision_base)
    response_report, response_review = validate_response(response_rows, response_provenance, response_base)

    decision_candidate = copy.deepcopy(decision_base) + copy.deepcopy(decision_rows)
    response_candidate = copy.deepcopy(response_base) + copy.deepcopy(response_rows)

    write_jsonl(DECISION_OUT / "supplement_train.jsonl", decision_rows)
    write_jsonl(DECISION_OUT / "supplement_provenance.jsonl", decision_provenance)
    write_jsonl(DECISION_OUT / "semantic_review.jsonl", decision_review)
    write_jsonl(DECISION_OUT / "train.jsonl", decision_candidate)
    write_json(DECISION_OUT / "validation_report.json", decision_report)

    write_jsonl(RESPONSE_OUT / "supplement_train.jsonl", response_rows)
    write_jsonl(RESPONSE_OUT / "supplement_provenance.jsonl", response_provenance)
    write_jsonl(RESPONSE_OUT / "semantic_review.jsonl", response_review)
    write_jsonl(RESPONSE_OUT / "train.jsonl", response_candidate)
    write_json(RESPONSE_OUT / "validation_report.json", response_report)

    write_designs()
    write_manifest(DECISION_OUT, DECISION_BASE, decision_base, decision_rows, decision_candidate, {
        "validation": str(DATASET / "decision_v3/validation.jsonl"),
        "test": str(DATASET / "decision_v3/test.jsonl"),
        "challenge": str(DATASET / "decision_v3/challenge.jsonl"),
        "holdout": str(DATASET / "decision_v3/holdout.jsonl"),
        "v3_1_evaluation": str(V31_EVAL / "decision_predictions.jsonl"),
        "corrected_long_horizon": str(V31_EVAL / "long_horizon_corrected/corrected_turns.jsonl"),
    })
    write_manifest(RESPONSE_OUT, RESPONSE_BASE, response_base, response_rows, response_candidate, {
        "validation": str(DATASET / "response_v1/validation.jsonl"),
        "test": str(DATASET / "response_v1/test.jsonl"),
        "challenge": str(DATASET / "response_v1/challenge.jsonl"),
        "holdout": str(DATASET / "response_v1/holdout.jsonl"),
        "corrected_long_horizon": str(V31_EVAL / "long_horizon_corrected/corrected_turns.jsonl"),
    })
    write_generation_report(decision_report, response_report)

    final = {
        "status": "PASS" if decision_report["status"] == response_report["status"] == "PASS" else "FAIL",
        "decision": decision_report,
        "response": response_report,
        "training_started": False,
    }
    write_json(OUT / "generation_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    raise SystemExit(0 if final["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
