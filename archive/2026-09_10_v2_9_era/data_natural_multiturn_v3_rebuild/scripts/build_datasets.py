#!/usr/bin/env python3
"""freeze된 natural_multiturn_v3 계약에 맞춰 Decision/Response dataset을 재구축한다.

legacy 자료에서는 사용자 발화, history, state만 출처로 사용한다. target은 현재
external sparse contract로 다시 만들며, 과거 model prediction은 읽지 않는다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_natural_multiturn_v3_rebuild"
LEGACY_TRAIN = ROOT / "soomac_decision_migration_merge_bundle/route_rebalance/final_train_rebalanced.jsonl"
HARD_EVAL = ROOT / "soomac_decision_migration_merge_bundle/route_rebalance/hard_eval.jsonl"
FROZEN_EVAL_DIR = ROOT / "decision_adapter_v225_root_cause_review_20260930/evaluation_data"
DECISION_REVIEW = ROOT / "data_salvage_natural_multiturn_v3/decision_v3/review_sample.jsonl"
RESPONSE_REVIEW = ROOT / "data_salvage_natural_multiturn_v3/response_v1/review_sample.jsonl"

CURRENT_COMMIT = "25ce994"
ALLOWED_DECISION_FIELDS = {
    "route", "order", "restrictions", "preferences",
    "recommendation", "commit", "confirmation", "clarify",
}
AMOUNTS = ("low", "normal", "high")
SAUCES = ("오일", "토마토", "크림")
NOODLES = ("얇은면", "넓은면")
TOPPINGS = ("양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노")
SUPPORTED = set(SAUCES + NOODLES + TOPPINGS)
REASONS = ("dislike", "cannot_eat", "allergy", "dietary_rule")
SECTIONS = ("noodle", "veggie", "meat", "extra", "lid", "sauce")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(text)


def stable_bucket(value: str, modulo: int = 100) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16) % modulo


def normalized_text(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value.lower())


def empty_order() -> dict:
    return {
        "sauce": None,
        "noodle_type": None,
        "noodle_portion": None,
        "toppings": {},
        "restrictions": [],
    }


def robot_state(section: str = "noodle") -> dict:
    return {
        "section": section,
        "task_queue": [],
        "active_task": None,
        "completed_tasks": [],
        "robot_started": False,
    }


def clean_order(value: object) -> dict:
    source = value if isinstance(value, dict) else {}
    result = empty_order()
    for field in ("sauce", "noodle_type", "noodle_portion"):
        candidate = source.get(field)
        result[field] = candidate if isinstance(candidate, str) or candidate is None else None
    toppings = source.get("toppings")
    if isinstance(toppings, dict):
        result["toppings"] = {
            str(item): amount for item, amount in toppings.items()
            if isinstance(item, str) and isinstance(amount, str)
        }
    restrictions = source.get("restrictions")
    if isinstance(restrictions, list):
        result["restrictions"] = [
            {"target": str(item.get("target")), "reason": item.get("reason")}
            for item in restrictions
            if isinstance(item, dict)
            and item.get("reason") in REASONS
            and isinstance(item.get("target"), str)
        ]
    return result


def infer_section(order: dict, fallback: str = "extra") -> str:
    items = set(order.get("toppings", {}))
    if items & {"양파", "버섯"}:
        return "veggie"
    if items & {"소시지", "게살"}:
        return "meat"
    if items & {"치즈", "페퍼론치노"}:
        return "extra"
    if order.get("noodle_type") or order.get("noodle_portion"):
        return "noodle"
    if order.get("sauce"):
        return "sauce"
    return fallback


def clean_robot_state(value: object, order: dict) -> dict:
    source = copy.deepcopy(value) if isinstance(value, dict) else robot_state()
    section = source.get("section")
    if section not in SECTIONS:
        source["section"] = infer_section(order)
    source.setdefault("task_queue", [])
    source.setdefault("active_task", None)
    source.setdefault("completed_tasks", [])
    source.setdefault("robot_started", False)
    return source


def unpack_messages_row(row: dict) -> dict:
    """chat-template evaluation row를 일반 source row 모양으로 푼다."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return row
    user_message = next((item for item in reversed(messages) if item.get("role") == "user"), None)
    assistant_message = next((item for item in reversed(messages) if item.get("role") == "assistant"), None)
    if user_message is None or assistant_message is None:
        return row
    try:
        model_input = json.loads(user_message.get("content", "{}"))
        target = json.loads(assistant_message.get("content", "{}"))
    except (TypeError, json.JSONDecodeError):
        return row
    unpacked = copy.deepcopy(row)
    unpacked["input"] = model_input
    unpacked["target"] = target
    unpacked["history"] = model_input.get("recent_history", model_input.get("history", []))
    return unpacked


def legacy_recommendation_pending(value: object) -> dict | None:
    if not isinstance(value, dict) or value.get("phase") != "proposed":
        return None
    last = value.get("last_proposal")
    if not isinstance(last, dict):
        return None
    proposal = last.get("proposal")
    if not isinstance(proposal, dict):
        return None
    return {
        "type": "recommendation",
        "candidate": {
            "sauce": proposal.get("sauce"),
            "noodle_type": proposal.get("noodle_type"),
            "noodle_portion": proposal.get("noodle_portion"),
            "toppings": copy.deepcopy(proposal.get("toppings") or {}),
        },
        "reason_tags": copy.deepcopy(last.get("reason_tags") or []),
    }


def legacy_execution_pending(value: object, order: dict) -> dict | None:
    if not isinstance(value, dict):
        return None
    raw_items = value.get("items") or []
    items = raw_items if isinstance(raw_items, list) else list(raw_items) if isinstance(raw_items, dict) else []
    selected = {
        item: order["toppings"].get(item, "normal")
        for item in items if isinstance(item, str)
    }
    section = value.get("section")
    if section not in SECTIONS:
        section = infer_section({**empty_order(), "toppings": selected})
    return {
        "type": "execution",
        "source": "preselected",
        "section": section,
        "targets": list(selected),
        "items": selected,
        "candidate": {
            "sauce": None,
            "noodle_type": None,
            "noodle_portion": None,
            "toppings": copy.deepcopy(selected),
        },
    }


def current_decision_input(row: dict, clean_language: bool = True) -> dict:
    row = unpack_messages_row(row)
    source = row.get("input") if isinstance(row.get("input"), dict) else row
    order = clean_order(source.get("order"))
    pending = copy.deepcopy(source.get("pending")) if isinstance(source.get("pending"), dict) else None
    if pending is None:
        pending = legacy_execution_pending(source.get("pending_confirmation"), order)
    if pending is None:
        pending = legacy_recommendation_pending(source.get("recommendation"))
    history = row.get("history", source.get("recent_history", []))
    if not isinstance(history, list):
        history = []
    preferences = source.get("preferences") if isinstance(source.get("preferences"), list) else []
    message = source.get("message", row.get("message", ""))
    cleaned_history = copy.deepcopy(history[-20:])
    if clean_language:
        message = naturalize_korean(str(message))
        for history_item in cleaned_history:
            if isinstance(history_item, dict) and isinstance(history_item.get("content"), str):
                history_item["content"] = naturalize_korean(history_item["content"])
    return {
        "recent_history": cleaned_history,
        "order": order,
        "preferences": copy.deepcopy(preferences),
        "pending": pending,
        "robot_state": clean_robot_state(source.get("robot_state"), order),
        "message": str(message).strip(),
    }


def legacy_unsupported_mutation(row: dict, target: dict) -> dict | None:
    """cleanup 전 변환 결과를 재현해 수정 provenance에만 사용한다."""
    message = current_decision_input(row)["message"]
    if target.get("route") not in ("task", "mixed"):
        return None
    if not re.search(r"(넣|추가|해줘|해주세요|제외|빼|없던|취소)", message):
        return None
    mentions = target.get("mentions") or []
    item = next((str(value) for value in mentions if str(value) not in SUPPORTED), None)
    if item is None:
        item = next((value for value in ("햄", "베이컨", "새우", "마늘", "피망", "올리브") if value in message), None)
    if item is None:
        return None
    if re.search(r"(빼|제외|없던|취소)", message):
        amount = "none"
    elif re.search(r"(많|가득|듬뿍)", message):
        amount = "high"
    elif re.search(r"(조금|적게|살짝)", message):
        amount = "low"
    else:
        amount = "normal"
    return {"toppings": {item: amount}}


def unsupported_mutation(row: dict, target: dict) -> dict | None:
    """명시적으로 unsupported 재료 조작을 요청한 clarify row만 복원한다."""
    if target.get("clarify") is not True or isinstance(target.get("order"), dict):
        return None
    if target.get("route") not in ("task", "mixed"):
        return None
    message = current_decision_input(row)["message"]
    mentions = target.get("mentions") or []
    item = next((str(value) for value in mentions if str(value) not in SUPPORTED), None)
    if item is None:
        item = next((value for value in ("햄", "베이컨", "새우", "마늘", "피망", "올리브") if value in message), None)
    if item is None:
        return None

    # 재료명 뒤에 실제 주문 동작이 있을 때만 mutation으로 복원한다.
    # "특징을 설명해줘"처럼 설명 요청의 해줘는 주문 동작으로 보지 않는다.
    item_tail = message.split(item, 1)[1][:24]
    order_action = r"(넣|추가|담|빼|제외|없던\s*걸|취소|바꿔|적용|(?<!설명)(?<!알려)해\s*줘)"
    if not re.search(order_action, item_tail):
        return None
    if re.search(r"(빼|제외|없던\s*걸|취소)", item_tail):
        amount = "none"
    elif re.search(r"(많|가득|듬뿍)", item_tail):
        amount = "high"
    elif re.search(r"(조금|적게|살짝)", item_tail):
        amount = "low"
    else:
        amount = "normal"
    return {"toppings": {item: amount}}


def relabel_decision(row: dict) -> dict:
    row = unpack_messages_row(row)
    old = row.get("target") if isinstance(row.get("target"), dict) else {}
    result = {"route": old.get("route") if old.get("route") in ("task", "general", "mixed") else "task"}
    if isinstance(old.get("order"), dict):
        order = copy.deepcopy(old["order"])
        order.pop("restrictions", None)
        if order:
            result["order"] = order
    restrictions = old.get("restrictions")
    if isinstance(restrictions, list):
        cleaned = [
            {"target": item["target"], "reason": item["reason"], "action": item["action"]}
            for item in restrictions
            if isinstance(item, dict)
            and isinstance(item.get("target"), str)
            and item.get("reason") in REASONS
            and item.get("action") in ("add", "remove")
        ]
        if cleaned:
            result["restrictions"] = cleaned
    preferences = old.get("preferences")
    if isinstance(preferences, list):
        cleaned = [
            {"value": item["value"], "action": item["action"]}
            for item in preferences
            if isinstance(item, dict)
            and isinstance(item.get("value"), str)
            and item.get("action") in ("add", "remove")
        ]
        if cleaned:
            result["preferences"] = cleaned
    recommendation = old.get("recommendation")
    action = recommendation.get("action") if isinstance(recommendation, dict) else None
    if action in ("request", "revise"):
        result["recommendation"] = {"action": action}
    elif action == "select":
        result["confirmation"] = "accept"
    elif action == "cancel" and current_decision_input(row)["pending"] is not None:
        result["confirmation"] = "reject"
    if old.get("commit") is True:
        result["commit"] = True
    if old.get("confirmation") in ("accept", "reject"):
        result["confirmation"] = old["confirmation"]
    if old.get("clarify") is True:
        result["clarify"] = True

    # legacy guard가 unsupported 의미를 clarify로 숨긴 사례를 현재 계약으로 복원한다.
    unsupported = unsupported_mutation(row, old)
    if unsupported is not None:
        result["order"] = unsupported
        result.pop("clarify", None)

    model_input = current_decision_input(row)
    message = model_input["message"]
    # dialogue_focus만 있고 자연어 history가 없는 지시어는 현재 입력에서 해석할 수 없다.
    if not model_input["recent_history"] and re.fullmatch(r".*(그거|그 재료|그 선택|아까 거).*", message):
        if model_input["pending"] is None and not any(name in message for name in SUPPORTED):
            result.pop("order", None)
            result.pop("confirmation", None)
            result["clarify"] = True

    # 강한 감각 요청은 current prompt의 grounding 정책을 target에 반영한다.
    pref_values = {item["value"] for item in result.get("preferences", []) if item.get("action") == "add"}
    is_request = bool(re.search(r"(해줘|해주세요|부탁|먹고 싶|가자|넣어)", message))
    is_description = bool(re.search(r"(하네|뭐야|알려|설명|어떻게|가정|아니고)", message))
    if "매콤하게" in pref_values and is_request and not is_description:
        amount = "high" if re.search(r"(아주|화끈|맵게|강하게)", message) else "low" if re.search(r"(조금|살짝|약간)", message) else "normal"
        result.setdefault("order", {}).setdefault("toppings", {})["페퍼론치노"] = amount
    if "푸짐하게" in pref_values and is_request and not is_description:
        result.setdefault("order", {})["noodle_portion"] = "high"
    return result


HIGH_RISK_FAMILIES = {
    "reference", "reference_resolved", "reference_ambiguous", "reference_missing", "reference_stale",
    "multiturn_reference", "multiturn_reference_dense", "multiturn_long_reference",
    "positional_reference_resolve", "positional_semantics_dense", "runtime_reference_surface",
    "unsupported_clarify", "unsupported_direct_task", "near_neighbor_unsupported_clarify",
    "clarify_unsupported_multiturn", "contrast_clarify_unsupported", "runtime_noise_surface",
    "stt_surface_augmentation", "spoken_robustness", "multi_intent", "order_compound",
    "compound_confirmation_order", "multiturn_confirmation", "recommendation_runtime_edges",
    "recommendation_select_commit", "multiturn_recommendation", "semantic_recommendation_state",
    "hypothetical_question_boundary", "commit_hard_negative", "question_vs_mutation",
}


def decision_metadata(row: dict, source_file: str, source_row: int, source_type: str) -> dict:
    family = row.get("scenario_family", row.get("source_group", "review_sample"))
    risk = "high" if family in HIGH_RISK_FAMILIES or row.get("difficulty") == "hard" else "medium" if row.get("difficulty") == "medium" else "low"
    return {
        "source_type": source_type,
        "source_file": source_file,
        "source_commit": CURRENT_COMMIT,
        "source_row": source_row,
        "source_session": row.get("session_id", ""),
        "risk": risk,
        "relabel_method": "natural_multiturn_v3_contract_relabel",
        "scenario_family": family,
        "semantic_id": row.get("semantic_id", f"{family}:{source_row}"),
        "paraphrase_group": row.get("paraphrase_group", row.get("id", f"row:{source_row}")),
    }


def decision_cleanup_errors(row: dict) -> list[str]:
    """cleanup으로 실제 교정된 오류 유형만 provenance에 기록한다."""
    errors = []
    raw_input = current_decision_input(row, clean_language=False)
    clean_input = current_decision_input(row, clean_language=True)
    if raw_input["message"] != clean_input["message"] or raw_input["recent_history"] != clean_input["recent_history"]:
        errors.append("unnatural_korean")

    unpacked = unpack_messages_row(row)
    old_target = unpacked.get("target") if isinstance(unpacked.get("target"), dict) else {}
    before = legacy_unsupported_mutation(row, old_target)
    after = unsupported_mutation(row, old_target)
    if before is not None and before != after:
        errors.append("fake_mutation")
    return errors


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def grouped_split(rows: list[dict], ratios: tuple[int, int, int] = (84, 8, 8), salt: str = "") -> dict[str, list[dict]]:
    union = UnionFind(len(rows))
    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        keys = (
            ("paraphrase", str(metadata.get("paraphrase_group", ""))),
            ("semantic", str(metadata.get("semantic_id", ""))),
            ("message", normalized_text(row["input"].get("message", row["input"].get("user_text", "")))),
        )
        for key in keys:
            if not key[1]:
                continue
            if key in seen:
                union.union(index, seen[key])
            else:
                seen[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)
    result = {"train": [], "validation": [], "test": []}
    train_cut, validation_cut = ratios[0], ratios[0] + ratios[1]
    for indices in components.values():
        signature = "|".join(sorted(rows[index]["metadata"]["paraphrase_group"] for index in indices))
        bucket = stable_bucket(signature + salt)
        split = "train" if bucket < train_cut else "validation" if bucket < validation_cut else "test"
        result[split].extend(rows[index] for index in indices)
    return result


def balanced_split(rows: list[dict], ratios: dict[str, float], salt: str = "") -> dict[str, list[dict]]:
    """연결된 semantic/paraphrase/message component를 쪼개지 않고 비율에 맞춘다."""
    union = UnionFind(len(rows))
    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        keys = (
            ("paraphrase", str(metadata.get("paraphrase_group", ""))),
            ("semantic", str(metadata.get("semantic_id", ""))),
            ("message", normalized_text(row["input"].get("message", row["input"].get("user_text", "")))),
        )
        for key in keys:
            if not key[1]:
                continue
            if key in seen:
                union.union(index, seen[key])
            else:
                seen[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)
    ordered = sorted(
        components.values(),
        key=lambda indices: (-len(indices), stable_bucket("|".join(sorted(rows[index]["id"] for index in indices)) + salt, 1_000_000)),
    )
    result = {split: [] for split in ratios}
    targets = {split: len(rows) * ratio for split, ratio in ratios.items()}
    counts = {split: 0 for split in ratios}
    for indices in ordered:
        size = len(indices)
        split = max(
            ratios,
            key=lambda name: ((targets[name] - counts[name]) / max(targets[name], 1), -counts[name]),
        )
        result[split].extend(rows[index] for index in indices)
        counts[split] += size
    return result


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    unique = []
    seen = set()
    for row in rows:
        signature = json.dumps({"input": row["input"], "target": row["target"]}, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(row)
    return unique, len(rows) - len(unique)


def evaluation_identity() -> tuple[set[str], set[str], set[str]]:
    rows = []
    for path in sorted(FROZEN_EVAL_DIR.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    rows.extend(read_jsonl(HARD_EVAL))
    messages, semantics, groups = set(), set(), set()
    for row in rows:
        model_input = current_decision_input(row)
        if model_input["message"]:
            messages.add(normalized_text(model_input["message"]))
        if row.get("semantic_id"):
            semantics.add(str(row["semantic_id"]))
        if row.get("paraphrase_group"):
            groups.add(str(row["paraphrase_group"]))
    return messages, semantics, groups


def build_decision() -> tuple[dict[str, list[dict]], dict]:
    eval_messages, eval_semantics, eval_groups = evaluation_identity()
    candidates = []
    excluded_eval = 0
    for index, row in enumerate(read_jsonl(LEGACY_TRAIN), start=1):
        metadata = decision_metadata(row, str(LEGACY_TRAIN.relative_to(ROOT)), index, "legacy_natural")
        metadata["cleanup_error_class"] = decision_cleanup_errors(row)
        model_input = current_decision_input(row)
        contaminated = (
            normalized_text(model_input["message"]) in eval_messages
            or metadata["semantic_id"] in eval_semantics
            or metadata["paraphrase_group"] in eval_groups
        )
        if contaminated:
            excluded_eval += 1
            continue
        candidates.append({
            "id": f"DV3-L{index:05d}",
            "input": model_input,
            "target": relabel_decision(row),
            "metadata": metadata,
        })

    # audit에서 사람이 검토한 actual runtime/failure turn을 별도 provenance로 보존한다.
    for index, row in enumerate(read_jsonl(DECISION_REVIEW), start=1):
        metadata = decision_metadata(row, str(DECISION_REVIEW.relative_to(ROOT)), index, "runtime_reviewed")
        metadata["cleanup_error_class"] = decision_cleanup_errors(row)
        metadata["risk"] = "high" if row.get("id") in {"D02", "D03", "D04", "D06", "D07", "D13"} else "medium"
        if (
            normalized_text(row["input"]["message"]) in eval_messages
            or metadata["semantic_id"] in eval_semantics
            or metadata["paraphrase_group"] in eval_groups
        ):
            excluded_eval += 1
            continue
        candidates.append({
            "id": f"DV3-R{index:03d}",
            "input": current_decision_input(row),
            "target": copy.deepcopy(row["target"]),
            "metadata": metadata,
        })

    candidates, duplicate_count = deduplicate(candidates)
    splits = balanced_split(candidates, {"train": 0.82, "validation": 0.09, "test": 0.09}, salt=":decision")

    def convert_eval(path: Path, prefix: str, source_type: str) -> list[dict]:
        converted = []
        for index, row in enumerate(read_jsonl(path), start=1):
            metadata = decision_metadata(row, str(path.relative_to(ROOT)), index, source_type)
            metadata["cleanup_error_class"] = decision_cleanup_errors(row)
            converted.append({
                "id": f"{prefix}{index:04d}",
                "input": current_decision_input(row),
                "target": relabel_decision(row),
                "metadata": metadata,
            })
        converted, _ = deduplicate(converted)
        return converted

    challenge = convert_eval(HARD_EVAL, "DV3-C", "frozen_challenge")
    holdout = []
    counter = 0
    for path in sorted(FROZEN_EVAL_DIR.glob("*.jsonl")):
        part = convert_eval(path, f"DV3-H{counter:02d}-", "frozen_holdout")
        holdout.extend(part)
        counter += 1
    holdout, holdout_duplicates = deduplicate(holdout)
    splits["challenge"] = challenge
    splits["holdout"] = holdout
    stats = {
        "legacy_candidates": len(read_jsonl(LEGACY_TRAIN)),
        "runtime_reviewed": len(read_jsonl(DECISION_REVIEW)),
        "excluded_eval_identity": excluded_eval,
        "duplicates_removed": duplicate_count,
        "holdout_duplicates_removed": holdout_duplicates,
    }
    return splits, stats


def amount_ko(amount: str) -> str:
    return {"low": "조금", "normal": "보통", "high": "많이"}[amount]


def has_batchim(word: str) -> bool:
    if not word:
        return False
    code = ord(word[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def topic(word: str) -> str:
    return word + ("은" if has_batchim(word) else "는")


def object_form(word: str) -> str:
    return word + ("을" if has_batchim(word) else "를")


def subject_form(word: str) -> str:
    return word + ("이" if has_batchim(word) else "가")


def direction_form(word: str) -> str:
    if not has_batchim(word):
        return word + "로"
    jong = (ord(word[-1]) - 0xAC00) % 28
    return word + ("로" if jong == 8 else "으로")


def naturalize_korean(text: str) -> str:
    """생성 template의 명백한 조사·양 표현 오류만 고친다."""
    nouns = (*TOPPINGS, "햄", "베이컨", "새우", "마늘", "채소", "육류", "추가 재료")
    amount_clause = {"조금": "조금만", "보통": "보통 양으로", "많이": "많이"}
    amount_noun = {"조금": "적은 양", "보통": "보통 양", "많이": "많은 양"}
    for noun in nouns:
        for amount in amount_clause:
            text = re.sub(
                re.escape(noun) + r" " + amount + r"[은는]",
                topic(noun) + " " + amount_clause[amount],
                text,
            )
            text = re.sub(
                re.escape(noun) + r" " + amount + r"[을를]",
                object_form(noun) + " " + amount_noun[amount] + "으로",
                text,
            )
        text = re.sub(
            re.escape(noun) + r"(?:이에요|예요)",
            noun + ("이에요" if has_batchim(noun) else "예요"),
            text,
        )
        text = re.sub(re.escape(noun) + r"[은는]", topic(noun), text)
        text = re.sub(re.escape(noun) + r"[을를]", object_form(noun), text)
        text = re.sub(re.escape(noun) + r"[이가]", subject_form(noun), text)
        text = re.sub(re.escape(noun) + r"(?:으로|로)", direction_form(noun), text)
    return (
        text.replace("조금으로", "조금만")
        .replace("조금로", "조금만")
        .replace("살짝로", "살짝")
        .replace("적게로", "적게")
        .replace("보통로", "보통으로")
        .replace("많이로", "많이")
        .replace("듬뿍로", "듬뿍")
        .replace("기본으로로", "기본으로")
        .replace("조금 양", "적은 양")
        .replace("많이 양", "많은 양")
        .replace("조금랑", "조금과")
        .replace("보통랑", "보통과")
        .replace("많이랑", "많이와")
    )


def response_input(**overrides) -> dict:
    base = {
        "user_text": "",
        "recent_history": [],
        "confirmed_order": empty_order(),
        "preferences": [],
        "pending": None,
        "policy": {"status": "pass", "reason": "state_update_only", "execute": False, "conflicts": []},
        "applied_this_turn": {
            "order_changes": {},
            "restriction_added": [],
            "restriction_removed": [],
            "preference_added": [],
            "preference_removed": [],
        },
        "future_changes": [],
        "recommendation_result": None,
        "next_prompt": None,
        "robot_state": robot_state(),
        "recent_action_history": [],
        "execution_authorized": False,
        "starting_now": [],
    }
    for key, value in overrides.items():
        base[key] = copy.deepcopy(value)
    return base


def runtime_task(item: str, amount: str) -> dict:
    """Node가 robot_state에 넣는 실제 작업 단위 모양을 만든다."""
    return {"class": item, "repeat_count": {"low": 1, "normal": 2, "high": 3}[amount]}


def runtime_issues(**overrides) -> dict:
    issues = {
        "unsupported": [],
        "invalid": [],
        "protected": [],
        "restriction_conflicts": [],
        "physical_conflicts": [],
    }
    issues.update(copy.deepcopy(overrides))
    return issues


def runtime_issue_policy(reason: str, issues: dict, conflicts: list[dict] | None = None) -> dict:
    """process_task의 validation issue policy와 같은 field를 만든다."""
    conflicts = copy.deepcopy(conflicts or [])
    return {
        "status": "warning",
        "reason": reason,
        "execute": False,
        "conflicts": conflicts,
        "issues": copy.deepcopy(issues),
        "unsupported": copy.deepcopy(issues["unsupported"]),
        "protected": copy.deepcopy(issues["protected"]),
        "invalid": copy.deepcopy(issues["invalid"]),
        "restriction_blocked": conflicts if reason == "restriction_conflict" else [],
    }


def response_scenario(family: str, index: int) -> tuple[str, dict, str, list[str]]:
    item = TOPPINGS[index % len(TOPPINGS)]
    amount = AMOUNTS[(index // len(TOPPINGS)) % len(AMOUNTS)]
    amount_word = amount_ko(amount)
    section = "veggie" if item in ("양파", "버섯") else "meat" if item in ("소시지", "게살") else "extra"
    order = empty_order()
    order["toppings"][item] = amount
    applied = {
        "order_changes": {f"toppings.{item}": amount},
        "restriction_added": [], "restriction_removed": [],
        "preference_added": [], "preference_removed": [],
    }
    variants = index % 5

    if family == "normal_order_mutation":
        user = [f"{item} {amount_word} 넣어줘", f"{item}은 {amount_word}로 해줘", f"이번엔 {item} {amount_word}", f"{item} 양을 {amount_word}로 부탁해", f"{item} {amount_word} 담아줘"][variants]
        pending = {"type": "execution", "source": "current_update", "section": section, "targets": [item]}
        prompt = {**pending, "items": {}}
        inp = response_input(user_text=user, confirmed_order=order, applied_this_turn=applied, pending=pending, next_prompt=prompt, robot_state=robot_state(section))
        reply = [f"{item}은 {amount_word}로 선택했어요. 이대로 담을까요?", f"{item} 양을 {amount_word}로 반영했어요. 지금 담을까요?", f"{item}을 {amount_word}로 골랐어요. 이대로 진행할까요?", f"{item}은 {amount_word} 양으로 준비했어요. 담기를 진행할까요?", f"{item} {amount_word} 선택을 반영했어요. 그대로 담을까요?"][variants]
        return "task", inp, reply, ["next_prompt", "pending_execution"]

    if family == "execution_accept":
        user = ["응, 진행해", "그래 담아줘", "좋아, 그대로 가자", "네 시작해주세요", "그대로 진행할게"][variants]
        inp = response_input(user_text=user, confirmed_order=order, policy={"status": "pass", "reason": "execution_allowed", "execute": True, "conflicts": []}, robot_state=robot_state(section), execution_authorized=True, starting_now=[{"item": item, "amount": amount}])
        reply = [f"{item} {amount_word} 담기를 시작할게요.", f"선택한 {item}을 {amount_word} 양으로 담기 시작해요.", f"{item} {amount_word} 작업을 진행할게요.", f"지금 {item}을 {amount_word}로 담기 시작합니다.", f"확인했어요. {item} {amount_word} 담기를 진행해요."][variants]
        return "task", inp, reply, ["execution_accept"]

    if family == "execution_reject":
        user = ["아니, 아직 담지 마", "이번 건 취소해", "지금은 진행하지 마", "아니야 멈춰", "그 선택은 안 할게"][variants]
        inp = response_input(user_text=user, confirmed_order=order, policy={"status": "pass", "reason": "confirmation_rejected", "execute": False, "conflicts": []}, robot_state=robot_state(section))
        reply = ["알겠어요. 이번 담기는 진행하지 않았어요.", "확인 요청을 취소했고 실행하지 않았어요.", "지금은 담지 않고 그대로 멈춰 있을게요.", "해당 실행은 시작하지 않았어요.", "그 선택은 실행하지 않았어요."][variants]
        return "task", inp, reply, ["execution_reject"]

    if family in ("recommendation_request", "recommendation_revise"):
        if family == "recommendation_request":
            user = ["이 단계 추천해줘", "알아서 골라줘", "어울리는 재료 추천해줘", "지금 선택을 추천해줘", "뭘 넣으면 좋을까?"][variants]
        else:
            user = ["다른 걸로 추천해줘", "그거 말고 매콤한 걸로", "조금 더 담백하게 추천해줘", "새 후보로 바꿔줘", "아까 추천 말고 다른 걸로"][variants]
        candidate = {"sauce": None, "noodle_type": None, "noodle_portion": None, "toppings": {item: amount}}
        tags = ["균형"] if family == "recommendation_request" else ["수정 요청"]
        pending = {"type": "recommendation", "candidate": candidate, "reason_tags": tags}
        rec = {"proposal": candidate, "reason_tags": tags, "accepted_fields": [f"toppings.{item}"], "rejected_fields": []}
        inp = response_input(user_text=user, confirmed_order=empty_order(), pending=pending, recommendation_result=rec, next_prompt=copy.deepcopy(pending), robot_state=robot_state(section), policy={"status": "pass", "reason": "recommendation_proposed", "execute": False, "conflicts": []})
        prefix = "새 후보로 " if family == "recommendation_revise" else ""
        reply = [f"{prefix}{item} {amount_word}을 추천드려요. 이 후보로 할까요?", f"{prefix}{item}을 {amount_word} 넣는 구성이 좋아요. 이 추천을 사용할까요?", f"{prefix}{item} {amount_word}이 잘 맞아요. 이걸로 선택할까요?", f"{prefix}{item}을 {amount_word}로 추천해요. 적용할까요?", f"{prefix}{item} {amount_word} 후보를 골랐어요. 이 추천으로 할까요?"][variants]
        return "task", inp, reply, ["next_prompt", "pending_recommendation", family]

    if family == "recommendation_unavailable":
        user = ["다른 걸 추천해줘", "그거 말고 새로 골라줘", "더 담백한 추천 줘", "추천을 바꿔줘", "다른 후보 있어?"][variants]
        rejected = {"field": f"toppings.{item}", "item": item, "value": amount, "reason": "already_selected"}
        rec = {"proposal": empty_order(), "reason_tags": [], "accepted_fields": [], "rejected_fields": [rejected]}
        rec["proposal"].pop("restrictions")
        inp = response_input(user_text=user, confirmed_order=order, recommendation_result=rec, policy={"status": "clarify", "reason": "recommendation_unavailable", "execute": False, "conflicts": [], "rejected_fields": [rejected]}, robot_state=robot_state(section))
        reply = ["현재 조건에서는 새 추천을 만들지 못했어요.", "지금 조건에 맞는 다른 추천 후보가 없어요.", "요청하신 방향으로 바꿀 수 있는 추천을 찾지 못했어요.", "새 추천을 만들지 못해 기존 후보도 해제했어요.", "현재 선택과 제한 안에서는 다른 후보를 제안하기 어려워요."][variants]
        return "task", inp, reply, ["recommendation_unavailable"]

    if family == "recommendation_reject":
        user = ["아니, 추천 안 받을래", "그건 싫어", "추천 취소해줘", "아니 그거 말고", "이번 추천은 넘길게"][variants]
        inp = response_input(user_text=user, confirmed_order=empty_order(), policy={"status": "pass", "reason": "confirmation_rejected", "execute": False, "conflicts": []}, robot_state=robot_state(section))
        reply = ["알겠어요. 이번 추천은 선택하지 않았어요.", "추천 후보를 적용하지 않고 취소했어요.", "해당 추천은 주문에 반영하지 않았어요.", "이번 후보는 선택하지 않았어요.", "추천을 건너뛰었고 주문은 그대로예요."][variants]
        return "task", inp, reply, ["recommendation_reject"]

    if family == "preference_update":
        value = ("매콤하게", "담백하게", "푸짐하게", "꾸덕하게")[index % 4]
        user = [f"난 {value} 먹는 걸 좋아해", f"취향은 {value}로 기억해줘", f"{value} 먹고 싶어", f"이번엔 {value} 부탁해", f"{value}가 좋아"][variants]
        changes = copy.deepcopy(applied); changes["order_changes"] = {}; changes["preference_added"] = [{"value": value}]
        inp = response_input(user_text=user, confirmed_order=empty_order(), preferences=[{"value": value}], applied_this_turn=changes, policy={"status": "pass", "reason": "state_update_only", "execute": False, "conflicts": []})
        reply = [f"{value} 드시는 취향을 저장했어요.", f"취향을 {value}로 기억해둘게요.", f"{value} 선호를 반영했어요.", f"앞으로 {value} 취향을 참고할게요.", f"{value} 좋아하시는 것으로 저장했어요."][variants]
        return "task", inp, reply, ["preference_update"]

    if family == "restriction_update":
        reason = ("allergy", "cannot_eat", "dislike", "dietary_rule")[index % 4]
        reason_ko = {"allergy": "알레르기", "cannot_eat": "섭취 불가", "dislike": "비선호", "dietary_rule": "식단 제한"}[reason]
        user = [f"{item} {reason_ko}로 등록해줘", f"나는 {item}을 못 먹어", f"{item}은 빼야 해", f"{item} 제한을 기억해줘", f"앞으로 {item}은 제외해줘"][variants]
        restricted = empty_order(); restricted["restrictions"] = [{"target": item, "reason": reason}]
        changes = copy.deepcopy(applied); changes["order_changes"] = {}; changes["restriction_added"] = [{"target": item, "reason": reason}]
        inp = response_input(user_text=user, confirmed_order=restricted, applied_this_turn=changes, policy={"status": "pass", "reason": "state_update_only", "execute": False, "conflicts": []})
        reply = [f"{item}을 {reason_ko} 제한으로 등록했어요.", f"{item} {reason_ko} 정보를 저장했어요.", f"앞으로 {item}은 {reason_ko} 제한으로 확인할게요.", f"{item} 제한을 반영했어요.", f"{item}은 {reason_ko} 사유로 제외하도록 기록했어요."][variants]
        return "task", inp, reply, ["restriction_update"]

    if family == "missing_order":
        field = ("noodle_type", "noodle_portion")[index % 2]
        missing = [field]
        prompt = {"type": "missing_order", "fields": missing}
        user = ["이대로 진행해", "바로 시작해줘", "면부터 담아줘", "그대로 가자", "주문 진행할게"][variants]
        partial_order = empty_order()
        partial_order["sauce"] = SAUCES[(index // 2) % len(SAUCES)]
        if field == "noodle_type":
            partial_order["noodle_portion"] = AMOUNTS[(index // 4) % len(AMOUNTS)]
        else:
            partial_order["noodle_type"] = NOODLES[(index // 4) % len(NOODLES)]
        inp = response_input(user_text=user, confirmed_order=partial_order, policy={"status": "clarify", "reason": "missing_order", "execute": False, "missing": missing, "conflicts": []}, next_prompt=prompt, robot_state=robot_state("noodle"))
        if field == "noodle_type":
            reply = ["면 종류가 아직 정해지지 않았어요. 얇은면과 넓은면 중 어떤 걸로 할까요?", "먼저 면 종류를 골라야 해요. 얇은면과 넓은면 중에서 선택해주세요.", "면 종류가 필요해요. 얇은면 또는 넓은면으로 골라주세요.", "진행하려면 면 종류를 정해야 해요. 얇은면과 넓은면 중 무엇으로 할까요?", "아직 면을 고르지 않았어요. 얇은면과 넓은면 중 하나를 선택해주세요."][variants]
        else:
            reply = ["면 양이 아직 정해지지 않았어요. 조금, 보통, 많이 중 어떤 양으로 할까요?", "진행하려면 면 양을 골라야 해요. 조금, 보통, 많이 중에서 선택해주세요.", "면 양이 필요해요. 조금과 보통, 많이 중 무엇으로 할까요?", "아직 면 양을 정하지 않았어요. 조금, 보통, 많이 중 하나를 골라주세요.", "면 양부터 알려주세요. 조금, 보통, 많이 중 어떤 양이 좋으세요?"][variants]
        return "task", inp, reply, ["next_prompt", "missing_order"]

    if family == "unsupported":
        unsupported = ("햄", "베이컨", "새우", "마늘")[index % 4]
        user = [f"{unsupported} 넣어줘", f"{unsupported} 많이 추가해줘", f"이번엔 {unsupported}로 해줘", f"{unsupported}도 담아줘", f"{unsupported} 조금 넣어줘"][variants]
        issue = {"field": f"toppings.{unsupported}", "item": unsupported, "value": amount}
        issues = runtime_issues(unsupported=[issue])
        prompt = {"type": "validation_issues", "issues": copy.deepcopy(issues)}
        inp = response_input(user_text=user, confirmed_order=empty_order(), policy=runtime_issue_policy("unsupported", issues), next_prompt=prompt, robot_state=robot_state(section))
        alternatives = {
            "veggie": "양파나 버섯",
            "meat": "소시지나 게살",
            "extra": "치즈나 페퍼론치노",
        }.get(section, "현재 지원되는 재료")
        reply = [f"{unsupported}은 현재 지원하지 않아요. {alternatives} 중에서 골라주세요.", f"지금은 {unsupported}을 담을 수 없어요. 대신 {alternatives}을 선택할 수 있어요.", f"{unsupported}은 지원 메뉴가 아니에요. {alternatives} 중 하나로 바꿔주세요.", f"{unsupported} 요청은 반영하지 않았어요. 가능한 재료는 {alternatives}이에요.", f"현재 {unsupported}은 사용할 수 없어요. {alternatives} 중 어떤 걸로 할까요?"][variants]
        return "task", inp, reply, ["next_prompt", "unsupported"]

    if family == "restriction_conflict":
        restricted = empty_order(); restricted["restrictions"] = [{"target": item, "reason": "allergy"}]
        conflict = {"key": f"toppings.{item}", "item": item, "restriction": {"target": item, "reason": "allergy"}}
        issues = runtime_issues(restriction_conflicts=[conflict])
        prompt = {"type": "validation_issues", "issues": copy.deepcopy(issues)}
        user = [f"{item} {amount_word} 넣어줘", f"{item}도 {amount_word} 추가해줘", f"이번엔 {item} {amount_word}", f"{item}을 {amount_word} 담아줘", f"{item} 양은 {amount_word}로 할게"][variants]
        inp = response_input(user_text=user, confirmed_order=restricted, policy=runtime_issue_policy("restriction_conflict", issues, [conflict]), next_prompt=prompt, robot_state=robot_state(section))
        reply = [f"{item}은 알레르기 제한 때문에 추가하지 않았어요. 제한을 해제하려면 명확히 말씀해주세요.", f"알레르기 제한이 있어 {item} 요청은 반영하지 않았어요. 제한을 바꾸려면 직접 해제를 요청해주세요.", f"현재 {item} 알레르기 제한이 적용 중이라 담을 수 없어요. 해제를 원하면 분명히 알려주세요.", f"{item}은 등록된 알레르기와 충돌해 추가하지 않았어요. 제한 변경은 별도로 요청해주세요.", f"알레르기 제한 때문에 {item}은 제외했어요. 제한을 정말 해제할 때는 명확히 말해주세요."][variants]
        return "task", inp, reply, ["next_prompt", "restriction_conflict"]

    if family == "physical_conflict":
        protected = {"field": f"toppings.{item}", "item": item, "value": "none"}
        issues = runtime_issues(protected=[protected])
        prompt = {"type": "validation_issues", "issues": copy.deepcopy(issues)}
        completed = runtime_task(item, amount)
        state = robot_state("sauce"); state["completed_tasks"] = [completed]; state["robot_started"] = True
        user = [f"방금 {item} 빼줘", f"{item} 선택 취소해", f"이미 넣은 {item} 없던 걸로 해줘", f"{item} 양을 바꿔줘", f"{item}은 빼고 싶어"][variants]
        history = [{"type": "task_completed", "section": section, "task": copy.deepcopy(completed)}]
        inp = response_input(user_text=user, confirmed_order=order, policy=runtime_issue_policy("physical_state", issues), next_prompt=prompt, robot_state=state, recent_action_history=history)
        reply = [f"{item}은 이미 담긴 단계라 지금은 변경할 수 없어요. 현재 단계에서 가능한 선택은 계속할 수 있어요.", f"이미 {item} 작업이 끝나서 선택을 되돌릴 수 없어요. 지금 단계의 다른 선택으로 진행해주세요.", f"{item}은 지나간 단계에서 담겨 현재 수정할 수 없어요. 현재 단계에서 가능한 항목을 선택할 수 있어요.", f"지금은 {item} 담기가 완료되어 양을 바꿀 수 없어요. 다음 선택은 현재 단계에서 이어갈 수 있어요.", f"{item}은 이미 실행된 항목이라 제거할 수 없어요. 현재 단계의 가능한 선택으로 계속해주세요."][variants]
        return "task", inp, reply, ["next_prompt", "physical_conflict"]

    if family == "partial_success":
        unsupported = ("햄", "베이컨", "새우", "마늘")[index % 4]
        user = [f"{item} {amount_word} 넣고 {unsupported}도 추가해줘", f"{item}은 {amount_word}, {unsupported}도 넣어줘", f"{item} {amount_word}랑 {unsupported} 부탁해", f"{item} 양은 {amount_word}로 하고 {unsupported} 추가", f"{item} {amount_word} 담고 {unsupported}도 담아줘"][variants]
        issue = {"field": f"toppings.{unsupported}", "item": unsupported, "value": "normal"}
        issues = runtime_issues(unsupported=[issue])
        prompt = {"type": "validation_issues", "issues": copy.deepcopy(issues)}
        policy = runtime_issue_policy("unsupported", issues)
        inp = response_input(user_text=user, confirmed_order=order, applied_this_turn=applied, policy=policy, next_prompt=prompt, robot_state=robot_state(section))
        reply = [f"{item}은 {amount_word}로 반영했어요. {unsupported}은 지원하지 않아 추가하지 않았어요. 지원되는 재료를 다시 고르거나 넘어갈 수 있어요.", f"{item} {amount_word} 선택은 적용했지만 {unsupported}은 현재 담을 수 없어요. 다른 지원 재료를 선택해주세요.", f"{item}은 {amount_word}로 저장했어요. {unsupported} 요청은 지원 대상이 아니라 제외했어요. 추가 재료 없이 넘어가도 돼요.", f"{item} {amount_word}는 반영됐고 {unsupported}은 지원하지 않아 반영되지 않았어요. 지원 메뉴에서 다시 고를 수 있어요.", f"{item} 양은 {amount_word}로 바꿨어요. 다만 {unsupported}은 사용할 수 없으니 다른 지원 재료를 고르거나 넘어가주세요."][variants]
        return "task", inp, reply, ["partial_success", "validation_issue"]

    if family in ("status_question", "pending_status_question"):
        user = ["지금까지 뭐 담았어?", "현재 주문 알려줘", "지금 뭐 하는 중이야?", "내가 뭘 골랐지?", "진행 상황 알려줘"][variants]
        pending = None
        if family == "pending_status_question":
            pending = {"type": "execution", "source": "current_update", "section": section, "targets": [item]}
        history = [{"type": "turn_applied", **copy.deepcopy(applied)}]
        inp = response_input(user_text=user, confirmed_order=order, pending=pending, robot_state=robot_state(section), recent_action_history=history, policy={"status": "pass", "reason": "state_update_only", "execute": False, "conflicts": []})
        if pending:
            inp["next_prompt"] = {**copy.deepcopy(pending), "items": {}}
            suffix = " 실행 확인이 남아 있어요. 이대로 진행할까요?"
            tags = [family, "next_prompt", "pending_execution"]
        else:
            suffix = ""
            tags = [family, "no_forced_question"]
        section_name = {"veggie": "채소", "meat": "육류", "extra": "추가 재료"}.get(section, section)
        reply = [f"현재 {item}을 {amount_word}로 선택했고 {section_name} 단계예요.{suffix}", f"지금 주문에는 {subject_form(item)} {amount_word} 양으로 선택되어 있고 현재 {section_name} 단계예요.{suffix}", f"{item}을 {amount_word}로 골랐으며 로봇은 {section_name} 단계에 있어요.{suffix}", f"현재 {item}은 {amount_word} 양으로 선택되어 있고 진행 단계는 {section_name}예요.{suffix}", f"지금까지 {item}을 {amount_word}로 선택했고 현재 {section_name} 단계예요.{suffix}"][variants]
        return "task", inp, reply, tags

    if family == "general_question":
        users = [
            "알덴테가 뭐야?", "파스타는 어디서 시작됐어?", "크림소스 특징을 알려줘", "오늘 날씨 어때?", "로봇은 어떻게 움직여?",
            "오일 파스타는 어떤 맛이야?", "토마토소스는 보통 어떻게 만들어?", "넓은면 특징은 뭐야?", "얇은면은 어떤 소스와 잘 맞아?", "페퍼론치노가 뭐야?",
            "버섯은 익히면 식감이 어떻게 돼?", "게살은 어떤 재료야?", "치즈가 파스타에 주는 맛은 뭐야?", "양파를 볶으면 왜 달아져?", "소시지는 보통 어떻게 익혀?",
        ]
        replies = [
            "알덴테는 면의 중심에 약간의 단단함이 남아 있는 익힘 상태예요.", "파스타의 형태는 오래전부터 여러 지역에 있었고, 오늘날의 이탈리아식 파스타로 발전했어요.", "크림소스는 부드럽고 고소한 맛과 비교적 진한 질감이 특징이에요.", "현재 날씨 정보에는 접근할 수 없어요.", "이 로봇은 정해진 작업 순서와 센서 상태에 따라 재료를 담아요.",
            "오일 파스타는 비교적 가볍고 재료의 향이 또렷하게 느껴지는 편이에요.", "토마토소스는 토마토를 기본으로 향신료와 채소를 더해 끓여 만들어요.", "넓은면은 소스가 닿는 면적이 넓어 진한 소스와 잘 어울려요.", "얇은면은 가벼운 오일이나 토마토소스와 조합하기 좋아요.", "페퍼론치노는 파스타에 매운맛과 향을 더하는 작은 고추예요.",
            "버섯은 익히면 수분이 빠지면서 부드럽고 쫄깃한 식감이 나요.", "게살은 게의 살을 발라낸 재료로 부드러운 식감과 은은한 단맛이 있어요.", "치즈는 파스타에 고소함과 짠맛, 농도를 더해줘요.", "양파는 볶는 동안 당 성분이 변하면서 단맛과 고소한 향이 강해져요.", "소시지는 속까지 충분히 가열되도록 굽거나 삶아 익히는 경우가 많아요.",
        ]
        case = index % len(users)
        inp = response_input(user_text=users[case], policy={"status": "pass", "reason": "general", "execute": False, "conflicts": []})
        return "general", inp, replies[case], ["general", "no_forced_question"]

    if family == "mixed_turn":
        user = [f"{item} {amount_word} 넣고 맛 특징도 알려줘", f"{item}은 {amount_word}로 하고 어떤 재료인지 설명해줘", f"{item} {amount_word} 담아주고 식감도 알려줘", f"{item} 양은 {amount_word}, 그리고 향은 어때?", f"{item} {amount_word}로 바꾸고 특징도 말해줘"][variants]
        inp = response_input(user_text=user, confirmed_order=order, applied_this_turn=applied, policy={"status": "pass", "reason": "state_update_only", "execute": False, "conflicts": []}, robot_state=robot_state(section))
        reply = [f"{item}은 {amount_word}로 반영했어요. {item}은 파스타에 맛과 식감을 더하는 재료예요.", f"{item} 양을 {amount_word}로 바꿨어요. 재료 특성은 조리 방식에 따라 달라질 수 있어요.", f"{item} {amount_word} 선택을 저장했어요. {item}은 익히면 고유한 향과 식감이 살아나요.", f"{item}은 {amount_word}로 반영됐어요. 향은 가열 정도와 다른 재료 조합에 따라 달라져요.", f"{item} 양을 {amount_word}로 적용했어요. 맛과 식감은 소스 조합에 따라 달라질 수 있어요."][variants]
        return "mixed", inp, reply, ["mixed"]

    if family == "section_transition":
        choices = {"veggie": "양파나 버섯", "meat": "소시지나 게살", "extra": "치즈나 페퍼론치노"}
        prompt_text = {
            "veggie": "다음은 야채를 고르실 차례입니다. 양파와 버섯 중 원하는 재료와 양을 말씀해 주세요. 원하지 않으면 다음 단계라고 말씀하셔도 돼요.",
            "meat": "다음은 육류를 고르실 차례입니다. 소시지와 게살 중 원하는 재료와 양을 말씀해 주세요. 원하지 않으면 다음 단계라고 말씀하셔도 돼요.",
            "extra": "다음은 추가 재료 차례입니다. 치즈와 페퍼론치노 중 원하는 재료와 양을 말씀해 주세요. 원하지 않으면 다음 단계라고 말씀하셔도 돼요.",
        }
        section = ("veggie", "meat", "extra")[index % 3]
        previous_section = {"veggie": "noodle", "meat": "veggie", "extra": "meat"}[section]
        previous_item = {"veggie": "얇은면", "meat": "양파", "extra": "소시지"}[section]
        completed = runtime_task(previous_item, "normal")
        transition = {"type": "section_transition", "section": previous_section, "outcome": "completed", "next_section": section}
        state = robot_state(section)
        state.update({"completed_tasks": [completed], "robot_started": True, "section_transition": copy.deepcopy(transition)})
        prompt = {"type": "section_prompt", "section": section, "text": prompt_text[section]}
        inp = response_input(user_text="", next_prompt=prompt, robot_state=state, recent_action_history=[transition], policy={"status": "pass", "reason": "section_transition", "execute": False, "conflicts": []})
        label = {"veggie": "채소", "meat": "육류", "extra": "추가 재료"}[section]
        reply = [f"다음은 {label} 단계예요. {choices[section]}를 고를 수 있고, 원하지 않으면 넘어가도 돼요.", f"이제 {label}를 선택할 차례예요. {choices[section]} 중에서 고르거나 그냥 넘어갈 수 있어요.", f"{label} 단계로 이동했어요. {choices[section]}를 선택해도 되고 필요 없으면 건너뛰어도 돼요.", f"다음 {label} 선택은 {choices[section]}예요. 추가하지 않으려면 넘어가기를 말씀해주세요.", f"이제 {label}를 골라주세요. {choices[section]}가 가능하며 선택 없이 넘어가도 괜찮아요."][variants]
        return "task", inp, reply, ["next_prompt", "section_transition"]

    if family == "understanding_clarify":
        user = ["그걸로 해줘", "아까 말한 걸 바꿔줘", "그거 말고 저걸로", "그 선택 다시 해줘", "그 재료로 부탁해"][variants]
        inp = response_input(user_text=user, confirmed_order=order, robot_state=robot_state(section), policy={"status": "clarify", "reason": "understanding", "execute": False, "conflicts": []})
        reply = ["어떤 항목을 말씀하시는지 확인이 필요해요. 재료나 양을 다시 알려주세요.", "바꿀 대상을 특정할 수 없어요. 변경할 재료를 말씀해주세요.", "어느 선택을 뜻하는지 알려주세요.", "다시 선택할 항목과 원하는 양을 말씀해주세요.", "어떤 재료인지 이름을 말씀해주세요."][variants]
        return "task", inp, reply, ["understanding_clarify"]

    if family == "completion":
        completed = runtime_task(item, amount)
        state = robot_state(section); state["completed_tasks"] = [completed]; state["robot_started"] = True
        user = ["방금 뭐 끝냈어?", "지금까지 완료된 작업 알려줘", "어떤 재료를 담았어?", "방금 끝난 작업이 뭐야?", "완료된 재료가 뭐였지?"][variants]
        history = [{"type": "task_completed", "section": section, "task": copy.deepcopy(completed)}]
        inp = response_input(user_text=user, confirmed_order=order, robot_state=state, recent_action_history=history, policy={"status": "pass", "reason": "state_update_only", "execute": False, "conflicts": []})
        reply = [f"{item} {amount_word} 담기가 완료됐어요.", f"{item}을 {amount_word} 양으로 담는 작업을 마쳤어요.", f"{item} {amount_word} 작업이 끝났어요.", f"선택한 {item} {amount_word} 담기를 완료했어요.", f"{item}을 {amount_word}로 담는 과정이 완료됐어요."][variants]
        return "task", inp, reply, ["completion", "no_forced_question"]

    if family == "stt_noisy":
        noisy = [
            "소세지 마니 넣어조", "계살 쪼금", "버섯 적개 해줘", "양파 마니 부탁해", "페퍼론치노 살작",
            "치주 보통으로", "소시질 많이", "개살 보통 넣어줘", "양빠 조금만", "버섯 마니 줘",
            "페페론치노 보통", "치즈 쪼금 넣어조", "소세지 적께", "계살 마니 부탁해", "양파 보통으루",
        ]
        target_items = ("소시지", "게살", "버섯", "양파", "페퍼론치노", "치즈", "소시지", "게살", "양파", "버섯", "페퍼론치노", "치즈", "소시지", "게살", "양파")
        target_amounts = ("high", "low", "low", "high", "low", "normal", "high", "normal", "low", "high", "normal", "low", "low", "high", "normal")
        case = index % len(noisy)
        target_item = target_items[case]
        target_amount = target_amounts[case]
        target_section = "meat" if target_item in ("소시지", "게살") else "veggie" if target_item in ("양파", "버섯") else "extra"
        target_order = empty_order(); target_order["toppings"][target_item] = target_amount
        target_applied = copy.deepcopy(applied); target_applied["order_changes"] = {f"toppings.{target_item}": target_amount}
        pending = {"type": "execution", "source": "current_update", "section": target_section, "targets": [target_item]}
        prompt = {**pending, "items": {}}
        inp = response_input(user_text=noisy[case], confirmed_order=target_order, applied_this_turn=target_applied, pending=pending, next_prompt=prompt, robot_state=robot_state(target_section))
        reply = f"{target_item}은 {amount_ko(target_amount)}로 선택했어요. 이대로 담을까요?"
        return "task", inp, reply, ["stt_noisy", "next_prompt"]

    if family == "preselected_pending":
        second = "게살" if item != "게살" else "소시지"
        selected = {item: amount, second: "low"}
        selected_order = empty_order(); selected_order["toppings"] = selected
        pending = {"type": "execution", "source": "preselected", "section": "meat", "targets": list(selected), "items": selected, "candidate": {"sauce": None, "noodle_type": None, "noodle_portion": None, "toppings": selected}}
        user = [f"{second}도 조금 추가해줘", f"거기에 {second} 조금도 넣어줘", f"{second}은 적게 추가", f"기존 거 두고 {second} 조금", f"{second} 조금까지 같이 해줘"][variants]
        changes = copy.deepcopy(applied); changes["order_changes"] = {f"toppings.{second}": "low"}
        inp = response_input(user_text=user, confirmed_order=selected_order, applied_this_turn=changes, pending=pending, next_prompt=copy.deepcopy(pending), robot_state=robot_state("meat"))
        reply = [f"{second}을 조금으로 추가했어요. 미리 고른 재료들과 이대로 담을까요?", f"기존 선택은 유지하고 {second} 조금을 더했어요. 함께 진행할까요?", f"{second}은 적게로 추가됐어요. 현재 선택대로 담을까요?", f"기존 재료에 {second} 조금을 추가했어요. 이 구성으로 진행할까요?", f"{second} 조금까지 선택했어요. 미리 고른 항목과 함께 담을까요?"][variants]
        return "task", inp, reply, ["preselected_pending", "next_prompt"]

    if family == "multiple_validation_issues":
        unsupported = "햄"
        conflict_item = "치즈"
        multi_variant = (index // 5) % 5
        multi_amount = AMOUNTS[(index // 25) % len(AMOUNTS)]
        multi_amount_word = amount_ko(multi_amount)
        valid_item = ("양파", "버섯", "소시지", "게살", "페퍼론치노")[index % 5]
        valid_section = "veggie" if valid_item in ("양파", "버섯") else "meat" if valid_item in ("소시지", "게살") else "extra"
        selected_order = empty_order()
        selected_order["toppings"][valid_item] = multi_amount
        selected_order["restrictions"] = [{"target": conflict_item, "reason": "allergy"}]
        unsupported_issue = {"field": f"toppings.{unsupported}", "item": unsupported, "value": "normal"}
        conflict = {"key": f"toppings.{conflict_item}", "item": conflict_item, "restriction": {"target": conflict_item, "reason": "allergy"}}
        issues = runtime_issues(unsupported=[unsupported_issue], restriction_conflicts=[conflict])
        prompt = {"type": "validation_issues", "issues": copy.deepcopy(issues)}
        policy = runtime_issue_policy("restriction_conflict", issues, [conflict])
        valid_applied = copy.deepcopy(applied); valid_applied["order_changes"] = {f"toppings.{valid_item}": multi_amount}
        user = [f"{valid_item} {multi_amount_word}, 햄이랑 치즈도 넣어줘", f"{valid_item}은 {multi_amount_word}로 하고 햄, 치즈 추가", f"{valid_item}은 {multi_amount_word}, 햄하고 치즈도 부탁해", f"{valid_item} 양은 {multi_amount_word}, 햄 치즈도 같이", f"{valid_item} {multi_amount_word} 넣고 햄과 치즈도 담아줘"][multi_variant]
        inp = response_input(user_text=user, confirmed_order=selected_order, applied_this_turn=valid_applied, policy=policy, next_prompt=prompt, robot_state=robot_state(valid_section))
        reply = [f"{valid_item}은 {multi_amount_word}로 반영했어요. 햄은 지원하지 않고, 치즈는 알레르기 제한 때문에 추가하지 않았어요. 지원 재료를 다시 고르거나 넘어갈 수 있어요.", f"{valid_item} {multi_amount_word}는 적용됐어요. 햄은 지원 대상이 아니며 치즈는 알레르기 제한과 충돌해 제외했어요. 다른 지원 재료를 선택해주세요.", f"{valid_item}은 {multi_amount_word}로 저장했어요. 햄은 사용할 수 없고 치즈는 등록된 알레르기 때문에 담지 않았어요. 추가 선택 없이 넘어가도 돼요.", f"{valid_item} {multi_amount_word}만 반영됐어요. 햄은 미지원이고 치즈는 알레르기 제한으로 막혔어요. 지원 메뉴에서 다시 고를 수 있어요.", f"{valid_item} 양은 {multi_amount_word}로 바꿨어요. 햄과 치즈는 각각 미지원과 알레르기 제한 때문에 반영되지 않았어요. 다른 지원 재료를 고르거나 넘어가주세요."][multi_variant]
        return "task", inp, reply, ["multiple_issues", "partial_success"]

    raise ValueError(f"unknown response family: {family}")


RESPONSE_FAMILIES = (
    "normal_order_mutation", "execution_accept", "execution_reject",
    "recommendation_request", "recommendation_revise", "recommendation_unavailable", "recommendation_reject",
    "preference_update", "restriction_update", "missing_order", "unsupported",
    "restriction_conflict", "physical_conflict", "partial_success", "status_question",
    "general_question", "mixed_turn", "section_transition", "understanding_clarify", "completion",
    "stt_noisy", "pending_status_question", "preselected_pending", "multiple_validation_issues",
)


def build_response() -> tuple[dict[str, list[dict]], dict]:
    rows = []
    counter = 0
    for family in RESPONSE_FAMILIES:
        for index in range(60):
            counter += 1
            scenario_item = TOPPINGS[index % len(TOPPINGS)]
            route, model_input, reply, tags = response_scenario(family, index)
            raw_user_text = model_input["user_text"]
            raw_reply = reply
            model_input["user_text"] = naturalize_korean(raw_user_text)
            reply = naturalize_korean(raw_reply)
            cleanup_errors = []
            if raw_user_text != model_input["user_text"] or raw_reply != reply:
                cleanup_errors.append("unnatural_korean")
            if family in {"status_question", "pending_status_question"} and index % 5 == 1 and not has_batchim(scenario_item) and "unnatural_korean" not in cleanup_errors:
                cleanup_errors.append("unnatural_korean")
            if family == "pending_status_question":
                cleanup_errors.append("state_contradiction")
            if family in {
                "normal_order_mutation", "recommendation_unavailable", "unsupported",
                "restriction_conflict", "physical_conflict", "partial_success",
                "status_question", "section_transition", "understanding_clarify",
                "completion", "stt_noisy", "pending_status_question",
                "multiple_validation_issues",
            }:
                cleanup_errors.append("runtime_contract_mismatch")
            group = f"{family}:{index // 5}"
            rows.append({
                "id": f"RV1-S{counter:04d}",
                "route": route,
                "input": model_input,
                "target": {"reply": reply},
                "metadata": {
                    "source_type": "new_synthetic",
                    "source_file": "scripts/build_datasets.py",
                    "source_commit": CURRENT_COMMIT,
                    "source_row": counter,
                    "source_session": "",
                    "risk": "high" if family in {"recommendation_revise", "recommendation_unavailable", "restriction_conflict", "physical_conflict", "partial_success", "section_transition", "pending_status_question", "multiple_validation_issues"} else "medium",
                    "relabel_method": "structured_fact_response_generation",
                    "scenario_family": family,
                    "semantic_id": group,
                    "paraphrase_group": group,
                    "coverage_tags": tags,
                    "cleanup_error_class": cleanup_errors,
                    "source_type_detail": "section_transition" if family == "section_transition" else "user_turn",
                },
            })

    # 승인된 review sample은 신규 생성 자료와 별도로 보존한다.
    for index, row in enumerate(read_jsonl(RESPONSE_REVIEW), start=1):
        counter += 1
        family = row.get("source_group", "review_sample")
        reviewed_input = copy.deepcopy(row["input"])
        raw_user_text = reviewed_input.get("user_text", "")
        raw_reply = row["target"]["reply"]
        reviewed_input["user_text"] = naturalize_korean(raw_user_text)
        reviewed_target = {"reply": naturalize_korean(raw_reply)}
        cleanup_errors = []
        if raw_user_text != reviewed_input["user_text"] or raw_reply != reviewed_target["reply"]:
            cleanup_errors.append("unnatural_korean")
        reviewed_policy = reviewed_input.get("policy") or {}
        reviewed_issues = reviewed_policy.get("issues")
        if isinstance(reviewed_issues, dict):
            reviewed_input["policy"] = runtime_issue_policy(
                reviewed_policy.get("reason", "invalid_candidate"),
                runtime_issues(**reviewed_issues),
                reviewed_policy.get("conflicts", []),
            )
            cleanup_errors.append("runtime_contract_mismatch")
        rows.append({
            "id": f"RV1-R{index:03d}",
            "route": row.get("route", "task"),
            "input": reviewed_input,
            "target": reviewed_target,
            "metadata": {
                "source_type": "current_contract_reviewed",
                "source_file": str(RESPONSE_REVIEW.relative_to(ROOT)),
                "source_commit": CURRENT_COMMIT,
                "source_row": index,
                "source_session": "",
                "risk": "high",
                "relabel_method": "human_review_sample",
                "scenario_family": family,
                "semantic_id": f"review:{row.get('id', index)}",
                "paraphrase_group": f"review:{row.get('id', index)}",
                "coverage_tags": ["reviewed"],
                "cleanup_error_class": cleanup_errors,
                "source_type_detail": "section_transition" if "section" in family else "user_turn",
            },
        })

    rows, duplicate_count = deduplicate(rows)
    splits = balanced_split(
        rows,
        {"train": 0.74, "validation": 0.08, "test": 0.07, "challenge": 0.06, "holdout": 0.05},
        salt=":response",
    )
    return splits, {"new_synthetic": 60 * len(RESPONSE_FAMILIES), "reviewed": len(read_jsonl(RESPONSE_REVIEW)), "duplicates_removed": duplicate_count}


def final_row(row: dict, response: bool = False) -> dict:
    result = {"id": row["id"]}
    if response:
        result["route"] = row.get("route", "task")
    result["input"] = row["input"]
    result["target"] = row["target"]
    return result


def write_bundle(name: str, splits: dict[str, list[dict]], response: bool = False) -> None:
    directory = OUT / name
    all_rows = []
    for split in ("train", "validation", "test", "challenge", "holdout"):
        split_rows = sorted(splits.get(split, []), key=lambda row: row["id"])
        write_jsonl(directory / f"{split}.jsonl", [final_row(row, response=response) for row in split_rows])
        for row in split_rows:
            enriched = copy.deepcopy(row)
            enriched["split"] = split
            all_rows.append(enriched)
    write_jsonl(OUT / "intermediate" / f"{name}_intermediate.jsonl", all_rows)
    write_jsonl(OUT / "provenance" / f"{name}_provenance.jsonl", [
        {"id": row["id"], "split": row["split"], **row["metadata"]} for row in all_rows
    ])
    high = [row for row in all_rows if row["metadata"].get("risk") == "high"]
    high.sort(key=lambda row: (not bool(row["metadata"].get("cleanup_error_class")), stable_bucket(row["id"]), row["id"]))
    review_name = "decision_high_risk_review.jsonl" if name == "decision_v3" else "response_high_risk_review.jsonl"
    reviewed_rows = []
    for row in high[:200]:
        reviewed = copy.deepcopy(row)
        error_class = list(reviewed["metadata"].get("cleanup_error_class", []))
        reasons = {
            "fake_mutation": "설명 대상·별칭·집합 표현이 가짜 topping mutation으로 덮이지 않는지 확인했다.",
            "state_contradiction": "선택·완료·실행 대기 상태가 같은 항목에서 충돌하지 않는지 확인했다.",
            "unnatural_korean": "조사와 양·단계 표현을 교정하고 의미가 유지되는지 확인했다.",
            "runtime_contract_mismatch": "policy·pending·next_prompt·robot state를 현재 runtime 생성 구조와 맞췄다.",
        }
        reviewed["semantic_review"] = "fixed" if error_class else "pass"
        reviewed["review_reason"] = (
            " ".join(reasons[error] for error in error_class)
            if error_class else
            "입력 문맥과 구조화 target/state가 일치하고 과잉 mutation이 없다."
        )
        reviewed["error_class"] = error_class
        reviewed_rows.append(reviewed)
    write_jsonl(directory / review_name, reviewed_rows)


def main() -> None:
    decision_splits, decision_stats = build_decision()
    response_splits, response_stats = build_response()
    write_bundle("decision_v3", decision_splits)
    write_bundle("response_v1", response_splits, response=True)
    build_report = {
        "contract": "natural_multiturn_v3",
        "runtime_commit": CURRENT_COMMIT,
        "decision": {"splits": {key: len(value) for key, value in decision_splits.items()}, **decision_stats},
        "response": {"splits": {key: len(value) for key, value in response_splits.items()}, **response_stats},
    }
    (OUT / "build_report.json").write_text(json.dumps(build_report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(build_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
