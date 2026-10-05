#!/usr/bin/env python3
"""Generate deterministic route-boundary train supplements and a sealed hard eval."""

from __future__ import annotations

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent

SAUCES = ("오일", "토마토", "크림")
NOODLES = ("얇은면", "넓은면")
TOPPINGS = ("양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노")
SUPPORTED = SAUCES + NOODLES + TOPPINGS + ("유제품", "갑각류", "육류", "비건")
UNSUPPORTED = (
    "햄", "베이컨", "살라미", "초리조", "스팸", "새우", "미트볼", "마늘",
    "브로콜리", "모짜렐라", "할라피뇨", "참치", "파프리카", "당근", "바질",
)
STYLES = ("short_spoken", "polite", "stt_like", "word_order", "multi_intent")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def empty_order() -> dict:
    return {
        "sauce": None,
        "noodle_type": None,
        "noodle_portion": None,
        "toppings": {},
        "restrictions": [],
    }


def empty_focus() -> dict:
    return {"current": None, "recent": []}


def base_input(
    message: str,
    *,
    order: dict | None = None,
    preferences: list[dict] | None = None,
    recommendation: dict | None = None,
    history: list[dict] | None = None,
    dialogue_focus: dict | None = None,
    reference_context: dict | None = None,
    robot_state: dict | None = None,
) -> dict:
    return {
        "order": copy.deepcopy(order if order is not None else empty_order()),
        "preferences": copy.deepcopy(preferences if preferences is not None else []),
        "recommendation": copy.deepcopy(
            recommendation
            if recommendation is not None
            else {"phase": "idle", "last_proposal": None, "history": []}
        ),
        "pending_confirmation": None,
        "dialogue_focus": copy.deepcopy(dialogue_focus if dialogue_focus is not None else empty_focus()),
        "reference_context": copy.deepcopy(
            reference_context if reference_context is not None else {"status": "none", "targets": []}
        ),
        "action_history": [],
        "robot_state": copy.deepcopy(
            robot_state
            if robot_state is not None
            else {
                "section": "topping",
                "task_queue": [],
                "active_task": None,
                "completed_tasks": [],
                "robot_started": False,
            }
        ),
        "message": message,
    }


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def general_speech_evidence(message: str) -> str:
    # Reviewer가 단순 topic label이 아니라 실제 일반대화 절을 확인하도록 표면 marker를 보관한다.
    for marker in (
        "설명해줘", "알려줘", "말해줘", "정리해줘", "궁금해", "비교해줘", "소개해줘",
        "뭐야", "어떻게", "왜", "차이는", "차이가", "어떤",
    ):
        if marker in message:
            return marker
    if "?" in message:
        return "?"
    raise ValueError(f"general speech evidence not found: {message}")


def make_row(
    row_id: str,
    family: str,
    message: str,
    target: dict,
    *,
    mentions: list[str] | None = None,
    task_evidence: list[str] | None = None,
    general_evidence: list[str] | None = None,
    order: dict | None = None,
    preferences: list[dict] | None = None,
    recommendation: dict | None = None,
    history: list[dict] | None = None,
    dialogue_focus: dict | None = None,
    reference_context: dict | None = None,
    robot_state: dict | None = None,
    unsupported_task: bool = False,
    pair_id: str | None = None,
    split: str = "train",
) -> dict:
    mentions = unique_in_order(mentions or [])
    target = copy.deepcopy(target)
    if mentions:
        target["mentions"] = mentions

    history = copy.deepcopy(history or [])
    row = {
        "id": row_id,
        "scenario_family": family,
        "semantic_id": row_id,
        "paraphrase_group": pair_id or row_id,
        "style": STYLES[(int(row_id.rsplit("_", 1)[-1]) - 1) % len(STYLES)],
        "difficulty": "hard",
        "context_type": "multi_turn" if history or reference_context and reference_context["status"] != "none" else "single_turn",
        "challenge_group": True,
        "history": history,
        "input": base_input(
            message,
            order=order,
            preferences=preferences,
            recommendation=recommendation,
            history=history,
            dialogue_focus=dialogue_focus,
            reference_context=reference_context,
            robot_state=robot_state,
        ),
        "target": target,
        "split": split,
        "generation_meta": {
            "source": "route_rebalance_boundary_generation",
            "pair_id": pair_id,
        },
        "_review_spec": {
            "expected_route": target["route"],
            "expected_mentions": mentions,
            "expected_target": copy.deepcopy(target),
            "task_evidence": task_evidence or [],
            "general_evidence": general_evidence or [],
            "unsupported_task": unsupported_task,
        },
    }
    return row


def history_for_focus(targets: list[str], *, previous: list[str] | None = None) -> tuple[list[dict], dict]:
    if previous is None:
        history = [
            {"role": "user", "content": f"{'하고 '.join(targets)} 얘기해줘"},
            {"role": "assistant", "content": "말씀하신 재료를 기준으로 설명해드렸어요."},
        ]
        focus = {"current": {"mentions": targets, "history_turn": 1}, "recent": []}
        return history, focus

    history = [
        {"role": "user", "content": f"{'하고 '.join(previous)} 얘기해줘"},
        {"role": "assistant", "content": "먼저 말씀하신 재료를 설명해드렸어요."},
        {"role": "user", "content": f"이번에는 {'하고 '.join(targets)} 알려줘"},
        {"role": "assistant", "content": "방금 말씀하신 재료도 설명해드렸어요."},
    ]
    focus = {
        "current": {"mentions": targets, "history_turn": 2},
        "recent": [{"mentions": previous, "history_turn": 1}],
    }
    return history, focus


def generate_general() -> list[dict]:
    rows: list[dict] = []

    def add(family: str, message: str, mentions: list[str] | None, evidence: str, **kwargs) -> None:
        number = len(rows) + 1
        rows.append(
            make_row(
                f"reb_general_{number:04d}",
                family,
                message,
                {"route": "general"},
                mentions=mentions,
                general_evidence=[general_speech_evidence(message)],
                **kwargs,
            )
        )

    supported_templates = (
        ("설명", "{e}에 대해 처음 듣는 사람도 알게 설명해줘"),
        ("맛", "{e} 맛의 특징은 보통 어떻게 표현해?"),
        ("역사", "{e}의 유래와 음식에서 쓰인 역사를 짧게 알려줘"),
        ("영양", "{e}에 들어 있는 대표적인 영양 특징이 궁금해"),
        ("조리", "{e} 조리 시 어떤 성질이 나타나는지 알려줘"),
        ("보관", "{e} 보관 시 무엇을 조심해야 하는지 알려줘"),
    )
    for index in range(80):
        entity = SUPPORTED[index % len(SUPPORTED)]
        evidence, template = supported_templates[(index // len(SUPPORTED)) % len(supported_templates)]
        add("supported_entity_general", template.format(e=entity), [entity], evidence)

    unsupported_templates = (
        ("정체", "주문하려는 건 아니고 {e}이라는 재료를 설명해줘"),
        ("풍미", "{e} 특유의 맛과 향은 어떻게 설명할 수 있어?"),
        ("문화", "{e} 활용 요리에는 무엇이 있는지 궁금해"),
    )
    for index in range(45):
        entity = UNSUPPORTED[index % len(UNSUPPORTED)]
        evidence, template = unsupported_templates[index // len(UNSUPPORTED)]
        add("unsupported_entity_general", template.format(e=entity), [entity], evidence)

    comparison_pairs = (
        ("치즈", "버섯"), ("오일", "크림"), ("토마토", "크림"), ("얇은면", "넓은면"),
        ("양파", "버섯"), ("소시지", "게살"), ("치즈", "모짜렐라"), ("햄", "소시지"),
        ("새우", "게살"), ("바질", "파슬리"), ("마늘", "양파"), ("베이컨", "살라미"),
        ("브로콜리", "버섯"), ("참치", "게살"), ("오일", "토마토"), ("유제품", "비건"),
        ("갑각류", "육류"), ("페퍼론치노", "할라피뇨"), ("파프리카", "당근"), ("미트볼", "소시지"),
    )
    for first, second in comparison_pairs:
        add(
            "entity_comparison_general",
            f"{first}, {second} 두 대상의 맛과 식감은 어떻게 달라?",
            [first, second],
            "맛과 식감 차이",
            pair_id=f"comparison_{first}_{second}",
        )
        add(
            "entity_comparison_general",
            f"{first}, {second} 두 대상을 영양과 조리 용도 기준으로 비교해줘",
            [first, second],
            "영양과 조리 용도 비교",
            pair_id=f"comparison_{first}_{second}",
        )

    robot_questions = [
        ("로봇이라는 말은 원래 어디에서 나온 거야?", ["로봇"]),
        ("로봇은 사람의 명령을 어떤 방식으로 이해해?", ["로봇"]),
        ("협동로봇과 산업용 로봇의 차이를 설명해줘", ["협동로봇", "산업용 로봇"]),
        ("조리 로봇은 일반 로봇과 설계가 어떻게 달라?", ["조리 로봇", "일반 로봇"]),
        ("로봇 팔의 관절은 어떤 원리로 움직여?", ["로봇 팔"]),
        ("로봇이 카메라로 재료를 구분하는 원리가 궁금해", ["로봇"]),
        ("로봇 센서는 사람의 감각과 어떤 차이가 있어?", ["로봇 센서"]),
        ("자동화와 로봇 공학은 같은 개념이야?", ["자동화", "로봇 공학"]),
        ("서비스 로봇이 식당에서 맡을 수 있는 역할을 알려줘", ["서비스 로봇"]),
        ("로봇의 안전 정지는 어떤 원리야?", ["로봇"]),
        ("조리 로봇을 청소할 때 중요한 점은 뭐야?", ["조리 로봇"]),
        ("로봇은 피로를 느끼지 않는데 왜 쉬어야 해?", ["로봇"]),
        ("로봇 제어기와 모터는 어떻게 신호를 주고받아?", ["로봇 제어기", "모터"]),
        ("로봇이 물체를 놓쳤는지 어떻게 알아차려?", ["로봇"]),
        ("로봇의 경로 계획이 무엇인지 쉽게 말해줘", ["로봇"]),
        ("이 로봇은 어떤 종류의 장치라고 보면 돼?", ["로봇"]),
        ("로봇이 음식점에서 사용된 역사를 알려줘", ["로봇"]),
        ("사람과 로봇이 같이 일할 때 지켜야 할 원칙은 뭐야?", ["사람", "로봇"]),
        ("로봇 팔 끝의 도구는 작업마다 왜 바뀌어?", ["로봇 팔"]),
        ("로봇의 반복 정밀도라는 말이 무슨 뜻이야?", ["로봇"]),
        ("로봇이 음성을 글로 바꾸는 과정도 설명할 수 있어?", ["로봇"]),
        ("로봇에게 자연어 모델이 필요한 이유가 뭐야?", ["로봇", "자연어 모델"]),
        ("로봇의 비상 정지 버튼은 일반 정지와 뭐가 달라?", ["로봇"]),
        ("조리 자동화가 위생 관리에 어떤 도움을 줘?", ["조리 자동화"]),
        ("로봇이 같은 동작을 반복해도 오차가 생기는 이유는 뭐야?", ["로봇"]),
        ("로봇의 작업 공간은 어떻게 계산해?", ["로봇"]),
        ("로봇에 쓰는 거리 센서는 어떤 종류가 있어?", ["로봇", "거리 센서"]),
        ("로봇이 재료 무게를 측정하는 방법을 알려줘", ["로봇"]),
        ("로봇 소프트웨어와 기계 장치는 어떻게 연결돼?", ["로봇 소프트웨어", "기계 장치"]),
        ("로봇이 실패를 감지한 뒤 보통 어떤 절차를 거쳐?", ["로봇"]),
        ("요리사가 쓰는 도구와 로봇 그리퍼를 비교해줘", ["요리사", "로봇 그리퍼"]),
        ("로봇의 속도를 너무 빠르게 하면 어떤 문제가 생겨?", ["로봇"]),
        ("로봇이 사람 가까이에서 움직일 때 속도를 줄이는 이유는 뭐야?", ["로봇"]),
        ("로봇 학습과 로봇 제어는 어떤 차이가 있어?", ["로봇 학습", "로봇 제어"]),
        ("로봇이 음식을 다루는 기술은 앞으로 어떻게 발전할까?", ["로봇"]),
    ]
    for message, mentions in robot_questions:
        add("robot_general_boundary", message, mentions, "로봇 일반 지식")

    food_topics = (
        "파스타", "알덴테", "소스 유화", "면 삶기", "이탈리아 요리",
        "글루텐", "감칠맛", "식품 보관", "위생 관리", "향신료",
    )
    food_templates = (
        ("원리", "{e}의 기본 원리를 쉽게 설명해줘"),
        ("역사", "{e} 발전 과정을 역사 순서대로 짧게 정리해줘"),
        ("실수", "{e}에서 초보자가 자주 하는 실수는 뭐야?"),
        ("특징", "{e}의 대표적인 특징을 세 가지로 알려줘"),
    )
    for index in range(35):
        topic = food_topics[index % len(food_topics)]
        evidence, template = food_templates[(index // len(food_topics)) % len(food_templates)]
        add("food_knowledge_general", template.format(e=topic), [topic], evidence)

    nondomain_messages = [
        "양자역학을 중학생도 이해할 수 있게 설명해줘",
        "대한민국의 근현대사를 짧게 정리해줘",
        "비가 내리는 원리는 뭐야?",
        "달이 지구 주위를 도는 이유를 알려줘",
        "컴퓨터가 이진수를 쓰는 까닭은 뭐야?",
        "오늘 기분이 가라앉을 때 할 만한 일을 알려줘",
        "좋은 질문을 만드는 방법이 궁금해",
        "책을 오래 기억하면서 읽는 방법을 알려줘",
        "운동 전에 준비 운동을 하는 이유가 뭐야?",
        "고양이가 골골거리는 이유를 설명해줘",
        "바다가 파랗게 보이는 이유는 뭐야?",
        "사진에서 원근감이 생기는 원리를 알려줘",
        "재즈와 클래식 음악의 차이는 뭐야?",
        "한글이 만들어진 배경을 짧게 말해줘",
        "비행기가 뜨는 원리를 쉽게 설명해줘",
        "식물이 빛을 향해 자라는 이유가 뭐야?",
        "확률과 가능성은 어떻게 달라?",
        "왜 잠을 자야 하는지 알려줘",
        "기억이 형성되는 과정을 쉽게 설명해줘",
        "팀 프로젝트에서 역할을 나누는 방법을 알려줘",
        "긴장될 때 호흡을 가다듬는 방법이 있어?",
        "색의 삼원색이 무엇인지 설명해줘",
        "인터넷에서 정보가 이동하는 원리를 알려줘",
        "지진이 발생하는 이유는 뭐야?",
        "시간대가 나라마다 다른 이유를 알려줘",
    ]
    for message in nondomain_messages:
        add("non_domain_general", message, [], "비주문 일반 질문")

    hypothetical_entities = SUPPORTED[:11] + UNSUPPORTED[:9]
    for entity in hypothetical_entities:
        add(
            "hypothetical_question_boundary",
            f"지금 주문을 바꾸자는 건 아니고 {entity} 추가 시 맛이 어떻게 달라져?",
            [entity],
            "가정적 맛 질문",
            pair_id=f"hypothetical_{entity}",
        )
        add(
            "hypothetical_question_boundary",
            f"{entity} 제외를 요청하는 건 아니야, 빠졌을 때 생기는 차이만 설명해줘",
            [entity],
            "가정적 제외 질문",
            pair_id=f"hypothetical_{entity}",
        )

    assert len(rows) == 300, len(rows)
    return rows


def generate_mixed() -> list[dict]:
    rows: list[dict] = []

    def add(family: str, message: str, target: dict, mentions: list[str] | None, task: str, general: str, **kwargs) -> None:
        number = len(rows) + 1
        rows.append(
            make_row(
                f"reb_mixed_{number:04d}",
                family,
                message,
                target,
                mentions=mentions,
                task_evidence=[task],
                general_evidence=[general],
                **kwargs,
            )
        )

    amount_words = (("조금", "low"), ("보통으로", "normal"), ("많이", "high"), ("빼고", "none"), ("넣고", "normal"))
    knowledge_clauses = (
        "향이 강해지는 이유도 알려줘",
        "영양상 특징도 설명해줘",
        "보통 어떤 요리에 쓰이는지도 알려줘",
        "식감이 생기는 원리도 설명해줘",
        "신선하게 보관하는 법도 알려줘",
    )
    for topping in TOPPINGS:
        for variant, ((word, amount), general_clause) in enumerate(zip(amount_words, knowledge_clauses)):
            if amount == "none":
                task_clause = f"{topping} 항목은 빼고"
            elif word == "넣고":
                task_clause = f"{topping} 넣고"
            else:
                task_clause = f"{topping} 양은 {word} 넣고"
            message = f"{task_clause}, {topping}의 {general_clause}"
            add(
                "order_mutation_general_mixed",
                message,
                {"route": "mixed", "order": {"toppings": {topping: amount}}},
                [topping],
                task_clause,
                general_clause,
                pair_id=f"mixed_order_{topping}_{variant}",
            )

    sauce_general = (
        "풍미가 만들어지는 과정도 알려줘",
        "다른 소스와의 차이도 설명해줘",
        "유화되는 원리도 알려줘",
        "기원이 어디인지도 알려줘",
        "잘 어울리는 음식도 설명해줘",
    )
    for sauce in SAUCES:
        for variant, general_clause in enumerate(sauce_general):
            task_clause = f"소스 선택은 {sauce} 항목으로 바꾸고"
            message = f"{task_clause} {sauce} 소스의 {general_clause}"
            order = empty_order()
            order["sauce"] = "토마토" if sauce != "토마토" else "오일"
            add(
                "order_mutation_general_mixed",
                message,
                {"route": "mixed", "order": {"sauce": sauce}},
                [sauce],
                task_clause,
                general_clause,
                order=order,
                pair_id=f"mixed_sauce_{sauce}_{variant}",
            )

    noodle_general = (
        "소스를 머금는 정도도 설명해줘",
        "식감 차이가 나는 이유도 알려줘",
        "어떤 요리에 많이 쓰이는지도 알려줘",
        "삶는 시간의 특징도 설명해줘",
        "모양이 그렇게 만들어진 이유도 알려줘",
    )
    for noodle in NOODLES:
        for variant, general_clause in enumerate(noodle_general):
            task_clause = f"면은 {noodle}으로 바꾸고"
            message = f"{task_clause} {noodle}이 {general_clause}"
            order = empty_order()
            order["noodle_type"] = "넓은면" if noodle == "얇은면" else "얇은면"
            add(
                "order_mutation_general_mixed",
                message,
                {"route": "mixed", "order": {"noodle_type": noodle}},
                [noodle],
                task_clause,
                general_clause,
                order=order,
                pair_id=f"mixed_noodle_{noodle}_{variant}",
            )

    assert len(rows) == 55

    query_general = (
        "맛 특징도 알려줘",
        "영양상 특징도 설명해줘",
        "조리할 때 식감이 변하는 과정도 알려줘",
    )
    for topping in TOPPINGS:
        for variant, general_clause in enumerate(query_general):
            order = empty_order()
            order["toppings"][topping] = ("low", "normal", "high")[variant]
            task_clause = f"지금 {topping} 양이 얼마나 되는지 알려주고"
            message = f"{task_clause} {topping}의 {general_clause}"
            add(
                "state_query_general_mixed",
                message,
                {"route": "mixed", "queries": [{"type": "order_item", "target": topping}]},
                [topping],
                task_clause,
                general_clause,
                order=order,
            )

    scalar_cases = [
        ("sauce", "토마토", "현재 소스가 뭔지 확인해주고", "토마토 소스의 산미가 생기는 이유도 설명해줘"),
        ("sauce", "크림", "지금 선택된 소스를 알려주고", "크림 소스가 걸쭉해지는 원리도 알려줘"),
        ("noodle_type", "얇은면", "현재 면 종류가 뭔지 말해주고", "얇은면의 식감 특징도 설명해줘"),
        ("noodle_type", "넓은면", "골라둔 면 종류를 확인해주고", "넓은면이 소스를 잘 머금는 이유도 알려줘"),
        ("noodle_portion", "normal", "지금 면 양이 어느 정도인지 알려주고", "파스타 1인분 기준도 설명해줘"),
        ("noodle_portion", "high", "선택된 면 양부터 확인해주고", "면 양을 정할 때 고려할 점도 알려줘"),
    ]
    for field, value, task_clause, general_clause in scalar_cases:
        order = empty_order()
        order[field] = value
        mention = value if value in SAUCES + NOODLES else "파스타"
        add(
            "state_query_general_mixed",
            f"{task_clause} {general_clause}",
            {"route": "mixed", "queries": [{"type": "order_field", "target": field}]},
            [mention] if mention in f"{task_clause} {general_clause}" else [],
            task_clause,
            general_clause,
            order=order,
        )

    for index in range(5):
        task_clause = ("지금 로봇이 무슨 작업 중인지 알려주고", "현재 로봇 상태를 확인해주고", "로봇이 지금 멈춰 있는지 알려주고", "지금 로봇이 어느 단계인지 말해주고", "로봇 작업 상황부터 보여주고")[index]
        general_clause = ("조리 로봇의 센서 구조도 설명해줘", "로봇 팔이 움직이는 원리도 알려줘", "로봇의 안전 정지 원리도 설명해줘", "조리 로봇의 역사도 짧게 알려줘", "로봇 그리퍼의 종류도 설명해줘")[index]
        message = f"{task_clause} {general_clause}"
        add(
            "state_query_general_mixed",
            message,
            {"route": "mixed", "queries": [{"type": "robot_status"}]},
            ["로봇"],
            task_clause,
            general_clause,
        )

    completed_generals = ("로봇의 작업 기록이 왜 필요한지도 알려줘", "자동화에서 완료 신호가 어떤 역할인지 설명해줘", "로봇이 동작 완료를 판단하는 원리도 알려줘")
    for index, general_clause in enumerate(completed_generals):
        task_clause = ("방금 로봇이 끝낸 작업을 확인해주고", "최근 완료한 로봇 작업을 알려주고", "조금 전에 끝난 작업부터 말해주고")[index]
        message = f"{task_clause} {general_clause}"
        add(
            "state_query_general_mixed",
            message,
            {"route": "mixed", "queries": [{"type": "robot_completed"}]},
            ["로봇"],
            task_clause,
            general_clause,
        )

    restriction_query_cases = (("게살", "갑각류"), ("치즈", "유제품"), ("소시지", "육류"))
    for target, category in restriction_query_cases:
        order = empty_order()
        order["restrictions"] = [{"target": target, "reason": "allergy"}]
        task_clause = "현재 등록된 알레르기 제한을 알려주고"
        general_clause = f"{category} 알레르기의 특징도 설명해줘"
        add(
            "state_query_general_mixed",
            f"{task_clause} {general_clause}",
            {"route": "mixed", "queries": [{"type": "restriction_status"}]},
            [category],
            task_clause,
            general_clause,
            order=order,
        )

    assert len(rows) == 90

    reference_targets = list(TOPPINGS)
    reference_general = (
        "맛 특징", "영양 성분", "조리할 때 향이 변하는 이유",
        "대표적인 사용법", "식감이 생기는 원리", "보관할 때 주의점",
    )
    for index in range(35):
        if index % 5 == 0:
            targets = [reference_targets[index % 6], reference_targets[(index + 1) % 6]]
            history, focus = history_for_focus(targets)
            task_clause = "둘 다 많이 넣고"
            target = {"route": "mixed", "order": {"toppings": {targets[0]: "high", targets[1]: "high"}}}
            context = {"status": "resolved", "targets": targets}
        elif index % 5 == 1:
            targets = [reference_targets[index % 6], reference_targets[(index + 1) % 6]]
            history, focus = history_for_focus(targets)
            task_clause = "첫 번째 거는 빼고"
            target = {"route": "mixed", "order": {"toppings": {targets[0]: "none"}}}
            context = {"status": "resolved", "targets": [targets[0]]}
        elif index % 5 == 2:
            targets = [reference_targets[index % 6], reference_targets[(index + 1) % 6]]
            history, focus = history_for_focus(targets)
            task_clause = "두 번째 거는 조금 넣고"
            target = {"route": "mixed", "order": {"toppings": {targets[1]: "low"}}}
            context = {"status": "resolved", "targets": [targets[1]]}
        elif index % 5 == 3:
            current = [reference_targets[index % 6]]
            previous = [reference_targets[(index + 2) % 6]]
            history, focus = history_for_focus(current, previous=previous)
            task_clause = "아까 그거 많이 넣고"
            target = {"route": "mixed", "order": {"toppings": {previous[0]: "high"}}}
            context = {"status": "resolved", "targets": previous}
            targets = previous
        else:
            targets = [reference_targets[index % 6]]
            history, focus = history_for_focus(targets)
            task_clause = "그거 넣고"
            target = {"route": "mixed", "order": {"toppings": {targets[0]: "normal"}}}
            context = {"status": "resolved", "targets": targets}

        explicit_entity = reference_targets[(index + 3) % 6]
        general_clause = f"{explicit_entity}의 {reference_general[(index // 6) % len(reference_general)]}도 설명해줘"
        message = f"{task_clause} {general_clause}"
        add(
            "reference_resolved_general_mixed",
            message,
            target,
            [explicit_entity],
            task_clause,
            general_clause,
            history=history,
            dialogue_focus=focus,
            reference_context=context,
        )

    ambiguous_pairs = (("치즈", "버섯"), ("양파", "소시지"), ("게살", "페퍼론치노"), ("토마토", "크림"), ("얇은면", "넓은면"))
    for index in range(25):
        candidates = list(ambiguous_pairs[index % len(ambiguous_pairs)])
        history, focus = history_for_focus(candidates)
        explicit_entity = SUPPORTED[(index + 4) % len(SUPPORTED)]
        task_clause = ("그거 빼고", "그거 조금 넣고", "그거 많이 넣고", "이거 제외하고", "방금 그거 넣고")[index % 5]
        ambiguous_general = ("특징은 따로 설명해줘", "보관할 때 주의점도 설명해줘")
        general_clause = f"{explicit_entity}의 {ambiguous_general[index // 15]}"
        message = f"{task_clause} {general_clause}"
        add(
            "reference_ambiguous_general_mixed",
            message,
            {"route": "mixed", "clarify": True},
            [explicit_entity],
            task_clause,
            general_clause,
            history=history,
            dialogue_focus=focus,
            reference_context={"status": "ambiguous", "targets": candidates},
        )

    for index in range(30):
        entity = UNSUPPORTED[index % len(UNSUPPORTED)]
        task_clause = (f"{entity} 넣어주고", f"메뉴를 {entity} 기준으로 바꿔주고")[index // len(UNSUPPORTED)]
        general_clause = (f"{entity}에 관한 기본 설명도 알려줘", f"{entity}의 유래도 설명해줘")[index // len(UNSUPPORTED)]
        message = f"{task_clause} {general_clause}"
        add(
            "unsupported_task_general_mixed",
            message,
            {"route": "mixed", "clarify": True},
            [entity],
            task_clause,
            general_clause,
            unsupported_task=True,
        )

    commit_phrases = ("바로 진행해", "이대로 진행해", "바로 시작해", "담기 시작해", "주문 확정해")
    for index in range(25):
        phrase = commit_phrases[index % len(commit_phrases)]
        entity = TOPPINGS[index % len(TOPPINGS)]
        general_clause = (
            f"{entity}의 맛 특징도 알려줘",
            f"{entity} 제조 과정도 설명해줘",
            f"{entity}의 영양 정보도 알려줘",
            f"{entity} 활용 요리도 설명해줘",
            f"{entity} 보관법도 알려줘",
        )[index // len(commit_phrases)]
        if index % 2 == 0:
            task_clause = f"{entity} 많이 넣고 {phrase}"
            target = {"route": "mixed", "order": {"toppings": {entity: "high"}}, "commit": True}
        else:
            task_clause = phrase
            target = {"route": "mixed", "commit": True}
        message = f"{task_clause}, {general_clause}"
        add("commit_general_mixed", message, target, [entity], task_clause, general_clause)

    restriction_cases = (
        ("게살", "allergy", "알레르기로 등록하고", "갑각류 알레르기가 무엇인지도 설명해줘", "갑각류"),
        ("치즈", "cannot_eat", "먹을 수 없는 재료로 등록하고", "유제품을 못 먹는 경우도 설명해줘", "유제품"),
        ("버섯", "dislike", "싫어하는 재료로 해두고", "버섯 종류도 알려줘", "버섯"),
        ("소시지", "dietary_rule", "식단 제한으로 등록하고", "육류를 피하는 식단도 설명해줘", "육류"),
        ("유제품", "allergy", "알레르기 제한으로 추가하고", "유제품 알레르기 특징도 알려줘", "유제품"),
    )
    restriction_prefixes = ("이번 주문에서", "안전하게", "내 정보에", "우선", "그리고")
    for index in range(25):
        target_name, reason, operation, general_clause, general_mention = restriction_cases[index % len(restriction_cases)]
        prefix = restriction_prefixes[index // len(restriction_cases)]
        task_clause = f"{prefix} {target_name} 항목은 {operation}"
        message = f"{task_clause} {general_clause}"
        add(
            "restriction_general_mixed",
            message,
            {"route": "mixed", "restrictions": [{"target": target_name, "reason": reason, "action": "add"}]},
            unique_in_order([target_name, general_mention]),
            task_clause,
            general_clause,
        )

    recommendation_cases = (
        ("매콤하게", "페퍼론치노", "페퍼론치노가 어떤 향신료인지도 알려줘"),
        ("크림 중심", "크림", "크림 소스의 유래도 설명해줘"),
        ("담백하게", "오일", "오일 소스의 특징도 알려줘"),
        ("푸짐하게", "치즈", "치즈가 숙성되는 원리도 알려줘"),
        ("채소 위주", "야채", "야채를 조리하면 단맛이 나는 이유도 알려줘"),
    )
    recommendation_prefixes = ("추천해주고", "구성을 골라주면서", "추천안을 만들어주고", "메뉴를 제안하면서")
    for index in range(20):
        criteria, mention, general_clause = recommendation_cases[index % len(recommendation_cases)]
        action_word = recommendation_prefixes[index // len(recommendation_cases)]
        if criteria.endswith("중심"):
            task_clause = f"현재 주문을 {criteria}으로 {action_word}"
        elif criteria.endswith("위주"):
            task_clause = f"현재 주문을 {criteria}로 {action_word}"
        else:
            task_clause = f"현재 주문은 {criteria} {action_word}"
        message = f"{task_clause} {general_clause}"
        add(
            "recommendation_general_mixed",
            message,
            {"route": "mixed", "recommendation": {"action": "request", "scope": "current", "criteria": criteria}},
            [mention],
            task_clause,
            general_clause,
        )

    assert len(rows) == 250, len(rows)
    return rows


def target_for_task_entity(entity: str, operation_index: int) -> dict:
    target = {"route": "task", "mentions": [entity]}
    if entity in TOPPINGS:
        target["order"] = {"toppings": {entity: ("low", "normal", "high")[operation_index % 3]}}
    elif entity in SAUCES:
        target["order"] = {"sauce": entity}
    elif entity in NOODLES:
        target["order"] = {"noodle_type": entity}
    else:
        target["clarify"] = True
    return target


def generate_hard_eval(train_messages: set[str]) -> list[dict]:
    rows: list[dict] = []

    def add(category: str, message: str, target: dict, mentions: list[str] | None, task: list[str], general: list[str], **kwargs) -> None:
        number = len(rows) + 1
        assert message not in train_messages, message
        row = make_row(
            f"hard_route_{number:04d}",
            category,
            message,
            target,
            mentions=mentions,
            task_evidence=task,
            general_evidence=general,
            split="hard_eval",
            **kwargs,
        )
        row["eval_category"] = category
        rows.append(row)

    pair_entities = list(SAUCES + NOODLES + TOPPINGS + UNSUPPORTED[:11])
    general_forms = (
        "{e} 자체가 어떤 재료인지 핵심만 말해줘",
        "{e}의 맛을 다른 재료와 구별해서 설명해줘",
        "{e} 활용이 시작된 배경을 알려줘",
        "{e}의 조리 특성을 간단히 정리해줘",
        "{e}에 관한 영양 상식을 알려줘",
    )
    task_forms = (
        "설명 말고 이번 주문에는 {e} 조금 넣어줘",
        "이번 선택은 {e} 기준으로 해줘",
        "실제로 {e} 많이 넣는 걸로 바꿔줘",
        "주문 항목에 {e} 추가해줘",
        "지금 선택을 {e} 기준으로 적용해줘",
    )
    pair_index = 0
    for entity in pair_entities:
        for form_index in range(5):
            pair_index += 1
            pair_id = f"hard_minpair_{pair_index:03d}"
            general_message = general_forms[form_index].format(e=entity)
            add(
                "general_task_minimal_pair",
                general_message,
                {"route": "general"},
                [entity],
                [],
                ["설명" if "설명" in general_message else entity],
                pair_id=pair_id,
            )
            task_message = task_forms[form_index].format(e=entity)
            task_target = target_for_task_entity(entity, form_index)
            task_target.pop("mentions", None)
            add(
                "general_task_minimal_pair",
                task_message,
                task_target,
                [entity],
                [("주문", "기준으로 해줘", "바꿔줘", "추가해줘", "적용해줘")[form_index]],
                [],
                unsupported_task=entity in UNSUPPORTED,
                pair_id=pair_id,
            )
    assert pair_index == 110

    mixed_eval_aspects = (
        "산지별 특징", "익을 때 향이 바뀌는 이유", "권장 보관 온도", "전통 요리에서의 쓰임",
        "식감이 생기는 구조", "대표적인 영양 성분", "신선도를 구별하는 방법", "손질할 때 주의점",
        "다른 재료와 어울리는 원리", "가열 시간에 따른 변화", "이름이 생긴 유래", "계절별 품질 차이",
        "향을 살리는 조리법", "식문화에서의 의미", "지역별 조리 방식", "수분 함량의 특징",
        "색이 변하는 이유", "맛을 표현하는 방법", "구매할 때 확인할 점",
    )
    for index in range(110):
        entity = TOPPINGS[index % len(TOPPINGS)]
        amount_word, amount = (("살짝", "low"), ("보통으로", "normal"), ("듬뿍", "high"), ("빼는 걸로", "none"))[index % 4]
        task_clause = f"주문에서 {entity} 양은 {amount_word} 해주고"
        aspect = mixed_eval_aspects[index // len(TOPPINGS)]
        general_clause = f"별개로 {entity}의 {aspect}도 알려줘"
        add(
            "mixed_boundary",
            f"{task_clause} {general_clause}",
            {"route": "mixed", "order": {"toppings": {entity: amount}}},
            [entity],
            [task_clause],
            [general_clause],
        )

    reference_eval_aspects = (
        "향이 강해지는 조건", "영양 성분의 특징", "가열할 때 생기는 변화",
        "신선하게 보관하는 방법", "대표적인 조리 용도", "맛을 표현하는 방법",
    )
    for index in range(35):
        targets = [TOPPINGS[index % 6], TOPPINGS[(index + 2) % 6]]
        history, focus = history_for_focus(targets)
        if index % 2:
            reference_phrase = "두 개 다 조금 넣고"
            resolved = targets
            patch = {targets[0]: "low", targets[1]: "low"}
        else:
            reference_phrase = "두 번째 거 많이 넣고"
            resolved = [targets[1]]
            patch = {targets[1]: "high"}
        explicit = TOPPINGS[(index + 4) % 6]
        general_clause = f"{explicit}의 {reference_eval_aspects[index // 6]}도 알려줘"
        add(
            "reference_mixed",
            f"{reference_phrase} {general_clause}",
            {"route": "mixed", "order": {"toppings": patch}},
            [explicit],
            [reference_phrase],
            [general_clause],
            history=history,
            dialogue_focus=focus,
            reference_context={"status": "resolved", "targets": resolved},
        )

    unsupported_eval_aspects = (
        "지역별 이름도 알려줘",
        "만드는 방식도 설명해줘",
        "맛을 표현하는 말도 알려줘",
    )
    for index in range(35):
        entity = UNSUPPORTED[index % len(UNSUPPORTED)]
        task_clause = f"메뉴에 {entity}도 추가해주고"
        general_clause = f"{entity}의 {unsupported_eval_aspects[index // len(UNSUPPORTED)]}"
        add(
            "unsupported_mixed",
            f"{task_clause} {general_clause}",
            {"route": "mixed", "clarify": True},
            [entity],
            [task_clause],
            [general_clause],
            unsupported_task=True,
        )

    query_general_aspects = (
        "요리에서 맡는 역할", "가열했을 때의 변화", "대표적인 영양 특징",
        "식감을 만드는 요소", "보관할 때 주의할 점", "다른 재료와의 조합",
    )
    query_task_phrases = (
        "현재 선택량만 확인해줘", "설정된 양만 말해줘", "담기로 한 분량만 알려줘",
        "주문에 잡힌 양만 확인해줘", "현재 수량 정보만 보여줘", "저장된 양만 답해줘",
    )
    for index in range(35):
        entity = TOPPINGS[index % len(TOPPINGS)]
        pair_id = f"hard_query_pair_{index + 1:03d}"
        occurrence = index // len(TOPPINGS)
        general_message = f"현재 주문 얘기는 말고 {entity}의 {query_general_aspects[occurrence]}을 알려줘"
        add(
            "query_vs_general",
            general_message,
            {"route": "general"},
            [entity],
            [],
            [query_general_aspects[occurrence]],
            pair_id=pair_id,
        )
        order = empty_order()
        order["toppings"][entity] = ("low", "normal", "high")[index % 3]
        task_message = f"일반 설명 말고 내 주문의 {entity} {query_task_phrases[occurrence]}"
        add(
            "query_vs_general",
            task_message,
            {"route": "task", "queries": [{"type": "order_item", "target": entity}]},
            [entity],
            [query_task_phrases[occurrence]],
            [],
            order=order,
            pair_id=pair_id,
        )

    commit_phrases = ("바로 진행해", "이대로 진행해", "바로 시작해", "담기 시작해", "주문 확정해")
    for index in range(25):
        entity = TOPPINGS[index % 6]
        phrase = commit_phrases[index % 5]
        task_clause = f"{entity} 양은 보통으로 넣고 {phrase}"
        general_clause = f"그다음 {entity}의 품종 차이도 설명해줘"
        add(
            "commit_mixed",
            f"{task_clause}, {general_clause}",
            {"route": "mixed", "order": {"toppings": {entity: "normal"}}, "commit": True},
            [entity],
            [task_clause],
            [general_clause],
        )

    return rows


def main() -> None:
    general = generate_general()
    mixed = generate_mixed()
    train_messages = {row["input"]["message"] for row in general + mixed}
    hard_eval = generate_hard_eval(train_messages)

    assert len(train_messages) == len(general) + len(mixed)
    assert len({row["input"]["message"] for row in hard_eval}) == len(hard_eval)

    dump_jsonl(HERE / "generated_general_raw.jsonl", general)
    dump_jsonl(HERE / "generated_mixed_raw.jsonl", mixed)
    dump_jsonl(HERE / "hard_eval_raw.jsonl", hard_eval)
    print(json.dumps({"general_raw": len(general), "mixed_raw": len(mixed), "hard_eval_raw": len(hard_eval)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
