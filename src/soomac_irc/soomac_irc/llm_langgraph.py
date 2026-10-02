import copy
from typing import Callable, TypedDict

from soomac_irc.agent_contract import new_decision
from soomac_irc.dialogue_focus import build_reference_context,new_dialogue_focus,update_dialogue_focus
from soomac_irc.llm_policy import (
    allowed_order_fields, build_applied_changes, build_completed_modification_warning,
    build_completed_restriction_warning, build_future_changes, build_next_prompt, build_turn_action_event, build_safe_clarify_decision,
    evaluate_runtime_policy, find_invalid_order_fields, find_invalid_queries, find_new_physical_restriction_conflicts,
    find_route_consistency_errors, has_explicit_commit_intent, recommendation_allowed_fields,
    reference_targets_are_supported, reference_targets_match_decision,
    restore_protected_order_values, validate_recommendation_proposal,)

# State / Decision 계약
# 실제 값 형식은 agent_contract.py가 잡고, 여기는 그래프가 들고 다닐 state 구조만 선언함

class SessionState(TypedDict):
    # 여러 사용자 턴 동안 유지하는 대화, 주문 상태
    order: dict  # 여러 턴 유지 현재까지 누적된 주문
    preferences: list[dict]  # 여러 턴 유지 맛·식감처럼 주문 필드 밖의 사용자 기호
    recommendation: dict  # 여러 턴 유지 추천 단계, 최근 추천안, 추천 이력
    pending_confirmation: dict | None  # 여러 턴 유지 다음 턴 답변을 기다리는 확인 건
    dialogue_focus: dict  # 여러 턴 유지 최근 user mention과 해당 history 턴을 보관하는 참조용 상태
    history: list[dict]  # 여러 턴 유지 저장 제한 없음; Decision 입력은 8192 token 안의 최근 대화만 사용
    action_history: list[dict]  # 여러 턴 유지 저장 제한 없음; 실제 주문 변경과 로봇/VLM 사건 이력


class Decision(TypedDict):
    route: str  # 한 턴만 사용 task/general/mixed 처리 경로
    mentions: list[str]  # 한 턴만 사용 현재 user 발화에서 직접 언급한 대상과 순서
    understanding: str  # 한 턴만 사용 ok면 처리 계속, clarify면 재질문
    order_patch: dict  # 한 턴만 사용 현재 발화에서 바꿀 주문 필드만 담음
    restriction_options: list[dict]  # 한 턴만 사용 알레르기·금지 재료 추가/해제 목록
    preference_options: list[dict]  # 한 턴만 사용 자유형 취향 추가/해제 목록
    recommendation: dict  # 한 턴만 사용 추천 생성·수정·선택·취소 요청
    commit: bool  # [한 턴만 사용] 지금 물리 실행까지 요청했는지
    confirmation: str  # [한 턴만 사용] 대기 중인 확인에 대한 accept/reject/none
    queries: list[dict]  # [한 턴만 사용] 주문 변경이 아닌 상태·설명 질문

# 한 번의 graph.invoke 동안 stage 사이에서만 쓰는 턴 state
class TurnState(TypedDict):
    session: SessionState  # 여러 턴 유지 이번 턴 결과를 담아 다음 턴으로 넘김
    previous_session: SessionState  # 한 턴만 사용 턴 시작 상태; 변경 전후 비교용
    user_text: str  # 한 턴만 사용 이번 사용자 발화
    robot_state: dict  # 한 턴만 사용 턴 시작 시점의 section·step·VLM 상태
    decision: Decision | None  # 한 턴만 사용 Decision Agent의 이번 발화 해석
    working_session: SessionState | None  # 한 턴만 사용 확인 답변까지 반영한 임시 상태
    candidate_session: SessionState | None  # 한 턴만 사용 이번 변경과 추천을 반영한 임시 상태
    recommendation_result: dict | None  # 한 턴만 사용 허용 필드 검사까지 끝난 추천 결과
    reference_context: dict | None  # graph 한 턴에서만 유지하는 reference clarify 원인과 대상
    policy: dict | None  # 한 턴만 사용 pass/warning/clarify/hitl/blocked 판정
    reply: str | None  # 한 턴만 사용 사용자에게 보낼 최종 응답


# 초기 state

def new_order_state() -> dict:
    # 실제 사용자 주문과 활성 restriction을 한 형식으로 관리하는 상태
    return {
        "sauce": None,  # 선택한 소스
        "noodle_type": None,  # 선택한 면 종류
        "noodle_portion": None,  # 면 양 low/normal/high
        "toppings": {},  # topping 이름 -> 양
        "restrictions": [],  # 금지 대상과 사유 목록
    }

def new_session_state() -> SessionState:
    return {
        "order": new_order_state(),  # 여러 턴에 걸쳐 계속 갱신되는 주문
        "preferences": [],  # 여러 턴에 걸쳐 누적되는 취향
        "recommendation": {  # 추천 lifecycle state
            "phase": "idle",  # idle/proposed/selected
            "last_proposal": None,  # 수정·선택할 최근 추천안
            "history": [],  # 이전 추천안 기록
        },
        "pending_confirmation": None,  # 현재 대기 중인 안전 확인
        "dialogue_focus": new_dialogue_focus(),  # history 턴에 연결되는 최근 user mention 기억
        "history": [],  # 전체 자연어 대화 이력
        "action_history": [],  # 실제 적용된 변경 이력
    }


# pending confirmation

def build_preselected_section_confirmation(session: SessionState, section: str) -> dict | None:
    # 미래 section에 미리 선택된 재료가 있는지 확인
    if section not in ("veggie", "meat", "extra"):
        return None

    items = {}  # 해당 section에 이미 골라둔 topping만 모음

    for key in allowed_order_fields(section):
        topping = key.split(".", 1)[1]  # toppings.cheese -> cheese
        amount = session["order"]["toppings"].get(topping)  # 기존 선택량

        if amount is not None:
            items[topping] = amount

    if not items:
        return None

    return {
        "type": "preselected_section",  # section 진입 전 재확인 타입
        "section": section,  # 곧 실행할 section
        "items": items,  # 이미 선택된 topping과 양
    }

def build_preselected_confirmation_reply(pending: dict) -> str:
    # 미리 선택된 section 실행 전 Python 고정 확인 문장을 만듬
    amount_words = {"low": "적게", "normal": "보통", "high": "많이"}  # 내부 enum -> 안내 표현
    item_text = ", ".join(  # 여러 재료를 한 문장으로 묶음
        f"{item} {amount_words.get(amount, amount)}"
        for item, amount in pending["items"].items()
    )
    return f"앞서 {item_text} 넣기로 하셨습니다. 그대로 담을까요?"


def resolve_pending_confirmation(session: SessionState, decision: Decision) -> tuple[SessionState, Decision]:
    # 보류된 안전 확인에 답한 뒤 같은 발화의 새 변경도 계속 처리
    working = copy.deepcopy(session)  # 확인 결과를 먼저 반영할 session 복사본
    effective = copy.deepcopy(decision)  # 확인 답변으로 commit이 바뀔 수 있는 decision 복사본
    pending = working["pending_confirmation"]  # 지난 턴에서 대기시킨 확인 건

    if pending is None or decision["confirmation"] == "none":
        return working, effective

    if pending["type"] == "preselected_section":
        working["pending_confirmation"] = None

        if decision["confirmation"] == "accept":
            effective["commit"] = True
        else:
            for item in pending["items"]:
                working["order"]["toppings"].pop(item, None)

            # plain "아니"는 False이고, "아니 치즈만 그대로" 같은 부분 유지 발화는
            # 모델이 order patch와 commit=true를 함께 내면 그 의도를 보존한다.
            effective["commit"] = decision["commit"]

        return working, effective

    if pending["type"] != "restriction_conflict":
        return working, effective

    working = copy.deepcopy(pending["candidate_session"])
    working["pending_confirmation"] = None

    if decision["confirmation"] == "accept":
        # restriction을 해제하고 보류됐던 실행 요청을 이어감
        for conflict in pending["conflicts"]:
            restriction = conflict["restriction"]  # 사용자가 해제에 동의한 제한

            if restriction in working["order"]["restrictions"]:
                working["order"]["restrictions"].remove(restriction)

        effective["commit"] = pending["decision"]["commit"] or decision["commit"]
        return working, effective

    # restriction을 유지하는 경우 충돌하는 미실행 주문값만 제거
    for conflict in pending["conflicts"]:
        key = conflict["key"]  # restriction과 충돌한 주문 필드

        if key.startswith("toppings."):
            topping = key.split(".", 1)[1]
            working["order"]["toppings"].pop(topping, None)
        elif key == "noodle_type":
            working["order"]["noodle_type"] = None
            working["order"]["noodle_portion"] = None
        else:
            working["order"][key] = None

    effective["commit"] = False
    return working, effective


# recommendation lifecycle

def apply_recommendation_proposal(session: SessionState, proposal: dict) -> SessionState:
    # 검증을 끝낸 추천안을 주문 복사본에 반영

    updated = copy.deepcopy(session)  # 원본 session을 직접 건드리지 않는 복사본
    apply_order_patch(updated["order"], proposal)
    return updated


def merge_recommendation(session: SessionState, decision: Decision, recommendation_result: dict | None) -> SessionState:
    # 추천안의 생성·수정·선택·취소 반영

    updated = copy.deepcopy(session)  # 추천 결과를 반영할 session 복사본
    request = decision["recommendation"]  # 이번 턴의 추천 요청
    action = request["action"]  # none/request/revise/select/cancel
    phase = updated["recommendation"]["phase"]  # 현재 추천 lifecycle 단계
    accept_pending = action == "select" or (  # 명시 선택 또는 proposed 상태에서 실행 요청
        action == "none" and decision["commit"] and phase == "proposed"
    )

    if action == "cancel":
        updated["recommendation"]["phase"] = "idle"
        updated["recommendation"]["last_proposal"] = None
        return updated

    if action in ("request", "revise"):
        if recommendation_result is None:
            return updated

        record = {
            "action": action,  # 생성인지 수정인지
            "scope": request["scope"],  # 추천할 주문 범위
            "criteria": request["criteria"],  # 사용자 추천 조건
            "proposal": copy.deepcopy(recommendation_result["proposal"]),  # 검증된 추천 주문값
            "reason_tags": copy.deepcopy(recommendation_result["reason_tags"]),  # 추천 근거 태그
        }
        updated["recommendation"]["last_proposal"] = copy.deepcopy(record)
        updated["recommendation"]["history"].append(copy.deepcopy(record))

        if decision["commit"]:
            updated = apply_recommendation_proposal(updated, record["proposal"])
            updated["recommendation"]["phase"] = "selected"
        else:
            updated["recommendation"]["phase"] = "proposed"

        return updated

    if not accept_pending or recommendation_result is None:
        return updated

    updated = apply_recommendation_proposal(updated, recommendation_result["proposal"])
    updated["recommendation"]["phase"] = "selected"

    if updated["recommendation"]["last_proposal"] is not None:
        updated["recommendation"]["last_proposal"]["proposal"] = copy.deepcopy(recommendation_result["proposal"])

    return updated


# 누적 state 변경

def apply_restriction_options(order: dict, options: list[dict]) -> None:
    # restriction operation을 주문 복사본에 순서대로 반영
    for option in options:  # 한 발화에 여러 제한 변경 가능
        restriction = {"target": option["target"], "reason": option["type"]}  # 제한 정보를 저장하는 통일 형식

        if option["action"] == "add" and restriction not in order["restrictions"]:
            order["restrictions"].append(restriction)
        elif option["action"] == "remove" and restriction in order["restrictions"]:
            order["restrictions"].remove(restriction)

def apply_preference_options(preferences: list[dict], options: list[dict]) -> None:
    # 자유로운 preference 문자열을 추가하거나 철회(이건 정해져 있진 않음)
    for option in options:  # 한 발화에 여러 취향 변경 가능
        preference = {"value": option["value"]}  # 취향 정보를 저장하는 통일 형식

        if option["action"] == "add" and preference not in preferences:
            preferences.append(preference)
        elif option["action"] == "remove" and preference in preferences:
            preferences.remove(preference)

def apply_order_patch(order: dict, patch: dict) -> None:
    # scalar None은 변경 없음, topping none은 삭제
    for field in ("sauce", "noodle_type", "noodle_portion"):  # scalar 주문 필드
        if patch[field] is not None:
            order[field] = patch[field]

    for topping, amount in patch["toppings"].items():  # topping별 추가·수정·삭제
        if amount == "none":
            order["toppings"].pop(topping, None)
        else:
            order["toppings"][topping] = amount


def apply_decision(session: SessionState, decision: Decision) -> SessionState:
    # restriction → preference → 주문 요청 순서로 세션 복사본에 반영
    updated = copy.deepcopy(session)  # 정책 검사 전 후보 session
    apply_restriction_options(updated["order"], decision["restriction_options"])
    apply_preference_options(updated["preferences"], decision["preference_options"])
    apply_order_patch(updated["order"], decision["order_patch"])
    return updated


# 한 턴 state 생성

def new_turn_state(session: SessionState, user_text: str, robot_state: dict) -> TurnState:
    # previous_session은 diff 계산용 턴 시작 snapshot이고, 실제 멀티턴 기억은 session["history"]에 있음
    return {
        "session": copy.deepcopy(session),  # graph stage가 최종 결과로 갱신할 session
        "previous_session": copy.deepcopy(session),  # 이번 턴 전/후 diff 기준
        "user_text": user_text.strip(),  # 앞뒤 공백을 제거한 현재 발화
        "robot_state": copy.deepcopy(robot_state),  # stage 사이에서 고정할 로봇 snapshot
        "decision": None,  # interpret_decision에서 채움
        "working_session": None,  # resolve_confirmation에서 채움
        "candidate_session": None,  # apply_workers에서 채움
        "recommendation_result": None,  # run_recommendation에서 채움
        "reference_context": None,  # Decision JSON과 SessionState에는 저장하지 않는 한 턴 내부값
        "policy": None,  # check_policy에서 채움
        "reply": None,  # generate_response에서 채움
    }


# LangGraph pipeline
def build_graph(
    call_decision: Callable[[SessionState, str, dict, dict | None], Decision],  # 발화 -> 구조화 decision
    call_recommendation: Callable[[SessionState, Decision, dict, list[str]], dict],  # 조건 -> 추천안
    call_response: Callable[[str, SessionState, dict, dict, list[dict], dict | None, list[dict], dict | None, dict, str], str],  # 확정 사실 + route -> 자연어 응답
):
    # Orchestrator → semantic repair → workers → recommendation → policy → Response Agent
    from langgraph.graph import END, START, StateGraph

    def interpret_decision(state: TurnState) -> dict:
        decision = call_decision(  # 전체 session.history를 포함한 session으로 발화 해석
            copy.deepcopy(state["session"]),
            state["user_text"],
            copy.deepcopy(state["robot_state"]),
            None,
        )
        return {"decision": decision}
    
    def apply_explicit_commit_gate(state: TurnState) -> dict:
        # Base Decision이 명시적 실행 표현을 놓친 경우 commit 하나만 False→True로 복구한다.
        # order/restriction/recommendation 등 기존 semantic은 같은 복사본에 그대로 보존한다.
        decision = copy.deepcopy(state["decision"])

        if (
            decision["commit"] is False
            and has_explicit_commit_intent(state["user_text"])
        ):
            decision["commit"] = True

        return {"decision": decision}

    def validate_and_repair(state: TurnState) -> dict:
        invalid_fields = find_invalid_order_fields(state["decision"], state["robot_state"])  # 현재 section에서 못 바꾸는 필드
        invalid_queries = find_invalid_queries(state["decision"])  # 지원하지 않는 query 타입
        route_errors = find_route_consistency_errors(state["decision"])  # route와 task semantic 조합 오류

        if not invalid_fields and not invalid_queries and not route_errors:
            return {}

        repair = {
            "invalid_fields": invalid_fields,  # 다시 해석해야 하는 주문 필드
            "invalid_queries": invalid_queries,  # 제거·수정해야 하는 query
            "route_errors": route_errors,  # task/general/mixed 의미와 semantic field의 모순
            "allowed_fields": allowed_order_fields(state["robot_state"]["section"]),  # 이번 section 허용 필드
            "previous_output": copy.deepcopy(state["decision"]),  # 첫 Decision 출력
            "instruction": "원래 발화에 대응되는 field와 route만 수정하고 임의 치환하지 않는다.",
        }
        repaired = call_decision(  # 같은 발화를 repair 정보와 함께 한 번만 재호출
            copy.deepcopy(state["session"]),
            state["user_text"],
            copy.deepcopy(state["robot_state"]),
            repair,
        )
        remaining = (  # 재호출 뒤에도 남은 invalid 항목
            find_invalid_order_fields(repaired, state["robot_state"])
            or find_invalid_queries(repaired)
            or find_route_consistency_errors(repaired)
        )

        if remaining:
            # 두 번째 Decision도 invalid면 첫 Decision의 안전한 문맥만 남기고 mutation semantic은 전부 폐기한다.
            repaired = build_safe_clarify_decision(
                state["decision"],
                state["user_text"],
            )

        return {"decision": repaired}
    
    def resolve_reference(state: TurnState) -> dict:
        # build_decision_inputs와 같은 deterministic helper를 다시 사용해 최종 Decision을 검증한다.
        # target 계산은 model inference 전에 이미 입력으로 전달됐고, 여기서는 결과 일치 여부만 확인한다.
        current_history_turn = len(state["session"]["history"]) // 2 + 1
        reference_context = build_reference_context(
            state["user_text"],
            state["session"]["dialogue_focus"],
            current_history_turn,
        )
        status = reference_context["status"]

        if status == "none":
            return {}

        decision = copy.deepcopy(state["decision"])

        if status in ("ambiguous", "missing", "stale"):
            # 후보 개수 부족, focus 없음, TTL 만료를 모두 안전한 clarification으로 보낸다.
            decision["understanding"] = "clarify"
            return {
                "decision": decision,
                "reference_context": {
                    "reason": "ambiguous_reference",
                    "targets": copy.deepcopy(reference_context["targets"]),
                },
            }

        resolved_targets = reference_context["targets"]

        if (
            decision["route"] != "general"
            and not reference_targets_are_supported(resolved_targets)
        ):
            # unsupported reference는 지원 메뉴로 치환하지 않고 기존 session mutation을 rollback한다.
            decision["understanding"] = "clarify"
            return {
                "decision": decision,
                "reference_context": {
                    "reason": "unsupported_reference",
                    "targets": copy.deepcopy(resolved_targets),
                },
            }

        if (
            decision["route"] != "general"
            and decision["understanding"] != "clarify"
            and not reference_targets_match_decision(decision, resolved_targets)
        ):
            # Python target과 Decision semantic이 다르면 임의 수정하지 않고 generic clarify로 차단한다.
            decision["understanding"] = "clarify"

        return {"decision": decision}

    def route_by_decision(state: TurnState) -> str:
        # general만 mutation 없는 전용 경로로 보내고 task/mixed는 기존 안전 pipeline을 그대로 사용한다.
        return "general" if state["decision"]["route"] == "general" else "task"

    def build_general_noop_policy(state: TurnState) -> dict:
        # 일반대화는 주문 state를 수정하지 않지만 reference가 모호하면 Response가 재질문할 수 있어야 한다.
        if state["decision"]["understanding"] == "clarify":
            policy = {
                "status": "clarify",
                "reason": "understanding",
                "execute": False,
                "conflicts": [],
            }
        else:
            policy = {
                "status": "pass",
                "reason": "general",
                "execute": False,
                "conflicts": [],
            }

        return {
            "session": copy.deepcopy(state["session"]),
            "recommendation_result": None,
            "policy": policy,
        }

    def resolve_confirmation(state: TurnState) -> dict:
        working, effective = resolve_pending_confirmation(  # pending 처리 후 기준 session과 유효 decision
            state["session"], state["decision"]
        )
        return {"working_session": working, "decision": effective}

    def apply_workers(state: TurnState) -> dict:
        candidate = apply_decision(state["working_session"], state["decision"])  # 정책 검사 전 후보
        return {"candidate_session": candidate}

    def run_recommendation(state: TurnState) -> dict:
        # 추천 생성과 기존 추천안 선택 모두 현재 restriction·물리 상태로 다시 검증한다.

        if state["decision"]["understanding"] == "clarify":
            # 대상이나 의미가 확정되지 않은 턴에서 추천 모델을 먼저 호출해도 결과는 policy에서 rollback된다.
            # 불필요한 inference와 recommendation_result 오염을 막고 기존 check_policy rollback은 그대로 사용한다.
            return {
                "candidate_session": copy.deepcopy(state["candidate_session"]),
                "recommendation_result": None,
            }

        request = state["decision"]["recommendation"]  # 이번 추천 요청
        action = request["action"]  # none/request/revise/select/cancel
        recommendation_state = state["candidate_session"]["recommendation"]  # 누적 추천 state
        accept_pending = action == "select" or (  # 기존 proposed 추천을 확정하는 경우
            action == "none"
            and state["decision"]["commit"]
            and recommendation_state["phase"] == "proposed"
        )

        if action not in ("request", "revise") and not accept_pending:
            candidate = merge_recommendation(  # cancel 또는 추천과 무관한 턴 반영
                state["candidate_session"], state["decision"], None
            )
            return {"candidate_session": candidate, "recommendation_result": None}

        if action in ("request", "revise"):
            scope = request["scope"]  # current/remaining/all 추천 범위
            allowed = recommendation_allowed_fields(scope, state["robot_state"]["section"])  # 추천 가능 필드
            raw_result = call_recommendation(  # 아직 policy 검증 전인 모델 추천값
                copy.deepcopy(state["candidate_session"]),
                copy.deepcopy(state["decision"]),
                copy.deepcopy(state["robot_state"]),
                allowed,
            )
        else:
            previous = recommendation_state["last_proposal"]  # 사용자가 선택한 최근 추천안

            if previous is None:
                decision = copy.deepcopy(state["decision"])
                decision["understanding"] = "clarify"
                return {"decision": decision, "recommendation_result": None}

            scope = previous["scope"]  # 기존 추천안 범위
            allowed = recommendation_allowed_fields(scope, state["robot_state"]["section"])  # 현재도 허용되는 필드
            raw_result = {
                "proposal": copy.deepcopy(previous["proposal"]),  # 기존 추천값 재사용
                "reason_tags": copy.deepcopy(previous["reason_tags"]),  # 기존 추천 근거 재사용
            }

        validation = validate_recommendation_proposal(  # restriction·section 기준으로 추천 필드 필터링
            state["candidate_session"],
            raw_result["proposal"],
            state["robot_state"],
            allowed,
        )
        recommendation_result = {
            "proposal": validation["proposal"],  # 살아남은 추천 주문값
            "reason_tags": copy.deepcopy(raw_result["reason_tags"]),  # 응답 설명용 근거
            "accepted_fields": validation["accepted_fields"],  # 적용 가능한 필드
            "rejected_fields": validation["rejected_fields"],  # 제한 때문에 거절된 필드
        }

        if not validation["accepted_fields"]:
            decision = copy.deepcopy(state["decision"])
            decision["understanding"] = "clarify"
            return {
                "decision": decision,
                "candidate_session": copy.deepcopy(state["candidate_session"]),
                "recommendation_result": recommendation_result,
            }

        candidate = merge_recommendation(  # 검증된 추천을 후보 session에 병합
            state["candidate_session"],
            state["decision"],
            recommendation_result,
        )
        return {
            "candidate_session": candidate,
            "recommendation_result": recommendation_result,
        }

    def check_policy(state: TurnState) -> dict:
        # 완료된 값의 직접 변경만 되돌리고 restriction 등 같은 턴의 안전한 변경은 보존한다.
        candidate = restore_protected_order_values(  # 이미 완료된 물리 단계 값은 원상 복구
            state["working_session"],
            state["candidate_session"],
            state["robot_state"],
        )
        physical_conflicts = find_new_physical_restriction_conflicts(  # 새 restriction과 완료값 충돌
            state["working_session"],
            candidate,
            state["robot_state"],
        )
        policy = evaluate_runtime_policy(  # 실행·질문·차단 여부 최종 판정
            candidate,
            state["decision"],
            state["robot_state"],
            physical_conflicts,
        )

        # reference가 없는 턴이나 이전 형식의 TurnState도 KeyError 없이 통과
        reference_context = state.get("reference_context")

        if (
            policy["status"] == "clarify"
            and policy["reason"] == "understanding"
            and reference_context is not None
        ):
            # evaluate_runtime_policy의 generic understanding을 resolver가 확정한 구체적 원인으로 교체한다.
            # 이 값은 Response 입력용 내부 policy이며 Decision JSON schema에는 영향을 주지 않는다.
            policy["reason"] = reference_context["reason"]
            policy["reference_targets"] = copy.deepcopy(
                reference_context["targets"]
            )

        if policy["status"] in ("pass", "warning"):
            next_session = copy.deepcopy(candidate)  # 후보 변경을 그대로 확정

        elif policy["status"] == "clarify" and policy["reason"] == "missing_order":
            # 현재까지 정상 선택된 값은 보존하고 부족한 값만 다시 질문한다.
            next_session = copy.deepcopy(candidate)  # 부족한 값 외의 선택은 유지

        elif policy["status"] == "hitl":
            next_session = copy.deepcopy(state["working_session"])  # 충돌 주문은 확인 전까지 보류
            next_session["pending_confirmation"] = {
                "type": "restriction_conflict",  # 다음 턴 confirmation 해석 기준
                "decision": copy.deepcopy(state["decision"]),  # 보류한 실행 요청
                "candidate_session": copy.deepcopy(candidate),  # 동의 시 되살릴 후보
                "conflicts": copy.deepcopy(policy["conflicts"]),  # 사용자에게 확인할 충돌
            }

        elif policy["status"] == "blocked" and policy["reason"] == "physical_state":
            # 완료값은 복구했으므로 같은 발화의 restriction·preference·미래 선택은 보존한다.
            next_session = copy.deepcopy(candidate)  # 완료값만 복구된 후보는 유지

        else:
            next_session = copy.deepcopy(state["working_session"])  # 안전한 기준 state로 rollback

        return {"session": next_session, "policy": policy}

    def generate_response(state: TurnState) -> dict:
        # 실제 session diff와 Python next_prompt를 만든 뒤 Response Agent에는 읽기 전용 사실만 전달한다.
        applied_changes = build_applied_changes(  # 이번 턴에 실제 session에 반영된 변경
            state["previous_session"],
            state["session"],
        )
        future_changes = build_future_changes(  # 아직 로봇이 실행하지 않은 미래 변경
            applied_changes,
            state["robot_state"],
        )
        if state["decision"]["route"] == "general":
            # 일반대화 뒤에 주문 section 질문을 자동으로 붙이지 않는다.
            next_prompt = None
        else:
            next_prompt = build_next_prompt(  # 다음에 물을 항목 또는 실행 안내
                state["session"],
                state["decision"],
                state["policy"],
                state["robot_state"],
            )

        # if state["policy"]["reason"] == "understanding":
        #     reply = "요청을 제대로 이해하지 못했습니다. 다시 말씀해 주세요."
        # elif state["policy"]["reason"] == "completed_restriction_conflict":
        #     reply = build_completed_restriction_warning(state["policy"]["conflicts"])
        # elif state["policy"]["reason"] == "physical_state":
        #     reply = build_completed_modification_warning(state["policy"]["blocked_changes"])
        # else:
        #     reply = call_response(  # 전체 대화 history는 넘기지 않고 현재 확정 사실만 넘김
        #         state["user_text"],
        #         copy.deepcopy(state["session"]),
        #         copy.deepcopy(state["policy"]),
        #         copy.deepcopy(applied_changes),
        #         copy.deepcopy(future_changes),
        #         copy.deepcopy(state["recommendation_result"]),
        #         copy.deepcopy(state["decision"]["queries"]),
        #         copy.deepcopy(next_prompt),
        #         copy.deepcopy(state["robot_state"]),
        #         state["decision"]["route"],
        #     )

        route = state["decision"]["route"]
        policy_reason = state["policy"]["reason"]
        reference_targets = state["policy"].get("reference_targets", [])

        if route == "task" and policy_reason == "ambiguous_reference":
            # 순수 task는 Base Gemma를 호출하지 않고 Python이 확정한 후보만 보여준다.
            # "또는"을 사용해서 메뉴 이름의 받침 여부와 관계없이 자연스럽게 연결한다.
            if reference_targets:
                targets_text = " 또는 ".join(reference_targets)
                reply = f"{targets_text} 중 어떤 것을 말씀하시는 건가요?"
            else:
                reply = "어떤 대상을 말씀하시는 건지 다시 알려주세요."

        elif route == "task" and policy_reason == "unsupported_reference":
            # unsupported 문자열은 지원 메뉴로 치환하지 않고 사용자가 말한 값을 그대로 안내한다.
            if reference_targets:
                targets_text = ", ".join(reference_targets)
                reply = f"말씀하신 {targets_text} 메뉴는 현재 제공하지 않아요."
            else:
                reply = "말씀하신 대상은 현재 제공하지 않는 메뉴예요."

        elif policy_reason == "understanding" and route != "mixed":
            # 기존 task/general generic clarify는 유지하지만 mixed는 아래 Response Agent로 보낸다.
            reply = "요청을 제대로 이해하지 못했습니다. 다시 말씀해 주세요."

        elif policy_reason == "completed_restriction_conflict":
            reply = build_completed_restriction_warning(
                state["policy"]["conflicts"]
            )

        elif policy_reason == "physical_state":
            reply = build_completed_modification_warning(
                state["policy"]["blocked_changes"]
            )

        else:
            # mixed + clarify도 이 경로를 사용한다.
            # rollback된 session과 Python policy를 읽고 task 재질문과 일반대화 답변을 함께 만든다.
            reply = call_response(
                state["user_text"],
                copy.deepcopy(state["session"]),
                copy.deepcopy(state["policy"]),
                copy.deepcopy(applied_changes),
                copy.deepcopy(future_changes),
                copy.deepcopy(state["recommendation_result"]),
                copy.deepcopy(state["decision"]["queries"]),
                copy.deepcopy(next_prompt),
                copy.deepcopy(state["robot_state"]),
                route,
            )

        session = copy.deepcopy(state["session"])  # history까지 기록할 최종 session 복사본

        # focus는 별도 턴을 만들지 않고, 이번 응답까지 포함될 기존 history 턴 번호를 그대로 사용한다.
        current_history_turn = len(session["history"]) // 2 + 1
        session["dialogue_focus"] = update_dialogue_focus(
            session["dialogue_focus"],
            state["decision"]["mentions"],
            current_history_turn,
        )

        action_event = build_turn_action_event(applied_changes, future_changes)  # 의미 있는 변경 요약

        if action_event is not None:
            session["action_history"].append(action_event)

        session["history"].append({"role": "user", "content": state["user_text"]})  # 다음 Decision 턴 기억
        session["history"].append({"role": "assistant", "content": reply})  # 다음 Decision 턴 기억

        return {
            "session": session,
            "reply": reply,
        }

    # graph wiring: 위 stage를 선언 순서 그대로 직렬 실행
    # graph wiring: route 검증 뒤 general은 no-op, task/mixed는 기존 pipeline으로 분기한다.
    graph = StateGraph(TurnState)
    graph.add_node("interpret_decision", interpret_decision)
    graph.add_node("explicit_commit_gate", apply_explicit_commit_gate)
    graph.add_node("validate_and_repair", validate_and_repair)
    graph.add_node("resolve_reference", resolve_reference)
    graph.add_node("build_general_noop_policy", build_general_noop_policy)
    graph.add_node("resolve_confirmation", resolve_confirmation)
    graph.add_node("apply_workers", apply_workers)
    graph.add_node("run_recommendation", run_recommendation)
    graph.add_node("check_policy", check_policy)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "interpret_decision")
    graph.add_edge("interpret_decision", "explicit_commit_gate")
    graph.add_edge("explicit_commit_gate", "validate_and_repair")
    graph.add_edge("validate_and_repair", "resolve_reference")
    graph.add_conditional_edges(
        "resolve_reference",
        route_by_decision,
        {
            "general": "build_general_noop_policy",
            "task": "resolve_confirmation",
        },
    )
    graph.add_edge("build_general_noop_policy", "generate_response")
    graph.add_edge("resolve_confirmation", "apply_workers")
    graph.add_edge("apply_workers", "run_recommendation")
    graph.add_edge("run_recommendation", "check_policy")
    graph.add_edge("check_policy", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()
