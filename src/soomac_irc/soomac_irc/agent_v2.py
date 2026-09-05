import json  # dict를 모델에게 넣을 JSON 문자열로 변환
import copy
from typing import TypedDict  # LangGraph State 안에 들어갈 key와 타입 설명

from langgraph.graph import END, START, StateGraph

from call_model_v2 import TOOL_SYSTEM
from order_v2 import SECTION_ORDER, apply_delta, enforce_constraints, missing_required, validate_delta, build_section_plan



# 프롬프트에 넣을 이전 대화 수
# user와 assistant가 한 쌍이라 실제 메시지는 아래에서 2배로 자름
HISTORY_TURNS = 6


class ToolCall(TypedDict): # Tool Calling 형식
    call_id: str
    name: str
    changes: dict # 이번 발화로 주문에서 바꿀 값


class AgentState(TypedDict, total=False):
    user_text: str # input text만 저장
    order: dict # 주문 상태
    section: str # 현재 섹션
    history: list[dict] # 멀티턴 [{user}, {assistant}] 대화
    action_history: list[dict] # 최근 Tool 선택, 그리고 파이썬 예외처리 결과
    completed_tasks: list[dict] # 로봇 작업 성공까지 확인된 작업

    # 거절, 추천, 남은 메뉴 안내, 건너뛴 섹션 관련 상태
    refusal_prompted_section: str | None
    recommendation_state: dict
    last_recommendation: dict
    remaining_item_prompted_section: str | None
    skipped_sections: list[str]

    # choose_action에서 만드는 값
    user_msg: dict # 모델에 실제로 전달한 유저 입력·주문 상태·섹션
    tool_call: ToolCall | None
    parser_status: str

    # execute_action에서 만드는 값
    action: str
    clean: dict
    dropped: list[str] # validate_delta가 버린 값(처음부터 주문에 X)
    changed: list[str] # apply_delta가 사용자 요청으로 실제 바꾼 값
    blocked: list[str] # enforce_constraints가 제약 때문에 다시 뺀 값
    missing: list[str]
    facts: dict

    # write_reply에서 만드는 값
    reply: str




FALLBACK_REPLIES = {
    "set_order": "주문을 반영했어요.",
    "set_order_and_confirm": "주문 변경과 현재 선택을 확인했어요.",
    "confirm_section": "현재 선택을 확인했어요.",
    "cancel_order": "주문을 취소할게요.",
    "respond": "네, 말씀해 주세요.",
    "describe_scene": "카메라 화면을 확인할게요.",
    "error": "요청을 이해하지 못했어요. 다시 말씀해 주세요.",
    "refuse_section": "현재 단계의 재료 제외 요청을 확인했어요.",
    "recommend_order": "추천할 주문 범위를 확인할게요.",
}


def build_graph(call_model, generate_reply):
    # call_model과 generate_reply는 llm_node가 모델 로드 후 넣는다

    def choose_action(state: AgentState) -> dict:


        """
        state = {
            "user_text": "크림 넓적면 보통에 치즈 많이",
            "order": 현재_주문,
            "section": "noodle",
            "history": 이전_대화,
        } -> 입력값


        """
        # 전체 history를 전부 넣지 않고 최근 대화만 사용
        history = state.get("history", [])
        # 전체 행동 기록도 전부 넣지 않고 최근 결과만 사용
        action_history = state.get("action_history", [])
        recent_actions = action_history[-HISTORY_TURNS:]
        
        recent_history = history[-(HISTORY_TURNS * 2):]

        section = state["section"]


        # 현재 섹션에서 지켜야 하는 설명
        section_rule = ""

        if section == "noodle":
            section_rule = (
                "현재는 면 단계이다. sauce, noodle_type, noodle_portion을 선택할 수 있다. "
                "치즈와 페퍼론치노는 미리 추가할 수 있다."
            ) # 모델한테 강제하는 설명 나중에 추가해야할 수도 ㅇㅇ
        elif section == "veggie":
            section_rule = (
                "현재는 야채 단계이다. toppings에는 양파와 버섯만 새로 선택할 수 있다. "
                "일반 확정에는 양파와 버섯 중 최소 하나가 필요하다. "
                "양파와 버섯을 모두 못 먹거나 모두 싫다고 명확히 말하면 refuse_section을 선택한다."
            )

        elif section == "meat":
            section_rule = (
                "현재는 육류 단계이다. toppings에는 소시지와 게살만 새로 선택할 수 있다. "
                "일반 확정에는 소시지와 게살 중 최소 하나가 필요하다. "
                "소시지와 게살을 모두 못 먹거나 모두 싫다고 명확히 말하면 refuse_section을 선택한다."
            )

        elif section == "extra":
            section_rule = (
                "현재는 추가 재료 단계이다. toppings에는 치즈와 페퍼론치노만 새로 선택할 수 있다. "
                "추가 재료는 선택 사항이다. 둘 다 못 먹거나 둘 다 싫다고 명확히 말하면 refuse_section을 선택한다."
            )

        elif section == "sauce":
            section_rule = "현재는 선택한 소스를 로봇이 투입하는 단계이다. 새로운 주문값을 선택하는 단계가 아니다."

        elif section == "lid":
            section_rule = "현재는 포장 확인 단계이다. 이미 담은 주문 재료는 변경하지 않는다."

        # 현재 section을 끝내기 전에 필요한 값도 모델에게 알려줌
        missing = missing_required(state["order"], section)

        # 모델이 현재 주문과 섹션, 완료 상태 알수 있도록
        user_msg = {
            "role": "user",
            "content": json.dumps(
                {
                    "order": state["order"],
                    "section": section,
                    "section_order": SECTION_ORDER,
                    "section_rule": section_rule,
                    "missing": missing,
                    "completed_tasks": state.get("completed_tasks", []),
                    "action_history": recent_actions,
                    "flow_state": {
                        "refusal_prompted_section": state.get("refusal_prompted_section"),
                        "recommendation_state": state.get("recommendation_state", {}),
                        "last_recommendation": state.get("last_recommendation", {}),
                        "remaining_item_prompted_section": state.get("remaining_item_prompted_section"),
                        "skipped_sections": sorted(state.get("skipped_sections") or []),
                    },
                    "message": state["user_text"],
                },
                ensure_ascii=False,
            ),
        }


        """
        {
        "order": {
            "sauce": null,
            "noodle_type": null,
            "noodle_portion": null,
            "toppings": {},
            "constraints": []
        },
        "section": "noodle",
        "section_rule": "현재는 면 단계이다...",
        "missing": [
            "sauce",
            "noodle_type",
            "noodle_portion"
        ],
        "message": "크림 넓적면 보통에 치즈 많이"
        }

        기존 state를 이런식으로 반환한다.

        """

        messages = [
            {"role": "system", "content": TOOL_SYSTEM},
            *recent_history,
            user_msg,
        ]

        # 여기서는 행동만 고르고, 실제 주문 변경은 execute action에서 한다

        tool_call = call_model(messages)

        """
        tool_call = {
            "call_id": "abc",
            "name": "set_order",
            "changes": {
                "sauce": "크림",
                "noodle_type": "넓적면",
                "noodle_portion": "normal",
                "toppings": {
                    "치즈": "high",
                },
            },
        }

        이런식으로 불러냄.

        즉 지금 손님의 말은 무슨 행동? 그리고 주문에서 무엇을 바꾼다는거지? 라는 느낌
        """

        return {
            "user_msg": user_msg,
            "tool_call": tool_call,
            "parser_status": "ok" if tool_call is not None else "invalid",
        }



    def execute_action(state: AgentState) -> dict:
        # 실제 주문은 이 턴의 모든 검증이 끝날 때까지 건드리지 않는다.
        order_before = state["order"]
        order = copy.deepcopy(order_before)
        section = state["section"]
        tool_call = state.get("tool_call")
        completed_classes = [task["class"] for task in state.get("completed_tasks", [])]

        action = "error"
        clean = {}
        dropped = []
        changed = []
        blocked = []

        if tool_call is not None:
            action = tool_call["name"]

            if action in ("set_order", "set_order_and_confirm"):
                #changes = tool_call["changes"]
                # 아래 중복 제거 과정에서 dict를 수정하므로 원본 Tool 출력은 보존한다.
                changes = copy.deepcopy(tool_call["changes"])

                # 모델이 기존 주문을 changes에 다시 복사해도 이번 변경으로 처리하지 않는다.

                for field in ["sauce", "noodle_type", "noodle_portion"]:
                    if field in changes and changes[field] == order[field]:
                        del changes[field]



                if isinstance(changes.get("toppings"), dict):
                    new_toppings = {}

                    for topping, amount in changes["toppings"].items():
                        if order["toppings"].get(topping) != amount:
                            new_toppings[topping] = amount

                    if new_toppings:
                        changes["toppings"] = new_toppings
                    else:
                        del changes["toppings"]

                # 이미 로봇이 담은 재료는 이후 발화로 변경하거나 제거하지 않는다.
                completed_dropped = []

                if order["noodle_type"] in completed_classes:
                    for field in ("noodle_type", "noodle_portion"):
                        if field in changes:
                            completed_dropped.append(f"completed.{field}={changes.pop(field)}")

                if order["sauce"] in completed_classes and "sauce" in changes:
                    completed_dropped.append(f"completed.sauce={changes.pop('sauce')}")

                if isinstance(changes.get("toppings"), dict):
                    for topping in list(changes["toppings"]):
                        if topping in completed_classes:
                            completed_dropped.append(f"completed.toppings.{topping}={changes['toppings'].pop(topping)}")

                    if not changes["toppings"]:
                        del changes["toppings"]

                clean, dropped = validate_delta(changes, section)
                dropped = completed_dropped + dropped

                # changed는 사용자 요청으로 실제 바뀐 항목
                changed = apply_delta(order, clean)

                # blocked는 제약 때문에 다시 빠진 항목
                blocked = enforce_constraints(order)

                # 새 제약이 이미 담은 재료와 충돌해도 완료된 재료는 되돌리지 않는다.
                for completed_class in completed_classes:
                    if completed_class in order_before["toppings"] and completed_class not in order["toppings"]:
                        order["toppings"][completed_class] = order_before["toppings"][completed_class]
                        blocked_field = f"toppings.{completed_class}"

                        if blocked_field in blocked:
                            blocked.remove(blocked_field)

                        dropped.append(f"completed.constraint_conflict.{blocked_field}")

                completed_sauce = order_before["sauce"]

                if completed_sauce in completed_classes and order["sauce"] != completed_sauce:
                    order["sauce"] = completed_sauce

                    if "sauce" in blocked:
                        blocked.remove("sauce")

                    dropped.append(f"completed.constraint_conflict.sauce={completed_sauce}")

                for blocked_field in blocked:
                    if blocked_field in changed:
                        changed.remove(blocked_field)

            elif action in ("refuse_section", "recommend_order"):
                # call_model_v2.parse_tool_call()을 통과한 새 Tool 전용 값이다.
                # 실제 섹션 스킵과 추천안 반영은 llm_node가 Python 상태를 확인한 뒤 결정한다.
                clean = copy.deepcopy(tool_call["changes"])

        # confirm_section일 때 llm_node가 다음 단계로 갈 수 있는지 확인할 때 사용
        missing = missing_required(order, section)

        # 현재 섹션을 확정하면 실제로 실행할 로봇 작업
        # 아직 confirm_section 전이면 예정 작업일 뿐 실제로 담기 시작한 것은 아님
        section_tasks = build_section_plan(order, section)

        # 모델이 담았다고 거짓말하지 못하게 검증·반영·제약 적용이 끝난 값만 Reply로 보냄 ㅇㅇ
        # changed는 들어간 것, blocked는 다시 빠진 것, dropped는 처음부터 못 들어간 것
        facts = {
            "history": state.get("history", [])[-(HISTORY_TURNS * 2):],
            "action_history": state.get("action_history", [])[-HISTORY_TURNS:],
            "flow_state": {
                "refusal_prompted_section": state.get("refusal_prompted_section"),
                "recommendation_state": state.get("recommendation_state", {}),
                "last_recommendation": state.get("last_recommendation", {}),
                "remaining_item_prompted_section": state.get("remaining_item_prompted_section"),
                "skipped_sections": sorted(state.get("skipped_sections") or []),
            },
            "user_text": state["user_text"],
            "action": action,
            "section": section,
            "order": order,
            "changed": changed,
            "blocked": blocked,
            "dropped": dropped,
            "missing": missing,
            "section_tasks": section_tasks, 
            "completed_tasks": state.get("completed_tasks", []),
        } # 아래에 다시 들어감

        return {
            "order": order,
            "action": action,
            "clean": clean, # 검사 통과한 변경값
            "dropped": dropped, # 아예 잘못돼서 처음부터 1차 필터에서 걸러진 값
            "changed": changed, # 실제 주문에 반영된 항목
            "blocked": blocked, # 알러지 등으로 인해 빠진 항목
            "missing": missing, # 섹션에 필요한 선택
            "facts": facts,
        }

    def write_reply(state: AgentState) -> dict:
        action = state["action"]

        if action == "error":
            return {"reply": FALLBACK_REPLIES["error"]}

        # 장면 묘사는 llm_node가 최신 카메라 이미지를 받은 뒤 만든다.
        # 추천 답변도 Python 검증을 통과한 추천값으로 llm_node가 만든다.
        # 여기서 Reply를 만들면 화면 또는 검증 전 추천값을 사용하게 된다.
        if action in ("describe_scene", "recommend_order"):
            return {"reply": FALLBACK_REPLIES[action]}
    

        # 정상 변경은 기존처럼 모델이 자연스럽게 답변
        # Reply 생성 실패가 주문 상태까지 터뜨리지 않게 Python 문장으로 대체
        try:
            reply = generate_reply(state["facts"])
        except Exception:
            reply = ""

        if not isinstance(reply, str) or not reply.strip():
            
            if action in ("set_order", "set_order_and_confirm"):
                if state["changed"] and (state["blocked"] or state["dropped"]):
                    reply = "가능한 주문 변경만 반영했고, 반영할 수 없는 내용은 제외했어요."

                elif state["changed"]:
                    reply = "요청한 주문 변경을 반영했어요."

                elif state["blocked"]:
                    reply = "식이 제약과 충돌한 항목은 주문에 반영하지 않았어요."

                elif state["dropped"]:
                    reply = "현재 단계에서 반영할 수 없는 요청이에요."

                else:
                    reply = "주문에서 새로 변경된 내용은 없어요."

            elif action == "confirm_section":
                task_names = [
                    task["class"]
                    for task in state["facts"]["section_tasks"]
                ]

                if state["missing"]:
                    reply = "아직 필요한 선택이 남아 있어요."

                elif len(task_names) == 1:
                    reply = f"{task_names[0]} 선택을 확인했어요."

                elif len(task_names) >= 2:
                    task_text = f"{', '.join(task_names[:-1])}와 {task_names[-1]}"
                    reply = f"{task_text} 선택을 확인했어요."

                else:
                    reply = "현재 선택을 확인했어요."

            elif action == "respond":
                reply = "요청을 정확히 이해하지 못했어요. 다시 말씀해 주세요."

            else:
                reply = FALLBACK_REPLIES.get(action, FALLBACK_REPLIES["error"])

        return {"reply": reply.strip()}

    # 한 턴은 세 단계만 순서대로 실행
    # confirm_section의 실제 섹션 이동과 cancel_order의 전체 초기화는 llm node가 담당
    graph = StateGraph(AgentState)

    graph.add_node("choose_action", choose_action)
    graph.add_node("execute_action", execute_action)
    graph.add_node("write_reply", write_reply)

    graph.add_edge(START, "choose_action")
    graph.add_edge("choose_action", "execute_action")
    graph.add_edge("execute_action", "write_reply")
    graph.add_edge("write_reply", END)


    # compile = 그래프 연결을 검사하고 실제 invoke 가능한 객체로 만드는 단계
    return graph.compile()



"""
한 턴에 지금 choose action -> execute action -> write reply 이렇게 3개의 노드로 돌아간다

"""
