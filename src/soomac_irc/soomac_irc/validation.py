from soomac_irc.order import SECTION_ORDER


def validate_result_fields(result: dict) -> None:
    # Agent 결과의 최상위 필드 형식만 검사한다. 입력은 변경하지 않는다.
    # 호출자: validate_agent_result. 성공 시 None, 실패 시 기존 ValueError.
    if not isinstance(result, dict):
        raise ValueError("agent 결과가 dict가 아님")

    if not isinstance(result.get("order"), dict):
        raise ValueError("agent 결과 order가 dict가 아님")
    if not isinstance(result.get("execution"), dict):
        raise ValueError("agent 결과 execution이 dict가 아님")
    if not isinstance(result.get("recommendation"), dict):
        raise ValueError("agent 결과 recommendation이 dict가 아님")
    if not isinstance(result.get("turn_result"), dict):
        raise ValueError("agent 결과 turn_result가 dict가 아님")

    policy_reply = result.get("policy_reply")
    if policy_reply is not None and not isinstance(policy_reply, str):
        raise ValueError("agent 결과 policy_reply가 문자열 또는 None이 아님")


def validate_execution_state(execution: dict) -> None:
    # dict임이 확인된 실행 상태의 기존 필드 형식을 검사한다.
    # 호출자: validate_agent_result. 상태 변경 없음, 성공 시 None.
    if execution.get("section") not in SECTION_ORDER:
        raise ValueError(f"agent 결과 section 이상함 : {execution.get('section')}")
    if not isinstance(execution.get("selected"), list):
        raise ValueError("agent 결과 selected가 list가 아님")
    if not isinstance(execution.get("task_queue"), list):
        raise ValueError("agent 결과 task_queue가 list가 아님")
    if execution.get("active_task") is not None and not isinstance(execution["active_task"], dict):
        raise ValueError("agent 결과 active_task가 dict 또는 None이 아님")
    if not isinstance(execution.get("completed_tasks"), list):
        raise ValueError("agent 결과 completed_tasks가 list가 아님")
    if not isinstance(execution.get("skipped_sections"), dict):
        raise ValueError("agent 결과 skipped_sections가 dict가 아님")
    if type(execution.get("robot_started")) is not bool:
        raise ValueError("agent 결과 robot_started가 bool이 아님")


def validate_agent_result(result: dict) -> None:
    # 상태 commit 전에 Agent 결과의 필수 구조를 검사한다.
    # Node의 일반 턴과 단계 제외 확인 경로에서 각각 한 번 호출한다.
    validate_result_fields(result)
    validate_execution_state(result["execution"])
