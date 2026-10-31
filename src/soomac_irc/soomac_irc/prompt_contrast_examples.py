"""독립된 prompt 시연. 실제 사용자 입력을 처리하거나 응답을 선택하지 않는다.

기존 시연은 그대로 보존하고 같은 상태에서 의미만 다른 대조를 덧붙인다.
builder는 정적인 시연 입력을 계약 모양으로 만들 때만 사용한다.
"""
import copy

from soomac_irc.agent_contract import new_session_state
from soomac_irc.response_contract import build_response_model_input


def extend_contrast_examples(examples):
    def robot(section):
        return {"section": section, "task_queue": [], "active_task": None,
                "completed_tasks": [], "robot_started": False}

    def decision(identifier, session, section, message, output):
        examples["decision"].append({
            "id": identifier,
            "input": {"recent_history": copy.deepcopy(session["history"]),
                      "order": copy.deepcopy(session["order"]),
                      "preferences": copy.deepcopy(session["preferences"]),
                      "pending": copy.deepcopy(session["pending"]),
                      "robot_state": robot(section), "message": message},
            "output": output,
        })

    cream = new_session_state()
    cream["order"].update(noodle_type="얇은면", noodle_portion="normal")
    cream["history"] = [
        {"role": "user", "content": "얇은면 보통으로 고르고 소스는 느끼한 걸로 추천해줘."},
        {"role": "assistant", "content": "소스는 크림을 추천드려요. 선택할까요?"},
    ]
    cream["pending"] = {
        "type": "recommendation",
        "candidate": {"sauce": "크림", "noodle_type": None, "noodle_portion": None, "toppings": {}},
        "reason_tags": ["느끼한 소스 요청"],
    }
    decision("C01", cream, "noodle", "크림으로 선택할게.",
             {"route": "task", "confirmation": "accept"})
    decision("C02", cream, "noodle", "크림으로 선택하고 바로 진행해줘.",
             {"route": "task", "confirmation": "accept", "commit": True})

    veggie = new_session_state()
    veggie["order"].update(sauce="크림", noodle_type="얇은면", noodle_portion="normal")
    veggie["order"]["toppings"] = {"양파": "low", "버섯": "low"}
    veggie["history"] = [
        {"role": "user", "content": "양파 많이, 버섯 조금 골라줘."},
        {"role": "assistant", "content": "채소 양을 선택했어요."},
        {"role": "user", "content": "아 그냥 둘 다 조금만 먹고 싶어."},
        {"role": "assistant", "content": "양파와 버섯 모두 적게 선택했어요. 담을까요?"},
    ]
    veggie["pending"] = {"type": "execution", "source": "current_update",
                         "section": "veggie", "targets": ["양파", "버섯"]}
    decision("C03", veggie, "veggie", "버섯만 보통 양으로 바꿔줘.",
             {"route": "task", "order": {"toppings": {"버섯": "normal"}}})
    decision("C04", veggie, "veggie", "버섯만 보통 양으로 바꿔서 바로 담아줘.",
             {"route": "task", "order": {"toppings": {"버섯": "normal"}}, "commit": True})

    meat = copy.deepcopy(veggie)
    meat["pending"] = None
    meat["order"]["toppings"]["버섯"] = "normal"
    meat["history"] += [
        {"role": "user", "content": "게살 말고 소시지를 먹을 생각이야. 아직 고르지는 말고."},
        {"role": "assistant", "content": "육류 선택은 기다릴게요."},
    ]
    allergy = {"action": "add", "target": "게살", "reason": "allergy"}
    for identifier, message, commit in (
        ("C05", "게살 알레르기가 있어서 소시지만 많이 골라줘.", False),
        ("C06", "게살 알레르기가 있어서 소시지만 많이 골라서 바로 담아줘.", True),
        ("C07", "게살 알레르기가 있으니 소시지만 많이 골라줘. 바로 담지는 마.", False),
    ):
        output = {"route": "task", "order": {"toppings": {"소시지": "high"}},
                  "restrictions": [copy.deepcopy(allergy)]}
        if commit:
            output["commit"] = True
        decision(identifier, meat, "meat", message, output)

    def response(identifier, section, toppings, history, text, changes, reply):
        session = new_session_state()
        session["order"].update(sauce="크림", noodle_type="얇은면", noodle_portion="normal")
        session["order"]["toppings"] = toppings
        session["history"] = history
        session["pending"] = {"type": "execution", "source": "current_update",
                              "section": section, "targets": list(toppings)}
        applied = {"order_changes": changes, "restriction_added": [], "restriction_removed": [],
                   "preference_added": [], "preference_removed": []}
        policy = {"status": "pass", "reason": "state_update_only", "execute": False, "conflicts": [],
                  "restriction_blocked": [], "restriction_unchanged": []}
        next_prompt = copy.deepcopy(session["pending"])
        next_prompt["items"] = {}
        data = build_response_model_input(text, session, policy, applied, [], None, next_prompt, robot(section), "task")
        examples["task"].append({"id": identifier, "input": data, "output": {"reply": reply}})

    response("S10", "meat", {"소시지": "high"}, [], "소시지만 많이 골라줘.",
             {"toppings.소시지": "high"}, "소시지를 많이 선택했어요. 이대로 담을까요?")
    response("S11", "veggie", {"양파": "low", "버섯": "normal"},
             copy.deepcopy(veggie["history"]), "버섯만 보통 양으로 바꿔줘.",
             {"toppings.버섯": "normal"}, "버섯을 보통 양으로 바꿨어요. 양파는 적은 양 그대로예요. 이대로 담을까요?")
    response("S12", "veggie", {"양파": "low", "버섯": "normal"},
             [{"role": "user", "content": "양파 조금, 버섯 보통으로 골라줘."},
              {"role": "assistant", "content": "그렇게 선택했어요. 담을까요?"}],
             "버섯은 보통 양으로 골라둔 거 맞지?", {},
             "네, 버섯은 보통 양으로 선택돼 있어요. 양파는 적은 양 그대로예요. 이대로 담을까요?")
