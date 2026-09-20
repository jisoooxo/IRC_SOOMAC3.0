# 2026-09-20 대기 중 추천안 부분 수정 가이드

참고:

- [`agent_v7.py`](../src/soomac_irc/soomac_irc/agent_v7.py) — Tool 결과 검증, 추천안 상태, 명시적 확인 parser이다.
- [`call_model_v7.py`](../src/soomac_irc/soomac_irc/call_model_v7.py) — 현재 Tool 프롬프트와 여덟 Tool JSON 계약이다.
- [`test_tool_transitions_v7.py`](../src/soomac_irc/test/test_tool_transitions_v7.py) — GPU 없이 실행하는 Tool·추천 상태 회귀 테스트이다.
- [`2026-09-20_work_log.md`](./2026-09-20_work_log.md) — 로그 원인과 완료·미구현 상태 기록이다.

→ **Tool 프롬프트와 여덟 Tool 형식은 유지한다. 추천 확인 중 모델이 `set_order`를 출력하더라도 변경 대상이 현재 추천안에 이미 있을 때만 추천안 복사본을 수정하며, 실제 주문은 건드리지 않는다.**

## 1. 확인 의도 parser에서 질문형을 먼저 차단한다

파일: `src/soomac_irc/soomac_irc/agent_v7.py`

현재 `classify_confirmation_intent()`의 부정어 검사 직후, `short_affirmatives` 정의 전에 다음 블록을 추가한다.

```python
    ambiguous_markers = (
        "진행해도될까",
        "시작해도될까",
        "진행할까",
        "시작할까",
        "진행해야하나",
        "시작해야하나",
        "해도돼",
        "해도될",
        "하는게맞",
        "괜찮을까",
        "생각해볼",
        "고민중",
    )

    if any(marker in normalized for marker in ambiguous_markers):
        return None
```

이 자리에서 살아있는 변수: `normalized`(`classify_confirmation_intent()` 36줄에서 정의)

판정 순서는 다음과 같이 된다.

```text
부정 → False
질문·주저 → None
명시적 긍정 → True
그 외 → None
```

## 2. 추천안 복사본만 수정하는 helper를 추가한다

파일: `src/soomac_irc/soomac_irc/agent_v7.py`

`_confirmation_block_reason()` 함수가 끝난 직후, `_section_items()` 함수 전에 다음 함수를 추가한다.

```python
def revise_pending_recommendation(
    order: dict,
    recommendation: dict,
    menu_changes: dict,
) -> dict:
    if recommendation.get("phase") != "confirming":
        return {"applied": False, "reason": "no_pending_proposal"}

    request = recommendation.get("request")
    recommended_changes = recommendation.get("recommended_changes")
    confirmed_sections = recommendation.get("confirmed_sections", [])

    if not isinstance(request, dict):
        return {"applied": False, "reason": "invalid_pending_request"}

    if not isinstance(recommended_changes, dict):
        return {"applied": False, "reason": "invalid_pending_changes"}

    if not isinstance(confirmed_sections, list):
        return {"applied": False, "reason": "invalid_confirmed_sections"}

    clean_changes, invalid = validate_menu_changes(menu_changes)

    if invalid or not clean_changes:
        return {
            "applied": False,
            "reason": "invalid_revision",
            "invalid": invalid,
        }

    proposal_keys = {
        field
        for field in ("sauce", "noodle_type", "noodle_portion")
        if field in recommended_changes
    }
    proposal_keys.update(
        f"toppings.{item}"
        for item in recommended_changes.get("toppings", {})
    )

    requested_keys = {
        field
        for field in ("sauce", "noodle_type", "noodle_portion")
        if field in clean_changes
    }
    requested_keys.update(
        f"toppings.{item}"
        for item in clean_changes.get("toppings", {})
    )

    # 현재 추천안에 없는 항목은 실제 주문 변경인지 추천안 수정인지 모호하므로 건드리지 않는다.
    if not requested_keys or not requested_keys.issubset(proposal_keys):
        return {
            "applied": False,
            "reason": "target_not_in_proposal",
            "requested_keys": sorted(requested_keys),
        }

    revised_changes = copy.deepcopy(recommended_changes)

    for field in ("sauce", "noodle_type", "noodle_portion"):
        if field in clean_changes:
            revised_changes[field] = clean_changes[field]

    if "toppings" in clean_changes:
        revised_toppings = revised_changes.setdefault("toppings", {})

        for item, amount in clean_changes["toppings"].items():
            if amount == "none":
                revised_toppings.pop(item, None)
            else:
                revised_toppings[item] = amount

        if not revised_toppings:
            revised_changes.pop("toppings", None)

    if not revised_changes:
        return {"applied": False, "reason": "proposal_became_empty"}

    recommendation_input = copy.deepcopy(request)
    recommendation_input["recommended_order"] = copy.deepcopy(revised_changes)
    clean, dropped, blocked = validate_recommended_order(
        order,
        recommendation_input,
    )

    # 부분적으로만 살아남으면 사용자가 의도하지 않은 추천안이 될 수 있으므로 자동 수정하지 않는다.
    if dropped or blocked or not clean:
        return {
            "applied": False,
            "reason": "revision_blocked",
            "clean": clean,
            "dropped": dropped,
            "blocked": blocked,
        }

    proposal = copy.deepcopy(order)
    applied, unchanged = apply_menu_changes(proposal, clean)

    if not applied:
        return {
            "applied": False,
            "reason": "revision_has_no_new_value",
            "unchanged": unchanged,
        }

    covered_sections = []

    for key in applied:
        if key in ("sauce", "noodle_type", "noodle_portion"):
            covered_section = "noodle"
        else:
            item = key.split(".", 1)[1]
            covered_section = physical_section_for("toppings", item)

        if covered_section not in covered_sections:
            covered_sections.append(covered_section)

    return {
        "applied": True,
        "reason": "revised",
        "clean": clean,
        "dropped": [],
        "blocked": [],
        "recommendation": {
            "phase": "confirming",
            "request": copy.deepcopy(request),
            "proposal": proposal,
            "recommended_changes": copy.deepcopy(clean),
            "covered_sections": covered_sections,
            "confirmed_sections": copy.deepcopy(confirmed_sections),
        },
    }
```

이 자리에서 살아있는 변수·함수: `copy`(1줄 import), `validate_menu_changes`·`validate_recommended_order`·`apply_menu_changes`·`physical_section_for`(7줄 import), 함수 인자 `order`·`recommendation`·`menu_changes`

중요: 이 helper에는 `order_after`를 넘겨도 내부에서 읽기만 한다. 실제 변경 함수에는 `proposal = copy.deepcopy(order)`만 넘긴다.

## 3. transaction 필드와 추천 확인 중 분기를 추가한다

파일: `src/soomac_irc/soomac_irc/agent_v7.py`

### 3-1. transaction 기본값

`transaction["recommendation_validation"]` 기본값에 다음 두 필드를 추가한다.

```python
            "proposal_revised": False,
            "revision_reason": None,
```

수정 후 전체 모양은 다음과 같다.

```python
        "recommendation_validation": {
            "clean": {},
            "dropped": [],
            "blocked": [],
            "proposal_created": False,
            "proposal_confirmed": False,
            "proposal_revised": False,
            "revision_reason": None,
            "needs_scope": False,
        },
```

이 자리에서 살아있는 변수: `transaction`(283줄에서 생성 중인 dictionary)

### 3-2. 추천 확인 중 `set_order` 방어

`recommend_order` 처리 블록이 끝나는 539줄 다음, `confirm_section` 추천 확정 블록 전에 추가한다.

```python
    # 추천 확인 중 발생한 메뉴 변경은 실제 주문이 아니라 대기 중 추천안만 수정한다.
    if (
        action in ("set_order", "set_order_and_confirm")
        and recommendation_after.get("phase") == "confirming"
    ):
        changes = copy.deepcopy(tool_call["changes"])
        menu_changes = {
            field: changes[field]
            for field in ("sauce", "noodle_type", "noodle_portion", "toppings")
            if field in changes
        }
        restriction_changes = changes.get("restriction_changes", [])

        # restriction 단독 요청은 기존 실제 안전 정책으로 내려보낸다.
        if restriction_changes and not menu_changes:
            pass

        # 메뉴 수정과 restriction이 섞이면 어느 한쪽도 반쯤 commit하지 않고 재질문한다.
        elif restriction_changes:
            transaction["recommendation_validation"]["revision_reason"] = (
                "mixed_menu_and_restriction"
            )

            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "transaction": transaction,
            }

        elif menu_changes:
            revision = revise_pending_recommendation(
                order_after,
                recommendation_after,
                menu_changes,
            )
            transaction["recommendation_validation"].update({
                "clean": copy.deepcopy(revision.get("clean", {})),
                "dropped": copy.deepcopy(revision.get("dropped", [])),
                "blocked": copy.deepcopy(revision.get("blocked", [])),
                "proposal_revised": revision["applied"],
                "revision_reason": revision["reason"],
            })

            if revision["applied"]:
                recommendation_after = revision["recommendation"]

            # set_order_and_confirm으로 오분류됐어도 같은 턴에 실행하지 않는다.
            # 수정된 추천안을 다시 읽고 다음 턴의 confirm_section에서 확정한다.
            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "transaction": transaction,
            }
```

이 자리에서 살아있는 변수: `action`(281줄), `recommendation_after`(270줄), `tool_call`(280줄), `order_after`(267줄), `execution_after`(268줄), `transaction`(283줄), `copy`(1줄 import), `revise_pending_recommendation`(2번 스니펫에서 정의)

이 분기는 일반 `set_order`의 `apply_menu_changes(order_after, editable)`보다 먼저 `return`한다. 따라서 추천안 수정 발화가 실제 주문 변경 경로까지 내려가지 않는다.

## 4. 추천안 수정 결과 응답을 추가한다

파일: `src/soomac_irc/soomac_irc/agent_v7.py`

`build_policy_reply()`에서 다음 줄 바로 뒤에 코드를 추가한다.

```python
    recommendation = transaction["recommendation_validation"]
```

추가할 코드:

```python
    if (
        action in ("set_order", "set_order_and_confirm")
        and recommendation.get("proposal_revised")
    ):
        clean = recommendation["clean"]
        amount_labels = {"low": "적게", "normal": "보통", "high": "많이"}
        revised_parts = []

        if clean.get("sauce"):
            revised_parts.append(f"{clean['sauce']} 소스")

        if clean.get("noodle_type"):
            revised_parts.append(clean["noodle_type"])

        if clean.get("noodle_portion"):
            revised_parts.append(f"면 양 {amount_labels[clean['noodle_portion']]}")

        for item, amount in clean.get("toppings", {}).items():
            revised_parts.append(f"{item} {amount_labels[amount]}")

        messages.append(
            f"추천안을 {', '.join(revised_parts)} 조합으로 수정했어요. "
            "이대로 할까요?"
        )

    elif (
        action in ("set_order", "set_order_and_confirm")
        and recommendation.get("revision_reason")
    ):
        if recommendation["revision_reason"] == "mixed_menu_and_restriction":
            messages.append(
                "추천안 수정과 식이·알레르기 조건을 한 번에 처리하지 않았어요. "
                "안전 조건을 먼저 따로 말씀해 주세요."
            )
        else:
            messages.append(
                "현재 추천안에서 어떤 항목을 바꿀지 정확히 확인하지 못했어요. "
                "추천안에 있는 재료와 변경 내용을 다시 말씀해 주세요."
            )

    elif action == "recommend_order":
```

그리고 바로 아래에 원래 있던 다음 줄은 삭제한다.

```python
    if action == "recommend_order":
```

즉 기존 `if action == "recommend_order":` 블록을 위 스니펫의 마지막 `elif action == "recommend_order":`에 그대로 연결한다.

이 자리에서 살아있는 변수: `action`(같은 함수 1239줄), `recommendation`(스니펫 바로 위에서 정의), `messages`(`build_policy_reply()` 시작부에서 정의)

응답 끝에 `?`가 있으므로 `llm_node_v7._build_turn_reply()`의 기존 질문 방어가 다음 단계 안내를 중복으로 붙이지 않는다.

## 5. 회귀 테스트를 추가한다

파일: `src/soomac_irc/test/test_tool_transitions_v7.py`

파일 끝에 다음 테스트를 추가한다.

```python
def _pending_full_recommendation():
    return {
        "phase": "confirming",
        "request": {
            "scope": "remaining",
            "sections": [],
            "targets": [],
            "excluded": ["버섯"],
            "preferences": [],
            "different": False,
        },
        "proposal": {
            "sauce": "토마토",
            "noodle_type": "얇은면",
            "noodle_portion": "normal",
            "toppings": {
                "양파": "normal",
                "소시지": "normal",
                "치즈": "normal",
            },
            "restrictions": [],
        },
        "recommended_changes": {
            "sauce": "토마토",
            "noodle_type": "얇은면",
            "noodle_portion": "normal",
            "toppings": {
                "양파": "normal",
                "소시지": "normal",
                "치즈": "normal",
            },
        },
        "covered_sections": ["noodle", "veggie", "meat", "extra"],
        "confirmed_sections": [],
    }


def test_pending_recommendation_removes_only_requested_item():
    order = new_order()
    result = validate_transaction(_state(
        "set_order",
        {"toppings": {"소시지": "none"}},
        user_text="소시지만 빼줘",
        order=order,
        recommendation=_pending_full_recommendation(),
    ))

    assert result["order"] == order
    assert result["recommendation"]["recommended_changes"]["toppings"] == {
        "양파": "normal",
        "치즈": "normal",
    }
    assert result["transaction"]["recommendation_validation"][
        "proposal_revised"
    ] is True


def test_pending_recommendation_does_not_redirect_unknown_target_to_order():
    order = new_order()
    recommendation = _pending_full_recommendation()
    result = validate_transaction(_state(
        "set_order",
        {"toppings": {"게살": "high"}},
        user_text="게살은 많이",
        order=order,
        recommendation=recommendation,
    ))

    assert result["order"] == order
    assert result["recommendation"] == recommendation
    assert result["transaction"]["recommendation_validation"][
        "revision_reason"
    ] == "target_not_in_proposal"


def test_pending_revision_and_confirm_is_forced_to_two_turn_confirmation():
    order = new_order()
    result = validate_transaction(_state(
        "set_order_and_confirm",
        {"toppings": {"소시지": "none"}},
        user_text="소시지 빼고 진행해",
        order=order,
        recommendation=_pending_full_recommendation(),
    ))

    assert result["order"] == order
    assert result["recommendation"]["phase"] == "confirming"
    assert result["transaction"]["confirm_validation"]["requested"] is False


def test_confirmation_parser_treats_question_form_as_ambiguous():
    assert classify_confirmation_intent(
        "소시지 빼고 진행해도 될까",
        "start",
    ) is None
    assert classify_confirmation_intent(
        "소시지 빼되 진행하지 마",
        "start",
    ) is False
    assert classify_confirmation_intent(
        "소시지 빼고 진행해",
        "start",
    ) is True
```

이 자리에서 살아있는 변수·함수: `copy`(테스트 파일 1줄 import), `validate_transaction`·`classify_confirmation_intent`(3~7줄 import), `new_order`(10줄 import), `_state`(25줄 정의)

실행 명령:

```bash
cd /home/roma/IRC_SOOMAC3.0
PYTHONPATH=src/soomac_irc pytest -q \
  src/soomac_irc/test/test_tool_transitions_v7.py \
  src/soomac_irc/test/test_response_followup_v7.py
```

현재 기본 system Python에는 `langgraph`가 없고 `gemma4_env`에는 `pytest`가 없어 이 환경 구성 그대로는 위 테스트가 실행되지 않는다. 두 패키지가 함께 있는 프로젝트 실행 환경에서 실행해야 한다.

## 적용 후 동작

```text
추천안: 토마토, 얇은면, 양파, 소시지, 치즈
사용자: 소시지만 빼줘
결과: 실제 주문 유지, 추천안만 토마토·얇은면·양파·치즈로 변경, 재확인
```

```text
추천안: 야채 조합
사용자: 게살은 많이
결과: 실제 주문과 추천안 모두 유지, 대상을 다시 질문
```

```text
사용자: 소시지 빼고 진행해
결과: 추천안만 수정하고 재확인. 같은 턴에는 로봇을 시작하지 않음
```

Cause: 일반 `set_order`는 현재 section 전용이 아니며 실제 `order_after`를 수정한다.

Effect: 추천 확인 중 관측된 오분류를 Python에서 좁게 흡수하면서 실제 주문·Tool 프롬프트·LoRA 입력 계약을 유지한다.

## 용어 정리

- **대기 중 추천안(pending proposal)**: 모델이 추천했지만 사용자가 아직 확정하지 않아 실제 주문에 들어가지 않은 조합이다.
- **추천안 복사본(proposal copy)**: 실제 주문과 분리해 수정·검증하는 임시 주문 객체이다.
- **질문형 확인(question-form confirmation)**: `진행해도 될까`처럼 진행 단어가 있지만 실행 명령은 아닌 발화이다.
- **좁은 방어(narrow guard)**: 모든 `set_order` 의미를 바꾸지 않고 확인 중 추천안의 기존 항목만 수정하는 조건부 처리이다.
