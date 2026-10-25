# ROS full-cycle core scenarios after any_order=True

These cases validate `STT/user text → Decision model → normalize_decision → confirmation resolution → apply_decision → policy → final session/ROS action`.

## A. Ordering-specific compound cases

1. **Confirmation reject + order patch**
   - Setup: `pending_confirmation.type=preselected_section`, meat items contain sausage/crab.
   - Utterance: `아니, 그런데 소시지는 많이로 바꿔줘`
   - Expected sparse: `{"confirmation":"reject","order":{"toppings":{"소시지":"high"}}}`
   - Assert: pending is cleared, sausage becomes high, no unintended commit.

2. **Restriction add + explicit removal**
   - Setup: empty restrictions and sausage unset.
   - Utterance: `소시지 알레르기 있으니까 소시지도 빼줘`
   - Expected sparse: `{"restrictions":[{"target":"소시지","reason":"allergy","action":"add"}],"order":{"toppings":{"소시지":"none"}}}`
   - Assert: allergy is stored and sausage is removed in the same turn.

3. **Query + order change**
   - Setup: noodle section.
   - Utterance: `토마토로 바꾸고 지금 면 종류가 뭐야?`
   - Expected: both `order.sauce=토마토` and `queries[type=order_field,target=noodle_type]`.
   - Assert: sauce mutates once and response answers the query.

## B. Confirmation and execution cases

1. **Partial keep + immediate execution — known residual case**
   - Setup: pending extra items are cheese high and pepperoncino low.
   - Utterance: `아니요 치즈만 그대로 담아줘`
   - Expected: cheese high + `confirmation=reject` + `commit=true`.
   - Assert: only cheese is queued; if commit is absent, record the known `v22g9038` residual failure.

2. **Accept + new item compound**
   - Setup: pending preselected section exists.
   - Utterance: `응, 그리고 소시지는 많이`
   - Expected: `confirmation=accept` and sausage high.
   - Assert: pending selection and new patch are both preserved.

3. **Recommendation select + commit**
   - Setup: recommendation phase is proposed.
   - Utterance: `그 추천으로 바로 진행해`
   - Expected: `recommendation.action=select`, `commit=true`.
   - Assert: selected proposal reaches the robot queue exactly once.

## C. No-mutation safety cases

1. **Ambiguous reference — known residual family**
   - Setup: cheese and pepperoncino were both discussed.
   - Utterance: `아까 거 적게로 해`
   - Expected: `{"clarify":true}`.
   - Assert: order/session/ROS queue remain unchanged.

2. **Unsupported near-neighbor — known residual case**
   - Utterance: `페퍼로니 조금 넣어줘`
   - Expected: `{"clarify":true}` rather than pepperoncino.
   - Assert: no ingredient is published or queued.

3. **Plain cancellation without a unique target**
   - Setup: multiple prior mutable actions.
   - Utterance: `그거 취소해`
   - Expected: `{"clarify":true}`.
   - Assert: no state mutation before clarification.

## D. STT and runtime boundary cases

1. **STT dislike — known residual case**
   - Utterance: `버섯은 진짜 실어`
   - Expected: mushroom dislike restriction add.
   - Assert: it must not become mushroom low.

2. **Noodle-type typo query — known residual case**
   - Utterance: `우리 아까 대화한 것중에 면 종료가 있지 않았어?`
   - Expected: `queries[type=order_field,target=noodle_type]`.
   - Assert: it must not become `robot_completed`.

3. **Robot status baseline**
   - Setup: active robot task exists.
   - Utterance: `지금 로봇 뭐 하고 있어?`
   - Expected: `queries[type=robot_status]`.
   - Assert: query response is returned without order mutation.
