# LLM 실제 출력·prompt/few-shot 대조 Worklog

[선행 Python 수정](2026-11-01_llm_pending_runtime_fix.md) · [전체 후속 계획](<IRC LLM Final 2 review and acceleration.md>)
→ 실제 raw 출력의 실패부터 확인하고, 규칙 누락과 규칙 미준수를 구별한다.

**후속 전체 시나리오 계측 완료:** [사용자 수락용 결과](2026-11-01_llm_scenario_regression.md) · [전체17턴 출력 검수](2026-11-01_scenario_output_review.md) · [계측 JSON](2026-11-01_scenario_measurements.json). 사용자 지시에 따라 복합6/분리11턴을 중간 오류에도 끝까지 넣었다. 실제 소스null/실행0/단계noodle 유지, D17/P17/Q0을 관측했다. 자연스러운 크림 추천 자체는 필요한 동작이며 후보·확정 주문 연결 오류와 구분한다. 검수자2명의 후속 검수도 수락·정리했다. 아래20회는 앞선 독립 입력 A/B 이력이다. 추가 D7회는 대신 실행하지 않았다. 다음 변경은 사용자 수락 판단 후 결정한다.

**최신 실행 상태:** 사용자 prompt 변경·기존 adapter 재시험 승인 뒤 D/P 규칙과 대조 예시를 적용했다. CPU 회귀834건/하위6건 및 실제 tokenizer 길이 검사를 통과했고, 한정 GPU A/B20회는 exit0으로 종료했다. 추가 D7회 보완/검증은 예고 후 사용자 중단으로 아직 시작하지 않았다. 학습·운영 adapter 경로 교체·ROS 실행은 하지 않는다.

## 2026-11-01 — 시작 기록

상태: **실제 출력·실패 당시 system/few-shot 국소 대조 완료. 최소 보완안 작성, prompt 자체는 아직 미수정.** 사용자 요청에 따라 이 기록을 먼저 만든 뒤 아래 대조를 수행했다. 현재 작업은 읽기·국소 대조와 문서 기록이며 새 모델 추론·학습·ROS·adapter 교체·commit/push는 하지 않는다.

### 직전 확인한 실제 출력

| 근거 세션·턴 | 사용자 발화 | raw → 정규화 → 처리 결과 |
| --- | --- | --- |
| 10/31 22:32 T2 | 크림으로 선택하고 바로 진행해 줘. | raw는 크림 선택+confirmation=accept, commit 누락 → commit=false → 추천 수락만 반영, 실행 재확인 |
| 10/31 22:32 T8 | 소시지만 많이 골라서 바로 담아줘. | raw는 소시지 high만 출력, commit 누락 → commit=false → 주문 반영만 함 |
| 10/31 21:31 T4 | 버섯만 보통 양으로 바꾸고 바로 담아줘. | raw는 버섯 normal만 출력, commit 누락 → commit=false → 양파 low 보존, 버섯 변경만 함 |
| 10/31 21:31 T2 | 크림으로 선택할게 바로 담아줘. | raw에 크림+commit=true → execute=true. 모든 복합 발화가 실패한 것은 아님 |
| 10/31 22:55 T3·6·8·10 | 바로 담아. / 바로 타마. | 네 턴 모두 raw commit=true → execute=true. 해당 면·채소·육류·치즈 robot_task_started/plan 기록 존재 |

22:32 T8의 Response는 그 턴에 소시지 high가 처음 추가됐는데 “이미 많은 양으로 선택되어 있어요”라고 말했다. Response 입력의 applied_this_turn에는 toppings.소시지=high가 있었다. 실제 값 오류와 변경 사실/시점 오류를 구별한다.

로그 경로:

- `logs/llm/2026-10-31/session_20261031_223257_597216_pid259195/agent_turns.jsonl`
- `logs/llm/2026-10-31/session_20261031_213138_630159_pid218963/agent_turns.jsonl`
- `logs/llm/2026-10-31/session_20261031_225558_592562_pid284444/agent_turns.jsonl`
- `logs/llm/2026-10-31/session_20261031_225558_592562_pid284444/runtime_events.jsonl`

Cause: 위 실패 턴은 모델 raw에 실행 의미가 빠졌다. Python이 commit=true를 false로 덮어쓴 반례가 아니다.

Effect: 선행 Python 두 분기 수정과 별개로, 실제 입력 system·few-shot·history·pending 및 P 변경 사실 계약을 국소 대조한다. 원본 데이터 전수감사를 재시작하지 않는다.

### 이번 대조의 완료 조건

1. 실패/성공 턴에 실제 전달된 system의 commit 규칙과 few-shot을 확인한다. 현재 디스크 prompt와 같은지도 구별한다.
2. 실제 history·pending·before/after 및 Response 입력을 비교해 복합 실행/변경 사실에 필요한 정보가 누락됐는지 판정한다.
3. 규칙 존재/충돌/예시 공백/모델 준수 실패를 분리해 최소 후속 보완안과 검증 방법을 남긴다. 미측정 모델 성능을 합격으로 표시하지 않는다.

## 2026-11-01 — 실제 입력 대조 결과

### 1. 확인 범위와 동일성

대조 대상은 22:32 T2·T8 실패, 21:31 T2 성공·T4 실패, 22:55 T3 성공의 Decision trace다. 각 trace의 실제 `messages`에 있는 system 전체를 현재 `agent_prompts.DECISION_SYSTEM`과 비교했다. **5개 모두 문자열 전체가 같았고**, 실제 few-shot JSON도 system 안에서 파싱했다. 실패 로그가 예시 없는 옛 prompt를 사용한 것은 아니다.

- Decision system SHA256: `6a02bb1b14ba4b4e2f2167563c31174c133b3a559c33194c6c2489f674ecf7b8`.
- 22:32 T8 Task Response system도 현재 `TASK_RESPONSE_SYSTEM`과 전체 문자열 일치. SHA256: `55c31bca50dd0900cb1a550ec243688f4f453dbc0424ad23a1f282c9ea039ccc`.
- 실패3턴 입력 토큰/선행 message 수: 크림6321/3, 소시지6723/17, 버섯6458/8. 실제 recent_history는 session_before.history의 builder 허용 범위와 같았고 현재 cap16384보다 작았다. **이3턴의 원인을 입력 초과·history 잘림으로 볼 근거가 없다.** 전체 세션의 임의 장기 기억이 영구 보존된다는 뜻은 아니다.
- 크림 성공/실패 두 요청은 message 외 model_input 전체가 같았다. system도 같다. 다만 trace에 실행 중 가중치 hash·모든 decoding 설정 identity가 없어 동일 모델 통제 A/B라고 부르지는 않는다.

소시지 실패 T8의 직전 assistant 발화는 allergy를 등록했다고 하면서 “소시지 혹은 게살”을 다시 언급한 모순 안내였다. allergy는 canonical order에 정상 기록돼 있었다. 이 history를 삭제하거나 사용자 확정 사실로 승격하지 않고 state와 대조해야 한다. 이 모순 안내가 commit 누락을 일으켰는지는 미확정이다.

Cause: 실제 system/history/state가 기록돼 있어 코드의 의도뿐 아니라 모델에게 전달된 내용을 직접 비교할 수 있었다.

Effect: prompt 미전달·예시 미삽입·해당 턴의 history 잘림은 이 실패3건의 근거에서 제외했다. adapter의 내부 실패 원인이나 학습 영향은 여전히 미확정이다.

### 2. 규칙은 있지만 직접적인 복합 대조가 부족하다

| 확인 항목 | 실제 입력에서 확인한 사실 | 결론 / 보완 방향 |
| --- | --- | --- |
| D commit 규칙 | 현재 선택 실행=true, 질문/가정/부정 제외, 실행 pending 수락은 Graph 변환, 후보 수락+명시 실행에 accept와 commit을 함께 출력할 수 있다는 규칙 존재 | 규칙 자체가 없었던 것은 아니다. 다만 “함께 출력할 수 있다”를 명시 실행 의도가 확정되면 누락하지 않는 계약으로 명료화할 여지가 있다. 이 표현이 실패의 원인이라고 입증한 것은 아님 |
| D few-shot11개 | D01은 추천 수락만, D02는 실행 수락만. D04는 다중 history 기반 취소+주문+commit, N02는 기존 주문 단독 실행이다 | 복합 예시가 전혀 없다는 진단도 틀림. 하지만 추천 수락+commit, 현재량 변경+commit, 새 재료 선택+commit, allergy+선택+commit의 직접 대조가 없음 |
| Task P few-shot6개 | 6개 모두 `applied_this_turn.order_changes={}`. Mixed S09에는 실제 주문 변경 예시가 있지만 Task system에는 그 Mixed 예시가 들어가지 않음 | Task P에는 “이번에 새 선택/변경됨”과 “이미 같은 값이었음”의 직접 비교 예시가 없다. 주어진 현재 값만 보고 과거부터 같은 값이었다고 말하는 오류를 겨냥해야 함 |
| P 사실 규칙과 실패 입력 | applied_this_turn에 있는 것만 이번 턴 반영이라고 하라는 규칙이 있음. T8에는 소시지 high 신규 변경을 전달했지만 “이미”라고 응답 | 전달 누락이 아니라 P의 사실/시점 준수 실패. before state field를 새로 만들기 전에 기존 변경 집합으로 대조한다 |
| pending 설명 | D 입력 설명은 실행 확인/추천 후보를 함께 미확정으로 표현, P도 generic pending은 아직 확정되지 않은 상태라고 표현함 | 추천 후보 미확정과 execution 대기 중 확정 주문을 분리해 설명할 필요. 단, 소시지 T8은 pending=null이므로 이 설명만으로 모든 commit 실패를 설명할 수 없음 |

읽은 코드: `agent_prompts.py`의 멀티턴/수량/confirmation/commit/Task Response 규칙과 system 조립부, `prompt_examples.py`의 실제 EXAMPLES, `decision_model.py`의 history/토큰 입력 및 sparse parse/normalize 경로. 예시는 응답 템플릿/키워드 gate가 아닌 system 내용이다.

### 3. 최소 후속 보완안 — 전면 재작성하지 않는다

1. **D 규칙 국소 명료화:** 현재 발화의 선택·변경·제약·추천 수락과 실행 의도를 각각 확인한다. 명시 실행이 확정되면 해당 patch/restrictions/confirmation과 `commit=true`를 함께 출력한다. pending이 없어도 대상이 확정되는 새 주문+실행은 표현한다. “바로” 문자열만으로 commit을 강제하지 않는다.
2. **D 최소 대조 예시:** 동일 history/state에서 추천 수락만 vs 수락+실행, 버섯 변경만 vs 변경+실행(양파 기존량 유지), 소시지 새 선택만 vs 선택+실행, allergy+선택+실행을 포함한다. 부정·인용·조건부·미래 실행 예약과 복합 history 지시어도 음성/양성 짝으로 다룬다. 기존 D04/N01/D03의 정상 의미는 유지한다.
3. **P 변경 사실 대조 예시:** 현재 값이 high인 동일 결과라도 applied에 신규 high가 있으면 “많이 선택했어요”, 변경 집합이 비고 기존 선택이 근거로 확인되면 “많이 선택돼 있어요”처럼 구별한다. low→normal 수정은 “보통으로 바꿨어요”, 이미 normal 유지와 분리한다. execute=false에서 담았다고 하거나 실제 next_prompt가 없는데 실행 질문을 생성하지 않는다.
4. **prompt→입력→Graph 계약 검수:** 독립 시연은 실제 history에 섞지 않는다. state/pending/next_prompt/patch와 예시 정답을 실제 builder/Graph로 대조한다. 새 예시를 붙인 뒤 실제 tokenizer 길이·history 보존을 측정하고 그 전에는 token cap을 올리거나 history를 줄이지 않는다. 새 수량 정책의 Q 예시 문제는 이미 확인된 장부를 재사용해 후속 정렬한다.
5. **실제 모델 대조:** 위 실패 입력과 단독 성공/부정/상이한 history 대조를 고정한 뒤 같은 adapter/prompt 외 조건에서 비교한다. raw의 필드·정규화·after order·pending·execute·P 시점 표현과 지연을 분리한다. CPU 정답 주입 통과를 모델 성능 개선이라고 보고하지 않는다. GPU 실행은 이번에 하지 않았다.

Cause: 규칙 존재만으로 모델 준수를 보장하지 못하고, 현재 few-shot에는 이번 실패와 직접 대응하는 대조 공백이 있다. 공백의 인과 기여도는 실제 모델 A/B 전 미확정이다.

Effect: 후속 수정은 D/P의 해당 규칙·대조 예시에 한정할 수 있다. 학습 데이터 전면 폐기, Python 문구 판정, 새 pending field, 무조건 commit 강제는 하지 않는다.

### 4. 이 작업에서 하지 않은 것

prompt/few-shot/Runtime·모델 경로·학습 데이터는 이번 대조 작업에서 변경하지 않았다. 앞선 Python 수정과 테스트는 그대로 보존했다. 최신 실제 로그는 Python 수정 전 기록이므로 수정 후 실제 모델 재시험으로 표시하지 않는다. 새 독립 검수 에이전트나 외부 모델도 호출하지 않았다.

### 5. 검증과 다음 한 단계

기존 `test_gl06_prompt_contract.py`와 `test_pending_rejection_regression.py`를 재실행해 **71 passed / exit0**을 확인했다. 기존 812건에 포함된 부분집합 재검증이며 새로운 71개 의미 사례가 추가됐다는 뜻이 아니다. SWIG 사용 중단 경고 2종이 있었고 `git diff --check`는 통과했다. 로그 분석·prompt 비교는 JSON 필드와 전체 문자열 비교로 수행했으며 모델을 호출하지 않았다.

다음 한 단계: §3의 D/P 규칙·대조 few-shot 국소 보완을 적용하고, 예시의 필드 정합성과 길이를 검증한 뒤 실제 모델 재시험을 별도로 진행한다. 이 문서는 적용 전 분석/보완안이며 prompt 변경 완료로 읽지 않는다.

## 2026-11-01 — 사용자 승인 후 prompt 적용·기존 adapter 비교

### 적용한 변경

- D: 선택·수량 변경·제약·후보 수락과 명시 즉시 실행 의도를 함께 추출하도록 명료화했다. 추천 수락만으로 실행하지 않으며, 질문·인용·부정·조건 미성립·나중 실행은 commit이 아니다. execution pending과 미확정 추천 후보의 설명을 구별했다.
- P: 이번 턴 변경 집합과 기존 선택 유지를 구별하도록 했다. 현재 high/normal 값만 보고 과거부터 같은 값이었다고 말하지 않는다. 실행/완료/pending 질문 경계는 유지했다.
- 독립 예시 D7개(C01~07), Task P3개(S10~12)를 추가했다. C01/02·03/04·05/06은 같은 입력 상태에서 발화와 실행 의도만 다르다. C07은 “바로 담지는 마”의 선택·제약 보존/비실행 예시다. S11/12는 같은 최종 주문에서 변경 사실만 구별한다. 기존 시연은 삭제하지 않았다.
- prompt 계약 버전: `compound_commit_change_facts_v2`. 예시 수 D18/Q4/Task9/General1/Mixed1, 총33개. Q system·Q 예시·학습 데이터·가중치와 Runtime/ROS/VLM 로직은 이번 prompt 단계에서 바꾸지 않았다.

Cause: 실제 입력에 규칙은 있었으나 복합 실행과 신규 변경/기존 유지의 직접 대조가 부족했다.

Effect: Python으로 발화를 재해석하거나 고정 응답을 보내는 경로 없이, 모델이 읽는 규칙·예시만 국소 보완했다. 성능 개선 여부는 아래 실제 추론 결과로 판단한다.

### 실행 전 검증

기존 GL06 예시 회귀의 예상 주문/실행 oracle에 새 C01~07의 명시적 정답을 추가했다. 새 예시의 schema·실제 serializer 재구성·Graph 전체 주문/제한/실행 결과를 검사했고, 대조 쌍과 P 변경 집합/확인 대상 검사도 추가했다. 테스트가 새 ID를 모르면 기존 주문 불변으로 가정하던 분기를 확장했으며 기존 예시 oracle은 보존했다.

LLM13파일 **834 passed, 하위6 passed**, exit0. 실제 tokenizer로 A/B10개 입력을 점검해 최대8617토큰이며 history/input 일치를 확인했다. token cap·history 창·출력 상한은 바꾸지 않았다.

### GPU A/B 범위

수행 스크립트: `src/soomac_irc/test/manual_prompt_alignment_probe.py`. 실행 산출물: `/tmp/soomac_prompt_alignment_20261101_v2/`의 `PLAN.json`, `IDENTITY.json`, `results.jsonl`, `EXIT.json`.

실제 로그 D 실패3개/단독 성공1개와 부정·인용·기존 low 보존 control3개, 실제 P 신규 high/normal 변경 입력2개와 기존 high 유지 control1개를 합친10개 입력에 old/current prompt를 각각 적용한다. 총20회로 한정하며 자동 재시도하지 않는다. 출력 시점 D의 실제 결과만 Graph에 넣어 full order·실행·선호·불필요 추천 여부를 검사한다. 후속 Q/P를 실제 추론한 것으로 세지 않는다. P는 raw를 직접 검토하고 문구 정규식으로 의미 합격을 자동 판정하지 않는다.

기존 `model_runtime.py`의 최신 best D/Q/P 경로를 변경 없이 읽는다. Base는 한 번 로드하고 세 adapter를 붙이되 이번 추론은 D/P만 호출한다. 같은 실행에서 prompt만 바꾸므로 이전 로그 비교보다 조건을 더 통제한다. GPU 시작 전 다른 compute 작업이 없는 것을 확인했으며 ROS/STT/TTS/VLM은 호출하지 않는다. adapter weight/config hash를 실행 기록에 남긴다. 이는 독립 입력 A/B이며 누적 closed-loop 전체 시연은 아니다.

### 실제 GPU 실행 종료 결과와 중단 시점

`EXIT.json`: exit_code=0, calls=20, GPU=`NVIDIA RTX PRO 4500 Blackwell`, peak_allocated_bytes=13627231744(약12.69GiB). PID65066은 실행 종료했다. 사용자 VRAM 질문 뒤 호스트를 다시 읽어 총32623MiB 중756MiB 사용, compute 프로세스 없음과 PID 소멸을 확인했다. 현재 낮은 VRAM은 시험 종료 뒤 모델 메모리가 해제된 상태와 부합한다.

**baseline도 기존 best adapter를 붙인 추론이다. Base-only 시험이 아니다.** 동일 Base에 최신 best D/Q/P adapter를 로드하고, D는 adapter_name=decision, P는 adapter_name=response를 각 generate 전에 활성화했다. D14회/P6회이며 Q는 로드만 했고 호출하지 않았다. 경로·weight/config hash는 `IDENTITY.json`에 기록했다. 세 Agent 전체 누적 시나리오 시험으로 세지 않는다.

| 검사 | 기존 prompt | 수정 prompt |
| --- | --- | --- |
| 크림 수락+실행 | commit 누락 | accept+commit=true, 실행 검사 통과 |
| 버섯 normal 변경+실행 | commit 누락 | 변경+commit=true, 양파 low 유지 |
| 소시지 high 선택+실행 | commit 누락 | 여전히 commit 누락 |
| 기존 소시지 low 유지 | normal로 잘못 변경 | 여전히 normal로 잘못 변경 |
| 부정·인용·단독 실행 | 비실행/정상 실행 구분 유지 | 실행 권한은 유지했으나 인용 시스템 질문의 route가 general로 바뀌어 별도 회귀 확인 |

D의 주문/실행/선호/불필요 추천 검사만 세면 기존3/7→수정5/7이다. 이 수치에는 route 의미 판정이 포함되지 않는다. 시스템 동작을 묻는 인용 control은 task가 맞으므로 새 general 출력은 미해결 회귀로 따로 기록하고 전체 합격이라 하지 않는다.

P 신규 high: “이미 많은 양으로 선택되어 있어요”→“소시지를 많이 선택했어요”로 시점 오류 개선. P 버섯 변경: 보통으로 바꾼 사실과 양파 low 유지 설명. P 기존 high 유지: 새로 변경했다고 하지 않고 현재 선택 상태 설명. 이3개 raw는 주담당이 입력 사실과 대조했으며 자유 표현 전체에 대한 보장은 아니다.

추가 토큰은 D+1883/P+1756이었다. 이번 표본의 호출 중앙값은 D4.994→6.660초, P5.389→6.927초다. 입력이 늘어 이 표본에서는 느려졌다. 일회·고정순서 A/B로 워밍업 효과가 섞일 수 있으므로 안정된 배속/p95로 일반화하지 않는다. 속도가 개선됐다고 보고하지 않는다.

사용자 중단 시점에 예고한 추가 보완(D 소시지 commit·기존량·인용 route)과 D7회 재시험은 **미시작**이다. 현재 prompt는 위20회에서 검사한 v2와 같으며, 소시지 실패와 route 회귀가 남은 후보이지 전체 시나리오 채택 완료본이 아니다.
