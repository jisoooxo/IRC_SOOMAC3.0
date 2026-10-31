# 복합 주문·즉시 실행 보강과 독립 의미 검수

[작업 기록](2026-11-01_dataset_rebuild_worklog.md) · [Claude 지시 원문](2026-11-01_claude_compound_review_brief.md) · [신규 대화와 검증 코드](../training/learning_first_v43/compound_semantics_20261101/author_compound.py)
→ 주문·수량·제약 변경과 실행 요청을 같은 Decision 정답에 넣고, 실제 후속 상태·응답까지 연결한다.

## 1. 현재 범위와 상태

2026-11-01 KST, `llm_final_2`. 사용자 요청은 Claude와의 의미 교차검수 및 「바로 해줘 / 바로 넣어줘」 복합 발화 보강이다. 이번 작업에서 운영 Runtime·prompt·few-shot·cache·VLM·ROS·운영 adapter 경로를 수정하지 않았다. 새 학습이나 GPU 모델 추론도 하지 않았다. 이전 학습 중지를 유지하며 기존 사용자 학습 승인을 철회한 것으로 해석하지 않는다.

**자료의 정답 연결 검사와 모델의 자연어 이해 성공은 서로 다르다.** 모델이 지금부터 이 표현을 반드시 이해한다고 주장하지 않는다. 이 자료로 학습한 뒤 실제 누적 대화에서 별도로 확인한다.

### KEEP / FIX / HOLD의 뜻

| 판정 | 의미 | 의미하지 않는 것 |
| --- | --- | --- |
| KEEP | 검수 범위에서 발화·정답·제공된 state가 맞아 유지·채택한다 | 현재 adapter가 실제로 그 발화를 성공했다는 뜻이 아님 |
| FIX | 확인된 잘못을 수정한 뒤 영향 history·필드를 다시 대조한다 | 기존 입력/history 전체를 지우거나 정상 자료를 모두 재생성하는 것이 아님 |
| HOLD | 정답이 모호하거나 근거가 부족해 해당 행의 학습 투입을 보류한다 | 모든 원본 삭제나 전체 프로젝트 중단을 뜻하지 않음 |

따라서 이번 「24대화 KEEP」은 교정한 gold 채택이다. 본 학습·실제 모델 누적 회귀·실물 시연 합격과 구분한다. 보고서의 문체 관찰/자료 의존성에 붙은 HOLD도 개별 학습 행 제외와 같은 뜻으로 취급하지 않는다.

최신 저장 후보는 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/candidates/compound_semantics_20261101_r2`이다. 그 안의 `dataset/MANIFEST.json` SHA256:

```text
7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa
```

현재 manifest의 `ENCODING_PASS_SEMANTIC_REVIEW_PENDING`은 **조립 시점의 불변 기록**이다. 이후 의미 검수의 채택 여부는 이 문서와 독립 보고서로 기록하며, 옛 영수증을 소급해 PASS로 바꾸지 않는다. 원본 r4_a2와 첫 실패 후보 r1도 보존한다.

## 2. 실제 추가된 대화와 정답

24대화·60사용자턴, D60/Q8/P41=109행 초안이다. 정확히 같은 입력·정답 28행은 기존 train 정답에 alias 연결했고, 새81행(D47/Q6/P28)만 추가했다. 109행 모두 train 정답 또는 같은 train 정답의 alias로 추적된다. 단어만 치환한 독립 한 줄이 아니라 실제 앞선 발화·응답·state가 다음 턴 입력에 누적된다.

| 대화 ID | 발화와 문맥 | 핵심 정답·불변 항목 |
| --- | --- | --- |
| G01~G03 | 기분 안 좋음/스트레스/우울 + 매운 것 못 먹음 + 전체 추천 → 「그 구성으로 바로 해줘/골라서 바로 넣어줘/지금 바로 진행해줘」 | D cannot_eat+추천요청 동시 기록 → Q 크림·얇은면·안전한 미래 토핑 → P 근거 있는 권유 → 다음 D accept+commit. 후보 수락 전에는 주문에 승격하지 않음 |
| G04~G06 | 얇은면 보통 선택+느끼한 소스 추천 → 「크림으로 골라서 바로 넣어줘/크림으로 할게 바로 해줘/추천한 크림으로 선택해서 지금 바로 넣어줘」 | 면 normal 유지, 크림 후보 수락+commit을 한 번에 처리 |
| G07~G09 | 양파 high+버섯 low → 둘 다 low → 「버섯만 normal로 바꾸고 바로 해줘/바꿔서 바로 넣어줘」 | order.patch는 버섯 normal만, 미언급 양파 low 유지, commit=true |
| G10~G12 | 선택 가능 재료 안내 → 첫 게살 알레르기+새 소시지 high+즉시 실행 | restrictions allergy 추가, 게살 none 취소 패치, 소시지 high, commit=true 동시 기록. 실제 order에는 none을 보관하지 않음 |
| G13~G15 | 앞선 매운맛 cannot_eat → 치즈 high 선택 → 페퍼론치노 제외+치즈 low+즉시 실행 | 강한 제한 유지, 치즈만 low, 페퍼론치노 실행 대상 없음, commit=true |
| G16 | 크림+얇은면 low 확정 → 실제 실행 질문 → 「바로 넣어줘」 | 기존 pending/주문 대상에 commit=true, 재확인으로 루프 돌지 않음 |
| G17 | 이미 확정한 크림+얇은면 low, pending 없음 → 선택 상태 질문 → 「바로 해줘」 | history와 실제 order로 유일한 대상 식별. pending이 없어도 명시적 commit 자체를 버리지 않음 |
| G18 | 「바로 넣어줘라는 말은 무슨 뜻이야?」 → 「지금은 바로 넣지 마」 | 뜻 질문과 부정은 실행 요청 아님. 실행하지 않고 선택 유지 |
| G19 | 「나중에 마음이 바뀌면 … 바로 해줘라고 말할게」 → 「지금 바로 해달라는 건 아니야」 | 미래 조건·부정은 현재 실행 아님 |
| G20 | 채소 low 선택 → 「버섯만 normal로 바꾸되 바로 담지는 마」 → 담긴 것은 아닌지 질문 | 수량 변경만 적용, 실행 거절 유지. P는 선택 변경과 실제 담기 구별 |
| G21 | 크림+얇은면/토마토+넓은면 두 대안 미확정 → 「그걸로 바로 해줘」 | 지시 대상이 둘이므로 clarify=true. 현재 고유 대상이 있는 G17과 구별 |
| G22 | 「면만 추천, 소스는 나중에」 → 「그 면으로 골라서 바로 넣어줘」 | 실행 의도 commit=true 보존. 소스 미선택은 missing_order여서 execute=false. 면만으로 매운맛을 해결했다고 말하지 않음 |
| G23 | 기분 안 좋음+느끼함 비선호+매운 것 못 먹음+전체 추천 → 수락+즉시 실행 | 사용자 메뉴 기준 토마토 제안, 크림/오일로 임의 대체하지 않음. 매운맛 제한 유지 |
| G24 | 8턴: 채소 선택→미래 소시지 양 취향만→채소 둘 다 low→게살 allergy→소시지 미주문 확인→버섯 normal→현재 채소 확인→「아까 바꾼 대로 바로 넣어줘」 | 선행 user7턴 누적, 양파 low/버섯 normal·allergy 보존. 소시지 취향을 확정 주문으로 승격하지 않음 |

현재 단계 이후 재료까지 추천/선택할 수 있다는 것과 로봇이 모든 물리 단계를 한 번에 건너뛴다는 것은 다르다. G01~03의 후보 수락은 미래 주문도 보존하지만 이번 실행 대상은 현재 면 단계다. 완료·단계 전환은 새로 만들지 않는다.

## 3. 교차검수의 채택·교정

독립 Codex sol/high는 원 초안24대화를 읽고 21 KEEP/3 FIX로 판단했다. FIX는 G10~12의 「육류는 어떤 게 있어? → 소시지와 게살」이 게살을 육류로 오분류할 수 있다는 지적이다. root가 P를 「소시지가 있고, 게살도 같은 단계에서 고를 수 있어요」로 교정하고 다음 턴 history를 재생성했다. 실행 의도·알레르기·수량 정답은 바꾸지 않았다.

Claude는 별도 Orca 세션에서 기준·Runtime·전체24대화/60턴 trace를 읽었다. [실제 저장 보고서](../training/learning_first_v43/compound_semantics_20261101/CLAUDE_REVIEW.md)는 교정 후24 KEEP이며, root가 최종 자료와 대조했다. 검수 중 제기된 「모든 대화의 중간 실행이 0」은 **신규 구간별 진단39개에만 해당**했다. 기존 전체 기반에는 `DA31-CORE-compound-T02/T05/T07/T09-D` 등 중간 실행과 명시적 오프라인 완료 bridge가 있다. Claude도 이 일반화를 정정했다. 정상 기존 자료를 다시 만들거나 삭제하지 않았다.

첫 조립은 G07~09 T01 P가 기존과 같은 입력인데 표현만 다른 정답이어서 `conflicting target`으로 중지했다. 기존 `NS01-C01-T01-P`와 같은 정답을 재사용해 alias 처리했다. 상충 검사를 약화하지 않았고, 이를 모델 실패나 새로운 의미 모순으로 세지 않는다. 원 초안→a2→a3의 실제 변경은 위6개 P와 그 영향 history뿐이다. 최종 검수 대상은 저장 후보 `authored/`에 남아 있다.

Cause: 자연어 실행 요청에 주문 변경·제약·이전 지시 대상이 함께 오면 정답의 한 필드만 맞아서는 실제 흐름이 이어지지 않는다.

Effect: 같은 턴의 D 패치·제약·commit, Q 후보, P 사실성, 다음 턴 history를 연결하고 부정 대조까지 같은 기준으로 확인한다.

## 4. 검증 근거와 남은 경계

실제 Graph를 **작성된 gold로** 재생한 24대화/60턴에서 필드·실행·호출·질문 계약 오류0, 예외0이다. commit 의도20턴 중 실제 실행19턴이며, 나머지1턴은 G22의 필수 소스 누락 차단이다. 신규81행은 실제 trainer AST·tokenizer·chat template·EOS·target mask로 재인코딩했고, 정확 중복28행은 동일 입력·정답의 train 행을 재사용했다. history 절삭은0이다. 기존8230행은 동일 full-record/system 영수증을 재결속했다.

| 항목 | 저장 결과 |
| --- | --- |
| 전체/train | 8311 / 6647 |
| D train / Q train / P train | 4360 / 623 / 1664 |
| 신규/정확 중복 alias | 81 / 28. alias 전부 train에 연결, holdout을 train으로 복사하지 않음 |
| 기존 원본 보존 | 기존 train은 byte prefix 그대로, validation/test/challenge/holdout은 byte 동일 |
| 입력 길이 | 최대 D8616/Q4332/P8430, cap D9216/Q5120/P9216 유지. 새 GPU 프로파일 성공을 뜻하지 않음 |

재현:

```bash
PYTHONPATH=src/soomac_irc /home/roma/miniconda3/envs/gemma4_env/bin/python -m pytest -q \
  training/learning_first_v43/compound_semantics_20261101/test_compound.py \
  training/learning_first_v43/conversation_supplement_20261101_r1/test_meaning_r4.py
```

최초43개 신규 계약+6개 기존 의미 회귀=49건 통과했다. 최종 저장 후보의 hash/정확 alias/원본 보존 검사2개를 추가한 뒤 **51건 통과**, exit0, SWIG 사용 중단 경고2건이다. 109행 모두 저장 후보 또는 정확 train alias에 있는지 실제 파일로 검증한다.

전역 의역 누수, 전체 데이터의 모든 의미·한국어 표현 무오류, 새 adapter의 실제 성공은 이번 결과로 인증하지 않는다. 수치 편향 합격 기준은 사용자 보류를 유지한다. 숫자를 통과시키려고 정상 좁은 추천525행을 일괄 삭제하거나, 「바로」 키워드로 Python이 commit을 강제하는 코드는 추가하지 않았다.

## 5. 최종 판정

**이번 보강24대화/109행의 작성된 gold는 국소 교정 뒤 채택한다.** Claude는 교정 후24 KEEP, Codex는 원 초안21 KEEP/3 FIX였고 root가 해당3개 분류 문장을 교정했다. 두 검수 모두 G22의 의도와 실행 가능성 구분, G24의 복합 누적 상태 보존에 동의한다. Codex가 최종 a3를 다시 검수했다고 쓰지 않는다. a3 국소 변경은 root의 실제 저장 검사와 Claude 추가 대조로 확인했다.

Claude 보고서 SHA256은 `af57e9daf939c5871dd2c563b9222d9266e4897d5d31a79f69babba8cc97f8de`이다. 원 보고서의 다음 표현은 root 정산에서 구분한다. 원 보고서를 유리하게 덮어쓰지 않는다.

| 원 보고서 표현 | root 실제 대조·처리 |
| --- | --- |
| 6턴 변경, 103행 재사용 | 원본→a3 record 비교는 **변경18행 / 동일91행**. 직접 P6행과 영향 history의 D9/P3행이 바뀌었다. 6은 직접 고친 P 개수이지 전체 변경행 수가 아님 |
| alias 조립은 아직 미완 | Claude가 본 것은 `/tmp` 초안이다. root는 이미 별도 r2에 조립하고 109행의 81신규/28정확 alias를 실제 train에 대조했다. Claude가 조립을 검수했다고 쓰지 않음 |
| DA31-CORE 의존성 HOLD | 기존 정상 전체 시연 자료를 계속 포함해야 한다는 주의다. 해당 자료는 실제 r4_a2와 새 r2에서 보존했다. 신규24대화를 탈락시키거나 기반 전체가 실행 불능이라는 판정이 아님 |
| 감정 응대 naturalness HOLD | 감정에 대한 짧은 공감이 없다는 선택적 문체 관찰이다. 새 gold의 필드 오류는 아니다. 감정 때문에 특정 메뉴가 기분을 치료한다는 근거를 창작하지 않으며, 후속 자연스러움 비교 항목으로만 유지한다 |

최종 시험은51건 통과다. 기존 학습 run `conversation_r3_20261101_dec29eead775/STATE.json`을 다시 읽었으며 여전히 `STOPPED_BEFORE_FULL_TRAINING_DATA_CORRECTION_REQUIRED`, `full_training_started=false`다. 자동 재시도·원본 삭제·운영 adapter 교체는 하지 않았다. Claude 세션은 보고서 작성 완료 후 입력 대기이며 추가 작업을 지시하지 않았다.

**다음 한 작업은 새 manifest를 결속한 학습 preflight와 학습 설정 점검이다.** 최신 사용자 지침에 따라 **Decision은 원격 별도, Response→Recommendation은 로컬 순차**로 분리한다. 기존 user 학습 승인은 유지하되 이전 r3 큐를 그대로 재개하지 않는다. 이전 batch1×acc16·6epoch를 새 최종값으로 자동 채택하지 않는다. 실제 새 adapter 성능은 학습 후 동일 누적 시나리오에서 별도 측정한다. 현재 adapter가 이 보강 자료를 이미 학습했다고 말하지 않는다. 평가기 자체의 오채점 검증과 첫 응답 사실성 기준은 [worklog 최상위 지침](2026-11-01_dataset_rebuild_worklog.md)을 따른다.

## 6. 사용자 시나리오의 최종 train 포함 재확인

사용자 「MD에 기록하고, 내 시나리오 제대로 들어가 있나 한 번 더 확인」 요청으로 최신 r2의 실제 train을 다시 읽었다. 기존 의미 검수를 전면 재개하거나 데이터셋을 재생성하지 않았다. manifest SHA는 위 값과 같고 파일 hash도 일치했다. [저장된 실제 실행 결과](2026-11-01_scenario_inclusion_recheck.json)와 [국소 확인 스크립트](../training/learning_first_v43/compound_semantics_20261101/check_saved_required_scenario.py)를 남긴다.

| 사용자 시나리오 | 최종 train의 실제 ID | 확인한 정답·결과 |
| --- | --- | --- |
| 얇은면 보통+느끼한 소스 추천 → 크림 선택하고 바로 담기 | `DA31-CORE-separate-T01-D/Q/P` → `DA31-CORE-compound-T02-D` | D 면 normal+추천 request, Q 크림 후보, P 크림 선택 질문. 다음 D 소스 크림+commit=true. 후보일 때 소스null, 수락 후에만 크림 확정, 면 양 normal 유지 |
| 양파·버섯 둘 다 적게 담아줘 | `DA31-CORE-compound-T04-D/P` | 양파low·버섯low. commit 없음, execute=false. 실제 execution pending과 일치하는 확인 질문 |
| 버섯만 보통 양으로 바꿔서 바로 담아 | `DA31-CORE-compound-T05-D` | 버섯normal+commit=true. 양파low, 크림·얇은면normal 보존, execute=true |
| 게살 알레르기가 있어서 소시지만 많이 바로 담아줘 | `DA31-CORE-compound-T07-D` | 실제 표기는 「알러지」 변형. allergy+소시지high+게살none+commit=true 동시 기록. 적용 후 게살 키 없음, 기존 채소량 보존, execute=true. 「알레르기」 표기는 polite/directive 및 G10~12에 별도 존재 |
| 페퍼론치노는 빼고 치즈만 적게 골라서 바로 담아 | `DA31-CORE-compound-T09-D` | 치즈low·페퍼론치노none+commit=true. 실제 주문에는 페퍼론치노 없음. 앞선 게살allergy·채소·소시지high 보존, execute=true |

기본 시연 계열은 실제 train에 D26/Q1/P10이 있다. 공통 첫 턴은 동일 정답을 공유하므로 compound용 Q/P가 별도 ID로 반복되지 않는다. 중간 즉시 실행4턴에는 현재 Runtime이 deterministic 시작 안내를 쓰므로 가짜 P 학습 행을 추가하지 않는다. 출력은 시작 안내이지 물리 완료 주장도 아니다.

재확인은 다음 범위를 실제 검사했다. 저장된 6개 사용자턴의 gold를 각각 **그 행의 저장된 session/robot snapshot**으로 실제 Graph에 주입했다. 수량·소스·제약·execute와 인접 저장행의 order/history 연속성을 대조했고 모두 통과했다. 마지막 치즈 턴에는 선행 사용자5턴이 누적돼 있다. 단계 사이의 추가 assistant 안내와 `completed_tasks`는 저장돼 있던 명시적 오프라인 bridge 입력이다. 이 검사에서 bridge를 새로 실행·판정하거나 실제 물리 완료를 만들지 않았다. 완전 닫힌 루프 모델 추론이나 ROS 테스트가 아니다.

「바로 해줘/바로 넣어줘」 등의 보강109행도 최종 train 행 또는 정확 동일 train alias로 모두 다시 확인했다. 기존45개 신규/저장 검사+6개 r4 의미 회귀를 재실행해 **51건 통과**, exit0, SWIG 경고2건이다. 신규 데이터 수정·학습·모델 교체 없이 포함 여부와 gold의 동작만 확인했다.

별도 Codex sol/high의 이번 국소 교차검수는 **찬성**이다. 실제 D train 4060/4071~4075행, Q train599행, P train1458행의 ID를 root가 다시 읽어 확인했다. 109개 중81개는 저장 record 자체가 같고, 별칭28개는 metadata/ID까지 같은 행이라는 뜻이 아니라 **input·target·route가 같은 train 학습 예시**라는 점도 확인했다. 이 판정은 「지정 r2 train의 저장 정답·CPU 검사 기준」이며 adapter의 실제 이해·실물 성공으로 확장하지 않는다. CLI 원시 기록은 `/tmp/soomac_scenario_inclusion_review.out`이다.

재현 명령:

```bash
PYTHONPATH=src/soomac_irc /home/roma/miniconda3/envs/gemma4_env/bin/python \
  training/learning_first_v43/compound_semantics_20261101/check_saved_required_scenario.py
```

Cause: 자료의 대표 요약만 보면 기본 전체 시연과 국소 보강이 최종 train에서 모두 보존됐는지 놓칠 수 있다.

Effect: 여섯 사용자턴·네 실행 지점의 실제 ID, 누적 이전 주문과 history, 새109행의 최종 위치까지 확인 가능한 근거를 남겼다. 모델 성공 판정은 여전히 별도이다.
