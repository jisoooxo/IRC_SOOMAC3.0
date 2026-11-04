# 로컬 Unsloth 본 학습·실모델 품질 평가 결과 (2026-11-03)

[학습 계획](2026-11-01_local_unsloth_production_plan.md) · [작업 기록](2026-11-01_training_package_build_worklog.md) · [상태](training_acceleration_20261101/STATE.json)
→ **Response와 Recommendation은 각각 6epoch 본 학습·저장·vLLM 로딩을 마쳤다. 그러나 기존 Decision을 붙인 17턴 누적 시나리오는 네 조합 모두 실패하므로 운영 adapter는 교체하지 않는다.**

## 1. 학습 실행과 저장 무결성

| Agent | 실제 본 학습 | 최저 validation loss | 마지막 validation loss | 최종 검증 |
| --- | --- | ---: | ---: | --- |
| Response | 624/624step, 6epoch | 0.8207007646560669 (step150) | 1.1994315385818481 (step624) | 정상 종료, full-state·완료 영수증·best/epoch6 원본 PEFT 통과 |
| Recommendation | 234/234step, 6epoch | 0.10413022339344025 (step117) | 0.11286281794309616 (step234) | 정상 종료, `checkpointing.verify_completion`·epoch6·full-state·best/epoch6 원본 PEFT 통과 |

Q 완료 영수증 SHA-256: `ab76cdc92d3c5b8fa332386bcfef57cb227f8a5b82f7ce883bad4c6aeb042e05`. Q best adapter SHA-256: `204d0a35746aef46c4bae18aa9a09c244b85e06853f1e37687ec3ba5803d6002`; Q epoch6: `8f33f34d2b9881b770f60c70d37a8ee1e3802a1d0d3fddead0f334f2de8577a2`. 두 Q 원본 PEFT 파일의 656개 텐서 키·config가 일치한다. Q 서비스는 `inactive/result=success/exit=0`, GPU 학습 프로세스는 종료됐다. 기존 HF checkpoint·원격 Decision·데이터/history/labels/6epoch 정책은 변경하지 않았다.

Cause: 기존 순차 실행기가 Response 정상 종료 후 Q 본 학습을 계속했고 두 완료 영수증이 실제 목표 step과 export SHA를 검증했다.
Effect: 학습 실행·복구 가능한 산출물은 확보됐으나, validation 수치만으로 실제 대화 품질을 합격시키지 않는다.

## 2. 봉인 24건: 원본 PEFT 실모델 출력

**2026-11-03 재부팅/임시 정리 후 증거 보존 보강:** 원래 `/tmp/.../new_holdout_v4_runtime`과 네 조합의 `/tmp` 평가 원본은 현재 존재하지 않는다. 생성기 `heldout_cases.py`로 24건을 [로컬 지속 평가 입력](../training/learning_first_v43/final/goals/GL10/eval_holdout_v4_20261103_recovered/NEW_FUNCTIONAL_CASES.json)에 다시 생성했고 SHA-256이 원본과 **정확히 `855a000c0fbb26546b83c5d2a22ffac3d2592f27161e9ba97fb4c2e125d8aa06`으로 일치**했다. 같은 release manifest와 24/24 runtime builder 재생 검사를 통과했다. P best·Q epoch6·기존 D 조합만 원본 PEFT 실추론을 [로컬 지속 예측 결과](../training/learning_first_v43/final/goals/GL10/eval_pbest_q6_20261103_recovered/PREDICTIONS.json)에 다시 남겼고 SHA-256은 `6ab0e8b7b76317431586f8d0b753a08344a3b92201e2593f00e78f2826a2764e`이다. 새 자동 결과는 D3/8, Q8/8, P8/8로 이전 **교정 후** 수치와 일치한다. Q8·P8 문장을 재검토했고 이 16건에서 새 명시적 모순은 찾지 못했다. 이전 세 조합의 원본 예측 파일은 재생성하지 않았으므로 아래 표는 그때 기록한 결과이며 현재 로컬 파일로 직접 재검증 가능한 행은 P best·Q epoch6 조합이다. 이 두 생성 디렉터리는 `.gitignore` 대상이므로 **Git에 백업된 것은 아니며** 임의 삭제하지 않는다.

봉인 파일 `/tmp/soomac-training-accel-20261101/new_holdout_v4_runtime/NEW_FUNCTIONAL_CASES.json` SHA-256 `855a000c0fbb26546b83c5d2a22ffac3d2592f27161e9ba97fb4c2e125d8aa06`의 D/Q/P 각 8건을 **네 조합 모두 실제 GPU 추론**했다. D는 로컬에 있는 기존 `remote_all_a519cf.../decision/best_adapter`를 고정했으며 새 원격 Decision 성적이 아니다.

| P 후보 | Q 후보 | 기존 D | Q | P | 단일 턴 판정 |
| --- | --- | ---: | ---: | ---: | --- |
| best(step150) | best(step117) | 3/8 | 7/8 | 8/8 | Q-06 버섯의 근거 없는 `low` 실패 |
| best(step150) | epoch6 | 3/8 | **8/8** | **8/8** | Q/P 봉인 단일 턴 자동검사 통과 |
| epoch6 | best(step117) | 3/8 | 7/8 | 8/8 | Q-06 동일 실패 |
| epoch6 | epoch6 | 3/8 | **8/8** | **8/8** | Q/P 봉인 단일 턴 자동검사 통과 |

Q-06 입력은 “버섯 하나로 새 채소 추천, 추가 수량 취향 없음”이다. 정답 `toppings.버섯=normal`에 대해 Q best가 `low`, Q epoch6이 `normal`을 출력했다. Q epoch6의 나머지 7건은 크림/토마토 취향, 매운맛 제한, 게살 알레르기, 근거 없는 추천량, 페퍼론치노 제외를 자동검사 통과했다. P 두 후보의 8개 실제 문장을 각 입력의 선택량·실행 허가·이전 대화와 직접 대조했고 명시적 모순은 발견하지 못했다. 이는 해당 8건의 검수이며 일반화 보증은 아니다.

첫 두 실행의 원본 `AUTOMATED_RESULTS.json`에는 P-03/P-05의 “아직 담기는 시작하지 않을게요/시작되지 않았어요”를 실행 주장으로 잡은 **판정기 오검출**이 남아 있다. 예측을 다시 생성하거나 정답을 바꾸지 않고 `PREDICTIONS.json`을 교정된 `evaluation_v43.py`로 CPU 재채점했다. 부정형만 제외하고 긍정 “담기 시작할게요” 검사는 유지했다. 관련 279개 테스트가 통과했다. 위 표는 이 재채점 수치다. 따라서 원본 자동 결과와 위 표의 숫자가 다른 이유는 모델이 아니라 판정기 교정이다.

P 봉인 8건 중 새 추천 금지 P-06은 사용자가 명시적으로 “새 추천은 하지 마세요”라고 한 경우다. **사용자는 추천을 요청했지만 D/Q가 후보를 만들지 못한 경우**, 그리고 `order.sauce=null`인데 “크림으로 골랐다”고 주장하는 경우는 이 8건의 통과율만으로 충분히 판정되지 않는다. 아래 17턴·반사실 시험은 바로 이 누락된 경계를 별도로 확인한 것이다. 따라서 P 8/8은 전체 Response 합격이 아니다.

Cause: 기존 텍스트 규칙이 “시작하지/시작되지”의 `시작`만 보고 미실행 설명을 실행 주장으로 오인했다.
Effect: P의 실제 안전한 부정 응답을 실패로 세지 않으면서 Q-06 수량 오류는 그대로 실패로 남는다.

## 3. 격리 vLLM 0.32·17턴 누적 회귀

네 조합의 원본 PEFT를 각각 임시 `/tmp/soomac_qp_vllm_20261103_*`에 운영 K→V 변환 경로와 동일하게 변환했다. 각 임시 서버는 포트 **18001**, `VLLM_GPU_MEM=0.45`, 기존 Gemma 4 베이스, D/Q/P 3개 LoRA로 실제 시작했고 모델 목록의 세 경로를 확인했다. 운영 8000 포트·운영 adapter·ROS 발행은 건드리지 않았다. 각 서버를 정확한 PID로 종료한 뒤 GPU와 18001 포트 반환을 확인했다.

| P 후보 | Q 후보 | 복합 6턴 실패 턴 | 분리 11턴 실패 턴 | 전체 통과 |
| --- | --- | ---: | ---: | --- |
| best | best | 3 | 7 | 아니오 |
| best | epoch6 | 3 | 7 | 아니오 |
| epoch6 | best | 6 | 11 | 아니오 |
| epoch6 | epoch6 | 6 | 11 | 아니오 |

**공통 첫 실패:** 기존 D가 “면은 얇은면 보통, 느끼한 소스 추천”에서 `recommendation.action=none`을 출력했다. 네 실행 모두 Q 호출은 **0/17턴**이었다. Q의 standalone 8/8이 실제 대화에서 재현됐다고 주장할 수 없다. P best는 Q 결과 없이 토마토를, P epoch6은 Q 결과 없이 크림을 임의 추천했다. P epoch6은 뒤이어 실제 `order.sauce=null`인데 “크림으로 골랐어요”라고 말한 턴도 있다. 그 뒤 `missing_order`가 발생했다. 이는 모델 문장과 실제 state의 명백한 모순이다.

기존 D의 봉인 실패 5건은 잘못된 면 이름, 크림 수락+즉시 실행에서 `commit` 누락, 게살 알레르기+소시지 즉시 실행에서 `commit` 누락, 실행 거절 필드 왜곡, 주문 유지 요청의 `general` 오분류이다. 누적 시나리오에서는 버섯 즉시 실행 누락, 소시지를 제한과 충돌시키는 상태, 육류/토핑 단계 불일치도 관측됐다. 실패 뒤에도 발화를 넣는 `measure-all`이므로 후반 실패 수는 독립 원인 개수가 아니라 앞선 오류의 전파까지 포함한다. 새 원격 D가 도착하면 같은 봉인 입력과 17턴을 다시 평가해야 한다.

Cause: 기존 D가 추천·복합 commit·문맥을 누락했고 P는 Q 결과가 없는데도 추천/선택 사실을 만들어 말했다.
Effect: 네 조합 모두 현재 실사용 시연 기준 미합격이다. Q epoch6 단일 턴 합격이나 vLLM 로딩 성공을 통합 품질 합격으로 바꾸지 않는다.

## 4. 판정과 남은 관문

### 4.0. 정답 Decision 조건부 재생: 실제 Q epoch6·P best는 6+11턴 통과

기존 **무주입** vLLM 시험에서 P best·Q epoch6은 복합 6턴 중 3턴, 분리 11턴 중 7턴이 실패했다. 원인 분리를 위해 별도 임시 프로세스에서 **Decision 함수 반환값만** 기존 release의 시연 정답 필드로 바꾸고, 동일한 `manual_scenario_stack_probe.run_episode(..., measure_all=True)`를 다시 실행했다. Recommendation epoch6과 Response best는 실제 vLLM 추론 그대로이며 graph, policy, session, history, 주문·실행·물리 단계 검사도 그대로다. 운영 Decision 모델·코드·어댑터는 수정하지 않았다.

| 조건 | 복합 6턴 | 분리 11턴 | 첫 턴 Q 호출·후보 | 의미 |
| --- | ---: | ---: | --- | --- |
| 기존 D 실제 추론 + Q epoch6 + P best | 3/6 통과 | 4/11 통과 | 0회·없음 | 통합 품질 미합격 |
| **정답 D 필드 주입** + Q epoch6 + P best 실제 추론 | **6/6 통과** | **11/11 통과** | 각 episode 1회·크림 | 조건부 하류 기능 확인, 통합 합격 아님 |

정답 D는 첫 턴 `recommendation.action=request`, “크림으로 선택할게. 바로 담아.”에 `confirmation=accept, commit=true`, 버섯 수정·게살 알레르기+소시지·치즈 제외/선택의 명시 필드를 사용했다. Q는 실제로 `proposal.sauce=크림`을 냈다. 첫 턴 P는 “느끼한 소스로 크림을 추천드려요. 크림으로 고를까요?”라고 답했고 `order.sauce=null`과 추천 pending을 유지했다. 이후 크림 확정+실행, 양파 low/버섯 normal 보존·실행, 게살 allergy 보존+소시지 high 실행, 페퍼론치노 제외+치즈 low 실행이 각 턴의 `order_matches`, `execute_matches`, `section_matches`, `history_*`, `plans_match`, `no_model_errors`를 모두 통과했다. 실제 문장도 이 상태와 대조했다. 임시 평가 서버 PID `364867`과 엔진 `365401`을 종료하고 GPU 반환을 확인했다.

**한계:** Decision 정답은 데이터셋의 시연 정답을 참조한 진단용 주입이며 미학습 일반화 평가가 아니다. 물리 완료는 하네스 시뮬레이션이지 ROS/VLM 실물 증거가 아니다. P의 무후보 임의 추천은 앞선 반사실 시험에서 그대로 재현됐으므로 해결됐다고 하지 않는다. 새 원격 D가 실제로 같은 필드를 내는지, 실패/예외 문맥에서도 P가 정직한지는 별도 검증해야 한다.

Cause: 올바른 D 필드가 첫 턴 Q를 호출하고 후속 복합 실행 의도를 전달하자, 동일한 실제 Q/P 어댑터와 Python 그래프가 주문·상태·실행을 끝까지 보존했다.
Effect: 이번 시연의 정상 경로에서는 D 출력이 결정적 관문임을 분리했지만, 기존 D의 실제 실패와 P 무후보 안전성 결함은 운영 합격을 계속 막는다.

### 4.1. 실패 지점의 원본 데이터·프롬프트 국소 대조

기준은 실제 본 학습에 쓴 봉인 release의 `records/{decision,response}/train.jsonl`이다. D의 `DA31-CORE-separate-T01-D`와 `NS03-F01-T01-D`는 정확히 “면은 얇은면으로 보통 양으로 주고, 소스는 느끼한 걸로 추천해줘.”에 `recommendation.action=request`를 정답으로 둔다. 따라서 이번 **기존 D 출력 `none`은 해당 두 정답과 반대**이며, 새 원격 D 결과를 시험한 것이 아니다.

P 학습 행 `NS01-D02-T01-P`는 `recommendation_result=null`, `pending=null`, `order.sauce=null`, `policy.reason=state_update_only`일 때 “소스 추천은 아직 준비되지 않았어요”를 정답으로 둔다. `NS01-D01-T01-P`는 같은 무후보 상태와 `missing_order`에서 “크림으로 요청하셨는데 아직 주문에 반영되지 않아 담기를 시작하지 못했어요”를 정답으로 둔다. 현재 P 프롬프트도 Q 결과가 없으면 특정 재료를 새로 추천하지 말고, structured state가 이전 assistant 발화보다 우선이라고 명시한다. 실제 P best의 토마토 추천과 P epoch6의 크림 확정 주장은 이 **국소 정답·프롬프트와 불일치**한다. 단, 이 두 행이 있다고 전체 P 데이터의 의미 오염이 없다고 판정할 수는 없다.

`NS03-F03-T02-P`의 `recommendation_result=null`인데 “지금은 크림을 추천드린 상태”라는 문장은 별도 `pending.type=recommendation`, `pending.candidate.sauce=크림` 및 직전 실제 추천 대화가 있어 정당하다. 단순히 `recommendation_result=null`만으로 오염이라고 세지 않는다.

추가 격리 반사실 시험: P best(step150)·Q epoch6 staged 조합을 임시 vLLM 18001에 다시 올리고, 실패 첫 턴의 **동일한 실제 Response 입력**을 `build_response_model_input` 재생성 결과와 같다고 assert한 뒤 P만 호출했다. 실제 입력(`recommendation_result=null`, `pending=null`, `policy.reason=state_update_only`)에는 다시 “소스는 토마토를 추천드려요”가 나왔다. 같은 입력에서 **정상적인 크림 후보·추천 pending·`recommendation_proposed`만 주입**하면 “소스는 크림을 추천드려요. 크림으로 고를까요?”가 나왔다. 두 입력 모두 `confirmed_order.sauce=null`, `execution_authorized=false`를 유지했다. 즉 이 P 후보는 올바른 상류 후보가 있으면 첫 턴을 맞추지만, 후보가 없을 때는 금지된 임의 추천을 한다. 이것은 원인 분리 진단이지 새 D·Q가 실제로 그 후보를 만들었다는 증거도, 17턴 합격 증거도 아니다. 평가 서버 PID `355454`와 엔진 `356014`는 종료했고 GPU compute 프로세스가 없는 것을 확인했다.

다운로드 폴더의 `5e5605...-result.tar`와 `8c48ed...-result.tar`는 원격 `unsloth_golden_smoke_s2_20261102`의 job1/job3 결과 이미지임을 각 `results/manifest.json`과 OCI `index.json`으로 대조했다. **새 Decision 6epoch 본 학습 adapter의 도착 증거로 사용하지 않는다.** 현재 로컬 production·vLLM 프로세스는 실행 중이지 않다.

Cause: 실패 입력의 정답과 런타임 프롬프트는 추천 근거·주문 상태를 구분하지만, 현재 붙인 기존 D와 실제 P 생성 문장은 그 구분을 지키지 않았다. P가 왜 규칙을 어겼는지의 학습상 근본 원인은 아직 미확정이다.
Effect: 데이터 전체를 잘못됐다고 단정하거나 Q를 재학습하지 않고, 새 D 도착 뒤 동일 회귀 및 P의 무근거 추천·허위 확정에 대한 별도 표적 평가를 우선한다.

1. **학습/저장/로딩:** P 624step·Q 234step 정상 완료. 두 agent의 best/epoch6 원본 PEFT와 격리 vLLM 변환·로딩 확인.
2. **단일 턴 기능:** Q는 epoch6이 best보다 낫고, P는 두 후보가 봉인 8건에서 동률이다. P best의 validation loss가 더 낮고 누적 회귀 실패가 상대적으로 적지만 그 회귀 역시 미합격이다.
3. **통합 품질:** 기존 D 고정 2×2의 17턴 전부 미합격. 특히 P의 무근거 추천·허위 크림 선택은 별도 보강 대상이다. 최종 운영 후보 확정·adapter 경로 교체는 하지 않는다.
4. **남은 검증:** 새 원격 Decision adapter 도착 후 동일 입력 재평가, D/Q/P 실패 턴의 원시 필드 대조, 필요 시 P 응답 데이터·프롬프트의 무근거 추천/허위 선택 오염을 국소 점검한다. 실제 로봇 완료/ROS 평가는 이번 격리 시험으로 증명하지 않는다.

평가 원본: `/tmp/soomac_qp_eval_20261103_{pbest_qbest,pbest_q6,p6_qbest,p6_q6}` 및 `/tmp/soomac_qp_scenario_20261103_{pbest_qbest,pbest_q6,p6_qbest,p6_q6}`. 각각 `IDENTITY`·`PREDICTIONS`·`RESULT`/`turns.jsonl`을 남겼다. `/tmp` 산출물은 재부팅 시 사라질 수 있으므로 위 경로는 이 시점의 원본 위치다. 운영 교체를 원하면 먼저 별도 승인과 새 D 포함 최종 합격이 필요하다.
