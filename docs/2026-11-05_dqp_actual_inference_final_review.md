# Decision 도착 후 D/Q/P 실제 추론 최종 평가

> **후속 사용자 판정(2026-11-05):** 소스 추천 직후 ‘그걸로 담아줘’에 소스를 선택하고 면 실행을 다시 묻는 것은 정상으로 인정한다. 아래 당시 자동 실패 점수/원문은 보존하며 이1건을 최신 의미 판정의 실행 누락으로 사용하지 않는다. 미도달8턴은 여전히 미검증이었다. 명시적 ‘그걸로 하고 면은 바로 담아줘’ 및 수정 기준의 실제 후속 평가는 [추가 계획](2026-11-05_prompt_multiturn_trial_plan.md)을 따른다. 일반/취향/제한 추출 문제는 별도이다.

결과 정본: [평가 디렉터리](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/) → 원시 입력·출력·상태·계획·SHA와 최종 후보 판정을 보존한다.

상태: 실제 추론·최종 평가 완료. **필수 대본용 조건부 후보 D6/Q6/Pbest150, 전체 범위 합격·운영 채택은 보류. 어댑터 병합 없음.**

## 1. 평가 범위와 데이터 감사

**가중치를 합치지 않았다.** 같은 base에 Decision(D), Recommendation(Q), Response(P)를 독립 LoRA로 붙여 실제 vLLM 호출을 수행했다. 운영 어댑터 경로, 학습, 원격 작업, VLM, 실물 ROS는 변경하지 않았다. 실제 D→Q→P 연결 평가와 D만 실제 추론하고 Q/P를 고정한 평가를 별도 기록했다.

기존 전체 input JSON의 train↔validation/test/challenge/holdout 정확 중복과 동일 input의 target 충돌 후보는 0건이었다. 다만 정규화 현재 발화는 D test 1건, P 총36건이 train과 겹쳤다. 3-gram Jaccard≥0.8 근접 후보는 D27건/P36건이다. 같은 발화라도 상태가 다를 수 있으므로 모두 잘못된 데이터는 아니지만, 템플릿 일반화 점수 과대평가 위험은 있다. 기존 Q/P 평가셋은 반복 후보 선택에 사용됐다. **‘오염 없음’이나 ‘비노출 최종 일반화 정확도’라고 판정하지 않는다.** [감사 원문](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/audit/REVIEW.md).

D validation309행은 입력 builder 재구성·schema 자기검사를 통과했다. 후보 출력 전에 general20/mixed10/task20을 ID hash순으로 선택하고 발화·정답·pending을 검수했다. 의미 정답309행을 전부 확정한 것은 아니다. 기존 Q 정답3건의 low→normal 교정은 이전 평가 복사본의 이력이며 봉인 데이터는 수정하지 않았다.

이번 동결 세트는 D117(기존 정상 상태44+validation50+새23), Q8, P19이다. 새50문항은 train 전체 input/정규화 발화 정확 중복0이며, 봉인 prompt 예시와의 비어 있지 않은 발화/전체 input 정확 중복도0이다. 의미적 독립성을 보증하지는 않는다. 필수 복합6+분리11턴은 train에 포함된 **기능 회귀 시험**이다. 실제 연결은 이17턴과 지시어27턴, 새 general/mixed/수량/제한/취향23턴, 총67턴/9대화로 구성했다. 실패 뒤에는 그 대화를 멈췄으며 미도달을 성공으로 세지 않았다.

Cause: 기존 평가의 반복 선택 노출과 템플릿 중복이 있었다.
Effect: 개발 진단·필수 회귀·새 확장 문항을 분리하고, 입력/정답은 신규 출력 전에 동결했다.

## 2. 모델·자료형·실제 서빙 동일성

도착 tar는 OCI 이미지 아카이브였다. 이미지 실행 없이 마지막 results 레이어의 압축 byte SHA256 `3c8fa1b16d3295cdef839d4cc72042ca5280300aa10c54843f3b9e074809dc11`을 검증하고 필요한 가중치·텍스트만 추출했다. 전체 tar SHA나 원격 base 가중치 전수 SHA를 확인했다고 주장하지 않는다. optimizer/pickle은 역직렬화하지 않았다.

원격 INPUT_IDENTITY의 학습 manifest/release/base config/tokenizer가 로컬 정본과 일치한다. D checkpoint273/546/819/1092/1365/1638의 adapter/config/trainer_state SHA를 검증했다. loss best는 epoch2/step546과 같은 가중치이다. 학습6epoch 완료·skipped_updates0 기록을 확인했으며, loss만으로 후보를 선정하지 않았다.

각 D 원본656개 tensor의 dtype/값은 변환본672개에 그대로 보존했다. Gemma global k_eq_v층의 V LoRA16개만 같은 어댑터 K에서 복사한 vLLM 호환 변환이다. D/Q/P 상호 병합은 없다. 기존 Q6개/P7개 후보도 원본·변환 SHA를 검증했다. 저장된 LoRA tensor는 float32이며, 이것과 계산 dtype BF16은 구분한다.

실제 서버는 내부 설치 환경의 Python3.12.16/vLLM0.32.0/torch 배포 metadata2.13.0/transformers5.19.0/bitsandbytes0.50.2를 사용했다. base는 `gemma-4-12B-it`, NF4/BF16, batch invariance=1, max model length10240, LoRA rank16, xgrammar이다. CPU 클라이언트 Python3.10.20/torch2.12.1+cu130/transformers5.14.1/PEFT0.21.0과 서버 패키지가 동일하다는 주장은 하지 않는다. **현재 실제 vLLM 서빙 환경에서 검증했다.** D/Q/P system prompt와 D/Q schema는 학습 정본과 동일하다.

각 run의 `SERVED_MODELS.json`으로 실제 LoRA 경로를 확인했다. `REQUESTS.jsonl`의 실제 전송 토큰을 CPU로 복호화하여 입력 JSON 값·자료형과 raw 응답을 대조했다. JSON key 순서만 무시했다. 독립 입력의 중립 fixture history와 실제 연결의 생성 history는 구분했다. `D_ONLY_TURNS.jsonl`의 Q/P fixture는 실제 HTTP 추론 수에 넣지 않았다.

Cause: 학습 라이브러리 버전 문자열만 같아도 실제 서빙 경로의 재현을 보장하지 못한다.
Effect: 가중치 출처·호환 변환·prompt·입력 타입·서버 등록 경로·실제 요청/응답까지 연결해 검증했다.

## 3. 후보 비교와 실제 답변

| 비교 조합 | D 엄격/117 | Q/8 | P/19 | 필수17턴 | 연결 필드 통과/관측 | 대화 통과/9 | route까지 일치한 대화/9 |
|---|---:|---:|---:|---|---:|---:|---:|
| [best2_current](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/best2_current/OUTPUTS.md) | 100 | 8 | 19 | FAIL | 48/54 | 5/9 | 4/9 |
| [epoch1](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/epoch1/OUTPUTS.md) | 33 | 8 | 18 | FAIL | 4/13 | 0/9 | 0/9 |
| [epoch2](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/epoch2/OUTPUTS.md) | 100 | 8 | 19 | FAIL | 48/53 | 5/9 | 4/9 |
| [epoch3](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/epoch3/OUTPUTS.md) | 99 | 8 | 19 | PASS | 57/59 | 7/9 | 6/9 |
| [epoch4](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/epoch4/OUTPUTS.md) | 99 | 7 | 19 | PASS | 53/57 | 6/9 | 4/9 |
| [epoch5](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/epoch5/OUTPUTS.md) | 96 | 8 | 19 | PASS | 58/59 | 8/9 | 6/9 |
| [epoch6](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/epoch6/OUTPUTS.md) | 96 | 8 | 19 | PASS | 58/59 | 8/9 | 6/9 |
| [final_d5_q6_pbest150](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/final_d5_q6_pbest150/OUTPUTS.md) | 재사용 | 재사용 | 재사용 | PASS | 58/59 | 8/9 | 6/9 |
| [final_d6_q6_pbest150](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/final_d6_q6_pbest150/OUTPUTS.md) | 재사용 | 재사용 | 재사용 | PASS | 58/59 | 8/9 | 6/9 |

최종 후보는 **D epoch6/step1638 + Q epoch6/step234 + P best150**이다. 이는 필수 대본용 조건부 후보이며, 범용 전체 합격/운영 채택은 보류한다. 원본·vLLM 변환 경로/SHA는 [SELECTION.json](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/SELECTION.json)에 있다. D5도 같은 기능대화8/9였으나 독립 mixed 문항에서 면 종류를 누락했고 D6는 맞혔다. 이 한 비교로 통계적 우월성을 주장하지 않는다. Q6/Pbest150은 기존 후보를 유지할 근거를 얻었으며 다른 모든 후보보다 우월하다는 시험은 아니다.

최종 실제 연결은 **58/67턴 통과, 1턴 실패, 8턴 미도달**이다. 관측한59턴 중58턴이 주문·계획·Q후보/clarify 필드 검사를 통과했다. route3건 불일치까지 포함하면55/59턴,6/9대화가 엄격하게 맞는다. 독립 Decision+고정 Q/P의 필수17턴도17/17이다. 실제 연결의 필수17턴은 Q를 실제 호출하며 P생성9턴·고정 안내8턴으로 나뉜다.

[필수17턴 입력·답변·계획 표](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/final_d6_q6_pbest150/REQUIRED_SCENARIO_OUTPUTS.md) · [전체 실제 답변과 D raw](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/final_d6_q6_pbest150/OUTPUTS.md) · [개별 D 실패 의미 검수](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/runs/epoch6/D_MISMATCH_REVIEW.json).

D6 독립 입력의 route별 엄격 점수는 task68/82, general16/23, mixed12/12이다. task의 엄격 불일치에는 정상 execution pending의 accept↔commit 및 추천 후보 수락의 동등 표현도 포함된다. route만 비교하면 task81/82, general16/23, mixed12/12이다. general 분류가 아직 약하다는 근거이며96/117을 그대로 실제 행동 정확도로 부르지 않는다.

전체9개 완료 run의 실제 HTTP 요청 **1852건**에 대해 전송 입력 값·자료형 대조를 통과했다. 응답 raw는1851건이 trace와 일치하며, 나머지1건은 D2의 중복 JSON key 파싱 오류로 trace raw가 없어 직접 대조하지 못했다. 해당 HTTP 원문과 오류는 보존했다. 독립 후보 평가는 D702회/Q48회/P133회이며, 같은 문항을 후보별 비교한 횟수다. D2와 Q6의 동일 독립 입력은 재사용해 중복 GPU 추론하지 않았다.

독립 입력의 클라이언트 관측 지연은 decision: 중앙값0.549초, 평균0.614초, 범위0.225~3.400초; recommendation: 중앙값1.272초, 평균1.418초, 범위1.071~2.029초; response: 중앙값0.992초, 평균1.249초, 범위0.533~2.663초. 서버 기동·STT/TTS·로봇 시간을 제외하며 새 HF 대조 실험이 없어 이번 수치로 HF 대비 배수를 계산하지 않는다. 연결 턴은 D/Q/P 및 repair 호출 수에 따라 달라진다.

D 엄격 점수는 동등한 accept/commit 표현도 불일치로 세므로 의미 정확도로 바꾸어 읽지 않는다. Q/P 자동 검사도 문장 의미 전수 보증은 아니다.

원래 연결 평가기의 order/plan 검사가 Q 호출·추천 후보 저장·clarification pending 일부를 검사하지 않는 것을 발견했다. 원본 raw/점수는 보존하고 모든 후보에 같은 `FIELD_AUDIT.json`을 추가했다. D4 취향 추천 첫 턴은 원래 PASS였지만 실제 Q 호출0·후보없음으로 실패다. D2의 모호한 변경 뒤 clarification 누락도 이 감사로 드러났다. route 불일치는 이 추가 필드 점수와도 별도로 기록한다.

Pbest150과 후반 epoch의 새 정상 입력19건에서는 주문·수량·선택/완료 사실의 모순을 관측하지 않았다. 일부 후반 P는 확인 질문을 생략한다. 과거 Pbest150의 실행 턴 수량 환각은 이번 결과로 없어지는 것이 아니다. **현재 실행 안내는 Python fixed를 유지했고, 그 턴은 P 생성 성공으로 세지 않는다.** 비실행 설명·추천 안내·일반 대화는 실제 P 출력이다. 일반 지식 답변의 역사/식품화학 주장을 외부 전문가 기준으로 전수 검증한 시험은 아니다.

Cause: 후보별 raw D의 작은 필드 차이가 누적 상태와 실행 계획에 다른 결과를 만든다.
Effect: loss 순위·문자열 점수와 실제 필수/멀티턴 수행 능력을 따로 비교한다.

## 4. 확인된 문제와 채택 제한

| 문제 | 직접 증거 | 영향·심각도 | 판정 및 후속 범위 |
|---|---|---|---|
| D2의 수량 변경 누락 | 복합4 ‘버섯만 보통 양으로 바꿔서 바로 담아’에 commit만 출력 | mushroom1 계획, 기대2. 중요 실행 오류 | D2 필수 시나리오 후보 제외. 추후 수정 시 수량+commit 복합 입력을 회귀 관문으로 유지 |
| D2/P2의 모호한 수락 오실행 | ‘그거 말고 다른 걸로 바꿔줘’ 뒤 clarify 누락, 이어 ‘응’ 수락 | 이전 onion1/mushroom3 계획 승인. 실물 실행은 없음 | 해당 조합 제외. 모델 추출과 pending/질문 전달 계약 모두 후속 검토 대상 |
| 추천 수락+실행 누락 | 후반 D의 추천 직후 ‘그걸로 담아줘’ → `confirmation:accept`, commit 없음 | 크림 선택만 하고 면 실행을 다시 질문 | 범용 멀티턴 전체 합격 차단. 명시적 ‘크림으로 선택할게. 바로 담아’ 통과와 구분 |
| 일반/상황·제한 추출 | D5/6의 일부 감정 대화 task 분류·음식 취향 추가; SCV2-D-0094 cannot_eat 제한 누락 | 일반대화 계약 오분류, 강한 제한과 취향의 차이. 이 문항의 실제 위험 재료 실행은 미측정 | 정상 식단 주문 통과를 전체 분류/제한 완벽성으로 확대하지 않음 |
| D4/Q4 추천 및 D5 필드 누락 | D4 Q호출 누락, Q4 새콤한 소스 빈 proposal; D5 독립 mixed 입력 noodle_type 누락 | 후보 미저장/추천 부재/주문 불완전 | Q4 제외 근거. D5와 D6의 동점 연결 점수를 의미적으로 구분 |

D5/6의 모호한 ‘응’ raw 수락은 runtime guard/repair에서 차단됐다. 이 회복 결과와 D 자체의 정답을 구분한다. 단순 execution pending의 accept는 runtime에서 commit으로 처리되지만, recommendation pending의 accept는 후보 선택만 한다. 이 차이가 짧은 실행 요청 실패의 직접 경로다. 모델이 그런 토큰을 선택한 내부 원인은 미확정이다.

실패 뒤 학습 자료를 대조한 결과 추천 pending+commit 정답51행, 복합 버섯normal+commit 동일 문장2행이 있었다. ‘그걸로 담아줘’ 정확 문장이 없다고 해서 관련 행동 자체를 학습하지 않았다고 결론 내리지 않는다. 해당 조회로 평가 정답을 바꾸거나 학습을 다시 돌리지 않았다.

Cause: 명시적 실행·추천 수락·모호성·제한을 구분하는 출력 필드가 일부 입력에서 누락됐다.
Effect: 필수 대본 통과 후보를 제시하되, 모든 멀티턴/제한 판단에 대한 무조건 합격이나 운영 교체는 하지 않는다.

## 5. 완료 조건과 남는 한계

원본 archive/checkpoint/봉인 데이터/실패 로그를 보존한다. 최종 선택은 평가상 후보 추천이며 운영 경로 변경·가중치 merge가 아니다. 실제 로봇 움직임, VLM 검증, STT/TTS, 물리 완료 성공은 이번 결과에 포함하지 않는다. 완료 이벤트는 명시한 offline fixture이다.

CPU 최종 회귀는3 tests+7 subtests 통과했다. 첫 pytest 명령은 자동 로드된 ROS launch_testing 플러그인이 lark를 찾지 못해 수집 전에 실패했다. 관계없는 플러그인 자동 로드만 끈 평가 명령으로 검사했으며 기존 환경에 패키지를 설치하지 않았다. 서버 최초 재시작의 TIME_WAIT 검사 오류와 평가 입력 준비의 공유 사전 오류는 원래 로그를 보존하고 평가 도구에서만 해결했다.

최종 확인: 소유9개 scope는 모두 inactive/dead이며 각 종료 코드0이다. GPU compute 프로세스 없음, 8000/18001 listener 없음, GPU 전체 메모리1303MiB는 데스크톱 수준이다. [정리 증거](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/FINAL_CLEANUP.json). 동결 source/processor SHA와 운영 config·model_runtime·launcher의 HEAD 일치 검사를 통과했다. 운영 Decision은 기존 v5 경로 그대로이며 새 D6를 적용하지 않았다. `git diff --check` 통과. 평가 회귀3 tests+7 subtests 통과.

원본 `RESULT.json`의 semantic_review=PENDING은 당시 자동 실행 상태로 보존했다. 수동 검수 결과는 별도 `SEMANTIC_REVIEW.json` 및 기존 best2_current의 `MANUAL_REVIEW.md`에 기록했다. `COMPARISON_PROGRESS.json`은 중간 생성물이고 정본은 [FINAL_EVALUATION.json](../training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/FINAL_EVALUATION.json)과 이 보고서이다.

**평가 작업5/5 완료. 모든 요구 범위의 시험·실패 기록·조건부 후보 선정은 끝났으며 모델 수정·학습·운영 채택은 수행하지 않았다.** 남은 실패는 미완료 평가를 숨긴 것이 아니라 최종 평가의 불합격/제한 결과다.
