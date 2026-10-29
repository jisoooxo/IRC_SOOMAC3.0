# IRC LLM Final — 계정 전환 인수인계

**2026-10-29 추가 읽기:** [그래프 역할 분리 구현 결과](../training/learning_first_v43/final/audits/2026-10-29_graph_refactor/RESULT.md) → [계획 §10](<IRC LLM Final graph refactor plan.md#10-현재-상태와-다음-한-단계>) → [실행 영수증](../training/learning_first_v43/final/audits/2026-10-29_graph_refactor/TEST_RECEIPT.md). 사용자 승인 범위 R1~R5 완료. 최종 778건·2subtests 통과, 독립 Codex sol/high 찬성·신규 차단0. Recommendation 실제 독립 노드와 로그 메서드를 분리했으며 기존 기능·계약을 대조했다. 학습/원격/adapter/VLM/ROS 정책 변경 없음. 기존 이중 오류 로그 누락/schema export 한계는 RESULT에 따로 기록했으며 변경하지 않았다. 아래 10월26일의 실행 상태는 당시 이력이다.

[실행 계획 정본](</home/roma/IRC_SOOMAC3.0/docs/IRC LLM Final plan.md>) · [오늘 검수 원본 모음](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/) · [외부 학습 작업장](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/final/)
→ **2026-10-25 하루 동안 수행한 데이터·코드·프롬프트·과거/최신 실로그·GPU·멀티에이전트 검수를 함께 이어받는 문서이다. 마지막 두 세션만 요약한 문서가 아니다.** 작성일 2026-10-26 KST.

## 0. 최신 요구 — 변경 불가 기준

**현재 인계(2026-10-26): GL05·GL06 명시 범위 완료, BATCH02 종료.** 다음 작업자는 `final/goals/GL06/RESULT.md` → `RECURRING_FAILURE_CONTRACT.md` → `TOKEN_BUDGET_REPORT.md` → `FEWSHOT_MANIFEST.json` 순으로 읽는다. 저장소 staging은 `training/learning_first_v43/final/goals/`, 외부 정본은 기존 SOOMAC_TRAINING_WORKSPACE의 동일 final/goals이다. 전체612pass·8subtests,32개예시와23개system시연/실제serializer/17개Decision replay·최종Codex찬성 증거를 보관한다. history/context와 이전 GL04/05 봉인은 보존했다.

반복 오류인 근거 없는 단계 진행 주장과 pending/next_prompt 없는 실행 질문은 예시교정·prompt계약·추가12회귀로 고정했다. 잘못된 assistant 질문 뒤 모호한 동의도 검사하되 정상실행수락/명시commit은 유지한다. **실제 자유생성 재발0을 입증한 것은 아니다.** GL07~13에서동일오류target FIX/DROP/HOLD 및새모델채택차단기준으로이어받는다. 기존trainer4096은현재prompt와불일치(32개중24개실패),격리16384후보의32개target masking/완결/추가trim0만검증했다. 새data/전체trainer/GPU/학습검증은후속필수다. 원격job·adapter·VLM·ROS·commit/push변경없음. GL07별도승인전착수금지. 아래미완료기록은과거이력이다.

**2026-10-26 최우선 정정: 프롬프트 개선은 계속하되 기존 adapter 성적을 새 모델의 최종 판정으로 사용하지 않는다.** GL05 코드·입력/발행·안전 계약과 GL06 prompt/few-shot·토큰 검증을 분리한다. 독립 oracle·복합 멀티턴 전후 필드·serializer·history 보존·편향/누수가 현재 기준이다. 기존 GPU 실패는 기록/데이터 보강 근거이지 아직 학습하지 않은 adapter의 불합격이 아니다. 새 모델 성능 NOT_RUN, GL05 미완료·GL06 준비만 완료를 유지한다. 다음 작업자는 기존 adapter 반복 튜닝을 자동 재개하지 말고 `training/learning_first_v43/final/goals/GL05/ACCEPTANCE_POLICY.md`와 최신 NEXT부터 읽는다.

**2026-10-26 최신: 담당은 LLM만이다. VLM 프롬프트·테스트·판정 로직은 건드리지 않는다.** 후속 작업자는 저장소 `AGENTS.md`를 먼저 읽는다. GL05는 구현/검증 진행 중이고 GL06은 준비 산출물만 작성됐다. 기존 VLM 변경은 별도 작업자 소유로 보존한다. 전체 해시 검사 실패를 없애려고 VLM을 되돌리지 않는다. 최신 상세는 GL05/STATE·WORKLOG·NEXT를 따른다.

2026-10-26 GL04 최신: **상태·실행 권한 범위 완료, Codex최종찬성.** 수량 enum·강한 cannot_eat·복합 pending/commit·명확화 권한을 구현했다. 코드493pass·8subtests / controlled20episode63turn / 기존어댑터GPU12상태검사12통과. B2raw오판은fallback으로막았고빈추천도남으므로모델이해/직원형성능완료아님. 새 SessionState/Decision 최상위필드없이pending.question_type·deferred만보강. `final/goals/GL04/RESULT·manifest·NEXT.md`와진행목록우선. 아래미구현/NOT_RUN은과거이력. GL05~15미착수·자동학습/운영교체/실물ROS/commit/push없음.

**최종 목표는 이전 대화와 현재 주문·선호·제약·실제 로봇 상태에 근거해 자연스럽게 권유하고 정확하게 실행하는 직원형 대화이다.** 메뉴 키워드를 외운 단일 턴 분류기나 “저장했어요”만 말하는 시스템이 아니다. 자연스러움은 길게 말하기가 아니라, 해당 상황에 필요한 설명과 다음 대화를 정확하게 이어가는 것이다.

사용자의 최신 추가 강조: **멀티턴이 잘 된 사례도 있다. 기존 멀티턴을 보존한다. 학습 응답을 한 줄 저장 보고로 획일화하지 않는다.** 현재 결함을 전체 멀티턴 기능 고장으로 일반화하지 않는다.

| 반드시 유지 | 하지 말아야 할 일 |
|---|---|
| history 저장·전달·역할·순서·기존 창/길이 처리, 기존 성공 멀티턴 | 오염된 것 같다고 history 삭제·요약·마스킹·축소, 한 턴 추론으로 복귀 |
| 모델이 자연어 의도와 멀티턴 참조를 판단 | Python 키워드 목록으로 사용자 의도 재판정, 새 provenance/state 신설을 기본 해법으로 사용 |
| schema·지원 메뉴/수량·물리 불가능 변경·강한 제한·pending·중복 실행 안전 검증 | 자연스러운 대화를 만든다는 이유로 안전 검증 제거 |
| 기존 ROS 토픽·메시지·lid/sauce 자동 진행·실행 중 STT 차단 | 임의 소스 재확인, VLM 정책 변경, 로봇 실행, 새 ROS 명령을 Response에 위임 |
| 10월 11일 기준 Python 고정 발화만 유지하고 이후 추가분 모델 경로 복원 | 파일 전체 rollback, 새 고정 멘트를 fallback이라는 이름으로 증설 |

학습의 잘못된 target·중복·복구 불가 오염은 활성 train/validation/test에서 제외한다. 원본은 별도 보존하며, 정제와 Runtime history 삭제를 혼동하지 않는다. 기존 전체를 폐기하거나 무조건 세 Agent를 모두 재학습하는 결론도 먼저 내리지 않는다.

원인: 정상 동작한 제어·멀티턴과 실패한 의미 판단·발화가 같은 시스템에 공존한다.
결과: 성공 사례를 보존하는 회귀와 실패 교정을 함께 수행해야 하며 “아무것도 실행하지 않는 안전한 모델”도 불합격이다.

## 1. 다음 담당자가 시작하는 순서

### 1.1 읽을 순서 — 5개 묶음

| 순서 / 예상 읽기 시간 | 문서 | 읽는 목적 |
|---|---|---|
| 1 / 10–15분 | **이 문서 전체** | 최신 목표, 오늘 전체 결론, 정정된 지적, 미완료 작업을 먼저 파악 |
| 2 / 15–25분 | [IRC LLM Final plan](</home/roma/IRC_SOOMAC3.0/docs/IRC LLM Final plan.md>) + [Astra 계획 검수](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-25_final_plan_review/ASTRA_PLAN_REVIEW.ko.md) | W0–W9·G0–G4, 금지 사항, B1/B2 호출·질문 계약. 계획 검수는 이전 406행 버전에 대한 판정임 |
| 3 / 20–30분 | [통합 조정자 판정](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-25_llm_goal_contract_reset/COORDINATOR_ADJUDICATION.ko.md) + [운영 stack 보고](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-25_active_stack_regression_audit/REPORT.ko.md) + [그 보고의 재판정](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-25_active_stack_regression_audit/ADJUDICATION.ko.md) | Codex·Claude·Astra의 교차검수와 오탐 철회. 원문보다 최종 재판정을 먼저 적용 |
| 4 / 15–25분 | [3담당 데이터 감사](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-25_multiagent_audit/REPORT.ko.md) + [미래 scope 진단](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-25_future_scope_audit/DIAGNOSIS.ko.md) + 아래 §6 coverage JSON | 확정 데이터 결함, 구조/언어/의미/편향 검수 범위를 구분 |
| 5 / 15–25분 | [225109 최신 판정](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-25_followup_plan_bundle/LATEST_ADJUDICATION.ko.md) + [232040·234046 보고](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-26_latest_sessions/README.md) + [Astra 최신 로그 검수](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-26_latest_sessions/ASTRA_LOG_REVIEW.ko.md) | 마지막 실로그의 실패/성공, 다음 수정의 직접 회귀 근거 |

원본 상세 보고는 §2 목록에서 담당 영역별로 추가 열람한다. 이전 문서의 “전수 PASS”, “Docker만 남음”, “최신 세션 미검수”는 작성 시점 기록이며 현재 상태로 복사하지 않는다. 최신 사용자 지시 → 최종 계획 → 이 인계의 상태/증거 연결 → 조정자 재판정 → 개별 worker 원문 순으로 충돌을 해소한다. 증거와 최신 지시가 실제 충돌하면 숨기지 말고 결정 항목으로 남긴다.

### 1.2 시작 환경과 경계

저장소 `/home/roma/IRC_SOOMAC3.0`, 작업 branch `llm_final`. 문서 작성 시작 HEAD는 `a4c522a0da772c631c6807b8bced0f1e5a5e6105`(로그 및 md 백업)이며 당시 `git status --short`는 비었다. 이후 문서 수정은 미커밋 변경일 수 있다. 새 담당자는 직접 다시 확인한다. 원본 `flexible_llm`을 수정하지 않는다.

```bash
cd /home/roma/IRC_SOOMAC3.0
git status --short
git branch --show-current
git log -1 --format='%H %s'
```

이 명령은 셸 명령이며 별도 Python 변수/import를 요구하지 않는다. 변경 파일을 읽기 전에 `/home/roma/AGENTS.md` 및 경로별 지침을 확인한다. 사용자/다른 작업자의 변경을 되돌리지 않는다. 이전 `tts_node.py` 변경은 사용자 변경이다.

새 계정에서는 예전 Orca Run/dispatch/터미널 ID를 살아 있는 권한으로 재사용하지 않는다. 앱·인증·현재 소유권을 조회하고 전용 검수 세션을 새로 연결한다. 사용자는 Codex 작업자에 **gpt-6-astra / high**를 요청했다. 실제 launcher의 effective 값을 확인하며 프롬프트에 모델명만 써서 변경했다고 말하지 않는다. 계정 토큰·HF 토큰은 문서/이미지/명령 출력에 넣지 않는다.

## 2. 현재 확인된 기준점과 증거

### 2.1 오늘까지 무엇을 했고 무엇이 남았나

| 작업 | 실제 상태 | 다음 담당자가 오해하면 안 되는 점 |
|---|---|---|
| 초기 8,230행 release·Docker r4 | 과거 생성·검사·이미지 기록 있음 | 당시 자동 PASS 이후 의미 결함 발견. 최종 새 목표의 합격본 아님 |
| 8,500행 scenario_context_v2 | 270행 보강 + 기존 8행 reason 교정, 초기 평가기 개선 구현 기록 있음 | 오늘 아무 코드/데이터도 안 바뀐 것은 아님. **후속 감사에서 발견한 결함 교정은 아직 미완료** |
| Codex/Claude 후보 검수 보고 회수 | D 5,671 + Q 723 + P 2,106, manifest 누락 ID 0 기록 | 전행 판정 제출이지 무결점·수정 완료가 아님. dataset_ready=false |
| 과거 구성 GPU 재시험 | 현재24턴, 현재코드+옛adapter27턴, 옛코드+옛adapter27턴 실행 기록 | 최신34턴 검증은 새 GPU가 아닌 기록 출력 재생. 세 시험을 완전히 동일 입력의 인과 A/B로 보지 않음 |
| 최종 계획·로그 진단 | 계획 작성/조건부 검수, 최신34턴 상세검수·34/34 재생·160개 기존 테스트 통과 | 최종 Runtime 교정·새 release 봉인·새 GPU 채택 검증·새 Docker는 남음 |

원격 학습은 사용자가 마지막으로 제공한 step 16/2454 로그에서 시작이 확인됐었다. 그 뒤 현 상태·완료 여부는 **UNKNOWN**이다. 현재 로봇 모델과 원격 학습 adapter를 혼동하지 않는다. 이 인계 작업은 원격 중단·재학습·adapter 교체를 하지 않는다.

### 2.2 멀티에이전트 검수 전체 지도

아래 경로는 모두 `training/learning_first_v43/reports/` 아래이다. 원문 지적은 해당 폴더의 조정자 재판정과 함께 읽는다. 숫자를 합산해 독립 검수 행 수로 만들지 않는다.

| 검수 묶음 | 대표 자료 / 보강된 내용 | 상태·한계 |
|---|---|---|
| 초기 Claude↔Codex 교차검증 | `2026-10-25_claude_cross_review/VERIFIED_REVIEW.ko.md`, `2026-10-25_codex_cross_review/CODEX_REVIEW.ko.md` | 평가기 핵심 4반례·추가 반례. “Docker만 남음” 철회 근거 |
| 정합성·의미·편향 3담당 + Claude | `2026-10-25_multiagent_audit/{REPORT.ko.md,integrity/REPORT.ko.md,semantics/REPORT.ko.md,bias/REPORT.ko.md,CLAUDE_DATA_REVIEW.ko.md}` | 전행 구조 검사, 475행 의미 집중검수, pending/Response 추가 sweep. Claude 보강270행 범위. 이후 전행 검수와 구분 |
| 전체 후보·과거 데이터 후속 검수 | `2026-10-25_future_scope_audit/`, `2026-10-25_active_stack_regression_audit/` | D/Q/P 판정 회수, Response C 재심, Oct11 원본4,749행 선언 범위 확인. 오탐 철회는 ADJUDICATION 우선 |
| 계약·데이터·구조·로봇 로그 5축 | `2026-10-25_llm_goal_contract_reset/ORCA_{CONTRACT,DATA,ARCHITECTURE,LOG,ROBOT_RUNTIME}_REVIEW.ko.md`, `COORDINATOR_ADJUDICATION.ko.md` | 구 Codex 계약 검수 + Astra 구조/최신로그 + Claude 데이터/로봇로그·Git 이력 보완. 전역 의미 전수 합격 아님 |
| 후속 live·최종 계획·마지막2세션 | `2026-10-25_followup_plan_bundle/`, `2026-10-25_final_plan_review/`, `2026-10-26_latest_sessions/` | 225109 명확화/pending, B1/B2, 마지막34턴 검수. 최신 평가기 반례 별도 Astra 재검수는 consumer_fenced로 미실행 |

추가 증거: `2026-10-25_historical_log_coverage_audit/REPORT.ko.md`, `2026-10-25_oct11_runtime_regression/REPORT.ko.md`, `2026-10-25_user_scenario_regression/REPORT.ko.md`, `2026-10-25_training_scenario_worklog.md`. 세부 reviewer 원문·프롬프트·JSON 증거를 보관하며 이 인계 때문에 덮어쓰지 않는다.

### 2.3 로그 분모와 실제 GPU를 구분한다

| 자료 | 확인 범위 | 의미 |
|---|---|---|
| Oct11 원본 | 19세션139턴/332이벤트, prompt271회·8변형 대조 | 전체 파싱·prompt 계보. 모든 턴 신규 의미 정독 인증 아님 |
| Oct25 초기 운영 stack | 38턴/80이벤트, 65호출 prompt/history 일치, Graph38/38 재생 | 현재 입력이 빠진 문제와 모델 출력 실패 분리 |
| Oct25 후속 동결 | 13세션116턴, 그중225109 19턴39이벤트 후속 검수 | 동결 이후 로그는 포함하지 않던 과거 분모 |
| Oct25 마지막 동결 | 15세션150턴306이벤트 구조 오류0; 최신232040·234046 34턴68이벤트 상세검수 | 150턴 전체를 같은 깊이로 의미 전수검수했다는 뜻 아님 |
| Git 과거 추가 | 40세션452턴 구조 분류·대표본문, 당시 전체 inventory70세션688턴 | 중첩/날짜가 다른 집계. 688+150으로 합산 금지, 실제 로봇/재생 여부 전체 확정 금지 |

[Oct11 GPU 보고](/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/reports/2026-10-25_oct11_runtime_regression/REPORT.ko.md): `b02a0f0` snapshot + Decision v3.2 epoch4 + Response v1 epoch1 + Recommendation Base 설정. 27턴 전체112.6초, 한 턴 평균3.904초. Recommendation은 해당 실행에서 요청이 나오지 않아 호출되지 않았다. 면 선택·야채 둘 다 low 번복은 성공했으나 첫 추천/선호 추출과 게살 제외 후 나머지 선택은 실패했다. 후속 토핑 단위시험을 연속 시나리오 완주로 세지 않는다. ROS는 mock이었다.

## 3. 최종 동작 계약

### 3.1 사용자 시나리오와 직원형 대화

아래 발화는 문체 예시이지 고정 정답 문자열이 아니다. 각 질문은 실제 pending 종류·대상과 일치해야 한다.

| 상황 | 원하는 상태·동작 | 자연스러운 표현 예시 |
|---|---|---|
| “매운 거 못 먹어. 기분도 별로인데 추천해줘” | 식이 요구와 추천을 함께 처리. 남은 코스 중 유효 조합을 미확정 추천 | “맵지 않은 크림 소스에 버섯과 소시지를 곁들이면 어떠세요?” — 유제품/육류 제한이면 이 예시 금지 |
| “크림으로 할게. 얇은 면 조금만” | 소스 사전 선택·면 low. 일반 주문만으로 실행하지 않음 | “크림 소스로 골랐고, 면은 얇은 면을 조금 선택하셨어요. 지금 면을 담을까요?” |
| “버섯 조금, 양파 많이 → 아니 둘 다 조금” | 두 항목 모두 low, pending도 최신값 | “양파도 조금으로 바꿨어요. 버섯과 양파를 둘 다 조금씩 담을까요?” |
| “게살 알레르기 있어. 고기 많이 먹고 싶은데 추천해줘” | allergy+수량 요구+추천을 보존, 허용 후보만 제안 | “게살은 빼고 소시지를 넉넉하게 추천드릴게요. 이걸로 고르시겠어요?” 선택 질문과 실행 질문을 구분 |
| 토핑 도착 / 맵지 않게 유지 | 관련 기억만 활용. 선호-only는 제안, 확정 주문은 실행 확인 | “매운 건 못 드신다고 하셨으니 페퍼론치노는 빼고 치즈로 할까요?” 이미 치즈 확정이면 “미리 고르신 치즈를 담을까요?” |

반대 선호 “매운 게 좋아”도 유지되면 나중 토핑에서 페퍼론치노를 제안할 수 있어야 한다. 철회됐거나 강한 제한과 충돌하면 제안하지 않는다. 긍정·부정, 반말·존댓말, 조사 생략, 어순 변경, 다른 메뉴·수량·기분에도 같은 의미 계약이 작동해야 한다. 특정 시나리오 문장만 외우는 성공은 불충분하다.

### 3.2 멀티턴 성공을 반드시 보존

225109 T9–10의 둘 다 low 변경→실행, 234046 T10의 변경+즉시 실행, T11–12의 게살 제한+소시지 high→pending 수락, 232040 T12의 추천 수락+실행은 보존할 성공 사례이다. 과거 Oct11 GPU에서도 야채 번복이 성공했다. 현재+미래 추천 주입→수락→미래 주문 저장→도착 단계 preselected 확인 경로도 모의 재생에서 성립했다. **기능이 전부 사라졌다는 결론은 반증됐다.**

history는 실제로 전달됐다. 초기65회와 최신60회 기록 입력에서 prompt 일치 및 history 전달이 확인됐다. 다만 처음 매운맛 요구를 지속 state에 저장하지 못하거나, 최신 오일을 입력받고 옛 토마토를 추천하거나, 자기 오답을 반복하는 실패가 있다. 오염된 assistant 발화가 실패의 직접 원인이라는 인과는 A/B 전까지 미확정이다.

원인: 입력 전달·지속 state 추출·맥락 해석·응답 사실성은 각각 다른 단계이다.
결과: history를 없애지 말고 어느 단계가 처음 틀렸는지 기록하여 해당 계약·프롬프트·데이터를 교정한다.

### 3.3 수락·즉시 실행·물리 상태

일반 “넣어줘”는 선택 후 확인 대기, 새 주문+“바로 담아줘”는 검증 후 즉시 실행, 기존 실행 확인 뒤 “응/바로 해줘”는 해당 최신 대상 수락이다. 같은 “응”이라도 추천 선택·명확화·실행 질문에 따라 달라진다. 추천+즉시 실행 복합 의도를 조기 return으로 잃지 않는다. 지나간/이미 담긴 재료를 자연어 취소만으로 실제 제거했다고 기록하지 않는다.

VLM FAIL→재시도→policy_success는 제어 진행이지 관측 PASS가 아니다. 최신2세션에서 치즈/소시지/오일3작업에 2회FAIL 뒤 진행이 있었다. 실물 투입 여부는 원본 영상 없이 확정하지 않는다. 기존 안내는 “재시도를 마쳤어요. 다음 단계로 진행할게요”이며 이를 “실제로 담았다고 거짓말했다”로 바꾸어 보고하지 않는다. 다른 Response 상태 환각은 별도 확정 사실이다.

## 4. Runtime 변경 명세 — 확정 문제와 우선순위

| ID | 확인된 문제 / 최초 실패 위치 | 다음 작업·보존 경계 |
|---|---|---|
| R01 | 최신234046 T1–6 raw `noodle_portion="많이"`; schema string 허용 후 domain에서 거절. commit도 양 오류로 차단 | schema/prompt/데이터 수량 계약을 low/normal/high로 일치. Python 한국어 치환으로 덮지 않음. T5 raw값 하나 high로 바꾸면 실행 허용됨은 모의 대조일 뿐 모델 교정 아님 |
| R02 | 2104 low/low를 high/low로 설명, 232040 T6 오일 변경을 부정, 없는 pending 주장 | Response에 전달된 최신 상태/applied 사실과 부정·시제·수량·대상을 맞춤. 상태가 맞으면 Decision을 불필요하게 바꾸지 않음 |
| R03 | Recommendation raw부터 좁은 제안/옛 소스; Python reason을 현재 단계로 치환하고 Response 우회 | broad/narrow 대조와 확정 주문 유지. scope는 이미 현재+미래를 포함. 이유 의미를 Python 키워드로 다시 만들지 않음 |
| R04 | 알레르기 options를 모델이 무시하고 Node 고정 목록도 게살 재안내; skip 질문과 실제 치즈 pending 불일치 | 발행 문장과 실행 대상 둘 다 검증. 제한 실행 차단 성공과 위험 권유 실패를 분리. 모든 재료명 언급 금지 아님 |
| R05 | mock에서 cannot_eat 매운맛이 선호로 downgrade, commit+recommendation 조기 return, clarify와 옛 execution 질문 충돌 | 강한 reason의 권위 보존·복합 의도·질문 우선순위 테스트. 모든 “못 먹어”를 무조건 allergy로 분류하지 않음 |

B1: **선호-only 사용자 턴**에도 자연스러운 다음 대화 경로가 필요하다. 적절한 새 추천은 Decision의 기존 recommendation 요청→Recommendation→validator→Response를 거치고, 기존 유효 pending은 재사용한다. 선호만으로 주문/실행을 만들지 않는다. 매번 자동 추천하거나 질문을 강제하지 않는다.

B2: **미해결 정정의 명확화가 옛 주문의 실행 확인보다 우선**이다. 보관 pending을 삭제하는 것과 이번 질문 의도를 구분한다. “갑자기 적게”→대상 명확화→“응”만으로 옛 high 실행 금지, “둘 다 적게”→최신 low→명시 실행은 성공해야 한다. 키워드 guard나 history 삭제로 해결하지 않는다.

파일 진입점: `src/soomac_irc/soomac_irc/{agent_contract.py,agent_prompts.py,llm_langgraph.py,llm_policy.py,llm_langgraph_node.py,decision_model.py,recommendation_model.py,response_model.py,evaluation_v43.py}`. 한 공유 파일은 한 명만 수정한다. 기존 네 추가 고정 분기 제거와 함께 prompt의 “거절은 Response에 오지 않는다” 지시도 바꿔야 한다. 이벤트 입력은 가짜 user 발화로 만들지 않는다.

## 5. 엄격한 프롬프트 작성·검증 기준

최신 사용자 정정, 현재 state, 과거 assistant 추측, 물리 실행 증거의 권위를 구분한다. history 오염 대응은 이 해석 원칙을 프롬프트에 명시하는 방향이며 새 런타임 오염 분류기를 만들지 않는다. 실제 serializer와 학습 input/prompt/tokenizer/template가 같아야 한다. 기존20 messages는 기본 상한이지20대화쌍이 아니며, Decision의 기존8192 token 초과 처리도 기록한다. 프롬프트를 늘리면서 실제 전달 history를 줄이지 않았는지 측정한다.

원인: 엄격한 계약과 자유로운 표현은 양립하며, 고정 표현 강제는 의미 정확성을 보장하지 않는다.
결과: 계약은 엄격히 하되 답변 길이·문장 자체를 획일화하지 않는다. 프롬프트 존재만으로 모델 준수를 주장하지 않는다.

### 5.1 추가 인계 — session·멀티턴 few-shot과 토큰 증액

**사용자 최신 요구: 세 Agent prompt에 session/state와 멀티턴이 연결되는 few-shot을 설계하고, 필요하면 토큰 예산 증액까지 계획한다.** 상세 정본은 최종 plan §5.6~5.7이다. 현재 Runtime prompt나 토큰 설정을 바꿨다는 뜻은 아니다.

1. 예시군은 같은 “응”의 후보선택/실행/무대상 대조, high→low 번복, 선호→중간대화→단계도착, allergy+수량+넓은추천/좁은추천, assistant오답 정정·clarify·VLM 근거이다. Agent마다 필요한 소수 예시를 실제 schema/serializer로 만들고 성공 멀티턴도 포함한다.
2. **few-shot 데모는 현재 session/history와 분리**한다. 예시의 알레르기·취향·주문이 실제 사용자 사실로 복사되지 않는지 시험한다. 이벤트는 가짜 user 메시지로 만들지 않는다. Response 예시도 검증된 후보/사실만 설명한다.
3. 예시 없는 기준과 후보를 같은 가중치·입력·생성 설정으로 비교한다. 예시 수·순서·메뉴/문체 편향을 검사한다. 선택용 dev와 최종 holdout을 분리하며 최종 시험 원문/정답/같은 family를 prompt에 넣지 않는다. 학습/추론 prompt·예시 버전 및 loss mask도 확인한다.
4. 입력 context, 출력 max_new_tokens, 학습 sequence 길이를 **각각** 측정·증액 후보로 관리한다. few-shot 때문에 기존 history가 더 잘리는 것은 실패이다. system/예시/history/state/현재발화별 토큰, p50/p95/p99/max, JSON/응답 잘림, 입력+출력 지원 한도, 학습 target 잘림을 검사한다.
5. 실제GPU에서 예시없는 기준 대비 prefill/생성/전체턴 지연·peak VRAM·OOM·학습 처리량을 측정한다. 지원 한도와 운영 예산 안에서만 증액하고, 초과 시 history 삭제가 아니라 예시 중복 축소·설계 선택을 보고한다. 새 학습·Docker·운영 변경의 기존 승인 경계는 유지한다.

산출물: `FEWSHOT_MANIFEST`, `TOKEN_BUDGET_REPORT`, prompt/serializer hash, 예시없는 기준 및 후보별 기능/비용 비교. 현재 소스의 Decision 입력8192·생성 상한 Decision1024 / Recommendation1024 / Response512는 앱 설정이며 모델 지원 한도 또는 세 Agent 공통 입력 한도를 뜻하지 않는다. 실제 설정은 착수 시 재확인한다.

원인: few-shot은 맥락별 판단을 보여 주지만 토큰과 예시 편향 비용도 발생한다.
결과: “예시 많이 넣기”가 아니라 기존 멀티턴 보존과 실제 품질 개선을 입증한 세트를 채택한다.

## 6. 데이터 정제·재구성 명세

### 6.1 실제 자료와 검수 범위

봉인 원본 `training/learning_first_v43/release/dataset/` 8,230행. 별도 미배포 후보 `training/learning_first_v43/candidates/scenario_context_v2/dataset/` 8,500행. 세 Agent별 records와 messages, provenance를 함께 대조한다. `2026-10-25_active_stack_regression_audit/SEMANTIC_COVERAGE.json`은 후보 전행 보고 회수, `OCT11_SEMANTIC_COVERAGE.json`은 과거 D3,750+P999 선언 범위의 근거이다. 선언된 COMPLETE를 최종 학습 승인으로 쓰지 않는다.

### 6.2 확정/재판정된 데이터 문제

| ID | 내용 | 처리 |
|---|---|---|
| D01 | Response RV12-S0581~0680 100행: 주문/targets 비었는데 기존 양파 실행 확인 target. r4/후보 동일 | 원가족 목적에 따라 입력 복원 또는 target 교정; 원본/정정 ledger. pending 객체 자체가 null이라는 설명은 틀림 |
| D02 | SCV2-D-0013/0044/0075/0106/0137: 선택 질문인데 execution pending, target 주입 시5/5 실행 | 질문·pending·target을 함께 교정. 정답 주입을 모델 오실행률로 쓰지 않음 |
| D03 | DV35-V-0088/0087 활성 제한 없는데 철회 pending; train restriction_removal pending0. 기본수량 low오표기6행 별도 지적 | 정상 철회/거절/취소 대조 보강, provenance 확인. 단순 target 문자열 교환 금지 |
| D04 | 조사·합성439행 대부분 입력 history; Response C232지적은 최종16 target+129 fixture+60정상+27보류/명료성 | 전부 target 오류로 삭제 금지. 16행은 scope/pending 정합화 후 답변 동기화 누락 계보 |
| D05 | Recommendation 후보723 중 물리1단계706·2단계12·빈5·3단계이상0; Response future_changes 전행빈값 | broad 조합→수락→단계도착, 미래변경 설명 보강. 미래 항목 사례 자체는122행 존재, 전혀 없다고 말하지 않음 |

초기 교정 수치 persisted none147, scope414, Response pending308, Decision pending12, 자연어210/Response target204, 추천중복99, 고정경로 Response620 제외는 **과거 파생 release 작업 이력**이다. 이후100행/16행 target 동기화 누락이 발견됐으므로 최종 무결점 증거가 아니다. 제거한620행도 모델 응답 경로 복원 후 필요한 가족인지 재평가해야 하며, 전부 자동 복구하거나 계속 제외하지 않는다.

### 6.3 단답을 자연스러운 대화로 바꾸는 데이터 기준

1. **사실부터 작성한다.** 독립 기대 상태·pending·허용 행동을 정한 후 자연어를 작성한다. 실제 실패 응답이나 현 Runtime 재생 결과를 그대로 gold로 복제하지 않는다.
2. **상황에 맞는 연결을 넣는다.** 선호/제약→이유 있는 후보, 변경→정확한 변경 설명과 필요한 질문, 이미 확정→중복 선택 없이 실행 확인. “저장했어요” 한 문장만 반복하는 분포를 피한다.
3. **길이는 목적에 맞춘다.** 짧은 확인·거절·종료는 짧아도 정상이다. 추천·설명은 필요하면 여러 자연스러운 문장으로 이어진다. 장문을 무조건 정답, 단문을 무조건 오류로 채점하지 않는다.
4. **의미와 언어를 따로 검수한다.** 조사/받침/병렬 연결·존댓말, 부정 범위·참조 대상·최신 정정·제약 철회·ASR 모호성을 구분한다. 깨진 assistant 합성문과 의도된 사용자 오수음 대조를 같은 정제로 지우지 않는다.
5. **실제 멀티턴으로 만든다.** 같은 “응/그거 빼고/둘 다 조금”이 서로 다른 state/pending에서 달라지는 사례, 성공 대조, 자기 이전 응답이 누적된 연속 평가를 넣는다. episode/family 단위 split으로 단순 의역 누수를 막는다.

원인: 한 줄 템플릿과 같은 상태 골격의 반복만 늘리면 새로운 대화·조합 일반화를 입증할 수 없다.
결과: KEEP/FIX/DROP/HOLD와 사유·원문ID·source hash·새 target hash를 전행 기록하고, 학습과 평가를 분리한다. HOLD/DROP은 새 활성 모든 split에서 제외한다.

편향 참고: 후보 Response train 상위 동일 답변320/1,771(18.07%), 그중 거짓 양파100행. 보강 추천 train12행 크림6/치즈6. 이는 편향 위험이며 메뉴 수를 억지로 균등화할 지시는 아니다. 제약상 치즈만 가능한 경우는 정상이다. Decision58.146%→57.371%는 **무history 비율**이지 정확도58%가 아니다.

### 6.4 추가 인계 — 편향 검수를 release 차단 조건으로 강화

최종 plan §6.6을 반드시 읽는다. 데이터·few-shot·평가기 각각에 **Agent별·split별·정제 전후·조건부 교차 분포**를 작성한다. 무history 비율이나 전체 평균만 맞춰 끝내지 않는다.

| 필수 묶음 | 검사 |
|---|---|
| 의미·권한 | 일반주문/즉시실행/수락대상/거절/clarify/skip/no-op/복합의도 쏠림, 항상실행·항상거절 모델 방지 |
| 메뉴·상태 | 메뉴·수량·현재/미래/완료·broad/narrow·history길이·정정/철회·복수제약·pending의 교차셀 |
| 언어·자연스러움 | 조사/받침·문체·부정·어순·ASR/모호성·참조대상·길이·반복문장·질문/권유빈도. 단답과 장문 어느 쪽도 강제 금지 |
| 출처·누수 | 원본/합성·generator/template/episode/family·완전/근접중복, few-shot과 최종시험 분리, 같은 상태의 의역 누수 |
| 실제 일반화 | 의미유지/의미반전 대조, 예시순서/메뉴변형, 미학습 유효조합·실발화 holdout, 분모 포함 유형별/최악군 성능 |

필수 셀의 최소 사례 수·허용 반복률·그룹 편차·평가 표본 수를 **생성 전에 근거와 함께 동결**한다. 기준 미정·누락 셀·사유 없는0행·누수·그룹별 기능 미달은 채택 보류이다. 제약상 치즈만 가능한 정상 상황은 억지 균등화하지 않으며 희귀 안전 사례도 빈도만으로 제거하지 않는다. 작은 표본은 미검증으로 표시한다. `BIAS_MATRIX`와 전행 ledger·독립 재판정을 함께 제출한다.

W2는 few-shot/토큰 설계, W3/W4는 편향 matrix·누수 없는 데이터/예시, W1은 judge 대조, W5는 동일조건GPU·그룹별 기능·토큰/지연/메모리 검증을 담당한다. 계획 G0~G4의 추가 조건이며 평균 점수가 좋아도 빈칸을 덮지 않는다.

## 7. 평가기 교정과 합격 기준

### 7.1 재현된 오채점 묶음

| 묶음 | 반드시 대조할 반례 |
|---|---|
| 초기4건 | 미승인 시작+뒤 질문 오통과; 유제품 allergy 크림 추천 오통과; 면만 완료인데 크림 완료 오통과; 부사 “보통” 수량 오인 |
| 데이터/계약 | 빈 targets의 양파 확인100행 통과; “좋아한다고 말씀하셨죠” 근거 없는 회상 미탐; skip plan 개수·sauce 값·low 한국어 표현 오채점 |
| 초기 실로그 | 문제 Response8개 중5개 통과. 위험 옵션/질문 미응답/질문과 pending 불일치 등; FAIL3개도 모든 오류 포착은 아님 |
| 마지막 실로그 | 오일 반영 부정, 오래된 토마토 정당화, 미적용 변경 약속, 비난에 칭찬 감사문 PASS; pending없는 실행 확인은 FAIL로 잡음 |
| 정상 대조 | “게살은 제외”는 위험 추천 아님; “오일 반영”에 ‘소스’ 글자가 없다고 탈락시키지 않음; “준비했어요. 담을까요?”를 무조건 물리 완료 주장으로 세지 않음 |

구조/상태/권한 평가와 대화 의미·자연스러움 평가를 분리한다. 문자열 규칙 무한 추가 또는 LLM judge 단독 합격은 부족하다. 원자 주장과 근거를 대조하고 정상/오답 최소 대조쌍으로 검사기 자체를 시험한다. judge 의견 불일치는 원문에서 조정한다. 평가기의 자연어 판정을 Runtime 차단 규칙으로 복사하지 않는다.

G0 입력/전행ledger, G1 oracle/평가기, G2 안전/상태, G3 기능/대화/성공회귀, G4 실제GPU/운영 재현은 최종 계획을 따른다. loss 하락·gold 자기채점·160개 기존 테스트 통과만으로 최종 채택하지 않는다. 모든 불명확 입력을 거절하여 무사고를 만드는 후보도 기능 gate 실패이다.

### 7.2 철회·제한된 검수 주장 — 다시 같은 오수정 금지

| 원 지적 | 최종 판정 |
|---|---|
| 옛 Decision33행에 none 누락 | preselected 거절 경로가 삭제함. 삭제33/33, 옛 입력 재선택32/33, r4 교정 입력 재선택33/33. none 일괄 추가 금지 |
| Response C232행 모두 잘못된 target | 16 target +129 fixture +60정상 +27명료성/oracle로 분리 |
| 일반 “담아줘”에 commit 추가 | 최초 일반주문 계약과 충돌하여 기각. 명시 실행/직전 질문 맥락을 봐야 함 |
| 현재 scope가 미래 추천을 모두 막음 / 단답은 모두Python | scope는 이미 열림. 모델 raw 좁은제안·단답과 Python reason/문장 치환을 분리 |
| VLM FAIL이면 실물 미투입 확정·반드시 정지 / 과거가 전체적으로 더좋음 | 영상/승인/동일조건 비교 없이 확정 금지. 기존 VLM 진행 정책과 성공 멀티턴 보존 |

Claude의 “보강270행 안전 결함 없음” 포괄 판정은 선택질문↔execution5행 반례로 제한됐다. 오래된 매운맛 극성 반전은 Decision raw에서 시작했으며 internal `preference_options`를 external `preferences`와 혼동한 저장 오류 주장은 기각됐다. **여러 에이전트가 동의해도 원본 재현이 우선**이다.

## 8. 필수 회귀 시나리오

최종 계획 F01~F20 및 실제 ZIP S01~S50 매핑을 사용한다. 최소 단위는 전체 주문 시나리오뿐 아니라 다음 대조 episode이다.

1. 맵지 않게/매운맛 선호→넓은 추천→선택→토핑 단계 권유, 철회/반대선호/유제품 제한 변형.
2. 야채high/low→둘 다low→확인/즉시 실행, 최신 수량을 말과 plan 모두 유지.
3. 게살 allergy+육류많이 추천→후보 선택/거절/변경→실행, 무해한 주문 유지와 위험 후보 제외를 함께 확인.
4. clarification+old pending→모호한응/명확한정정의 대조, 선호-only/유효후보/확정주문 사용자턴과 이벤트 각각 확인.
5. 이미 담긴 재료 취소, 강한 제한 철회, no-op 제거, skip/빈commit, 부분VLM FAIL, 과거assistant오답 정정, 반복/오수음/일반대화 대상 참조.

동일 사용자 문장만 바꾼 같은 state를 train/test로 나눠 성공을 부풀리지 않는다. 실로그 원문 holdout은 별도 유지하며 학습에 사용할 변형도 family 기준 누수를 검사한다.

## 9. 저장소·학습 디렉토리 정리

실행 계획 정본은 `docs/IRC LLM Final plan.md`, 계정 인계 정본은 `docs/IRC LLM Final handoff.md`이다. 외부 작업장 `.../learning_first_v43_llm_final/final/`에 진입점·사본·검수 패키지를 둔다. 614파일 이전 봉인 묶음 `.../reports/2026-10-25_llm_final_followup_bundle/`는 수정하지 않는다. `TRAINING_HISTORY/INDEX.md`는 대부분 원본으로 가는 바로가기이며 실제 이관 완료가 아니다.

이번 계정 인계 묶음은 문서와 감사 산출물 사본이다. 코드/모델/checkpoint/tar/원본데이터/실로그 전체 이관을 뜻하지 않는다. 같은 컴퓨터 계정 교체는 로컬 경로를 사용할 수 있지만, 다른 컴퓨터라면 별도 manifest에 적힌 원본 위치/가중치가 추가로 필요하다. 감사 스크립트 일부는 `Path(__file__)` 기반이므로 사본 위치에서 실행하지 말고 문서의 원래 저장소 경로에서 실행한다.

각 학습회차 README는 학습 날짜→설정/identity→실제 결과→채택/기각→실패ID/rollback 순으로 남긴다. mtime을 학습 날짜로 쓰지 않는다. 오래된 보고서를 정리하면서 삭제하거나 `.gitignore`로 가리는 것을 실제 이관 완료라고 부르지 않는다.

## 10. Agent별 비교·재학습·Docker

운영 설정 라벨 D v3.4 epoch4 / Q v2 epoch4 / P v1.3 epoch3, Oct11 마지막 D v3.2 epoch4 / Q Base / P v1 epoch1을 구분한다. 실제 로드 model/adapter/base/tokenizer 경로·hash·revision을 새 비교 전에 고정한다. 현재 config만으로 당시 로그 가중치를 증명하지 않는다. Recommendation 과거 strict43/43도 현재 broad/자연스러움 gate 통과 증거는 아니다.

GL11은 기존 checkpoint → prompt-only → 최소Runtime의 학습 전 비교이며 실패도 기록하여 GL12 Agent별 학습 필요 판단에 전달한다. GL13에서만 별도 승인 후 필요한 Agent의 정제data adapter 학습·새 가중치 재평가를 수행한다. 새 adapter 최종 채택은 해당 모델의 독립 holdout·P1/유동 변형/route 3종·기존 성공 회귀·안전/지연 기준을 통과한 뒤 판단한다. 기존 adapter 성적은 비교용이며 학습 전 새 모델 성능은 NOT_RUN이다. 각 Agent는 같은 Base에서 독립 LoRA이며 서로의 adapter를 연속 덧씌우지 않는다. 원격 r4 결과는 도착하면 별도 후보로 평가하며 자동 교체하지 않는다. GPU 지연은 턴/총시간/로딩을 분리한다.

Docker 과거 문제: tokenizer_config hash mismatch, main과 고정revision 캐시 불일치, `/tmp/dataset-quality-audit.json` 권한 충돌. 기록상 r4는 checkpoint monitor 경로·UID1000 재작성·mounted-model preflight로 보완했다. base revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` 기록을 실제 입력 hash와 재대조한다. 검사를 제거해 우회하지 않는다. `Downloads/TRAINING_IMAGE_SPEC.ko_v2_9 (1).md` 및 프로토콜 ZIP은 실파일 확인 후 사용한다.

**새 학습·체크포인트 교체·ROS 실물 실행·원본 branch 변경·원격 취소/재시작은 별도 승인 대상이다.** 이 인계 문서 작성은 그 작업을 실행한 것이 아니다.

## 11. 작업 패키지 — 새 Codex의 첫 작업

| 순서 / 계획ID | 해야 할 일 | 종료 증거 |
|---|---|---|
| 1 / W0, 30–60분 | Git/최신 로그/active 모델/데이터/보고서 동결 상태 재확인. 기존 검수 재사용, 바뀐 부분만 추가 | identity/hash·새로그 목록·완료/미완료 표 |
| 2 / W1·W2, 각2–6시간 추정 | 평가기 oracle 교정과 Runtime 계약 설계를 병렬. 수량·B1·B2·고정발화 inventory를 먼저 | 정상/오답 반례, 호출표, history무변경·안전보존 diff. 공유파일 단일작성자 |
| 3 / W3·W4, 7–14시간 이상 추정 | 기존 판정 재심→전행KEEP/FIX/DROP/HOLD→자연스러운 세Agent 후보 재구성 | 100/5/2/16 등확정문제 처리, 조사·의미·편향·누수 ledger, G0/G1 |
| 4 / W5·W6, 1–3시간+GPU실측 | 같은동결episode로 기존/prompt/runtime 실제GPU 연속 대화 비교 | state·raw·최종발행문·plan·latency·실패층, Agent별 재학습 필요 결정 |
| 5 / W7–W9, 별도승인 | 필요한학습·재평가·실제경로이관·Docker검증·채택 | G0–G4·회차README·digest·복구경로. 승인/증거 없으면 진행완료 선언 금지 |

시간은 작업량 추정이지 이미 경과한 시간이나 원격 학습 ETA가 아니다. 사용자에게 먼저 W0 결과와 W1/W2 수정 경계를 짧게 보고한다. 단순 진단을 반복하며 전체 감사를 처음부터 재실행하지 않는다. 추가 worker가 필요하면 코드/평가기, 데이터, 실로그/회귀를 분담하고 조정자는 최종 판정을 소유한다. 이전 worker 원문 지적을 그대로 데이터 수정명령으로 전달하지 않는다.

### 11.1 새 계정에 붙여넣을 시작 프롬프트

**Goal 모드 시작/재개는 [IRC LLM Final goals](</home/roma/IRC_SOOMAC3.0/docs/IRC LLM Final goals.md>) §4 프롬프트를 우선 사용한다.** 최신 지시는 Goal 3개씩 순차 진행이다. BATCH01(GL01→GL02→GL03)부터 시작하고 묶음 내 선행 Goal 완료 후 다음으로 이어가되, 다음 묶음은 자동 시작하지 않는다. 사용량 중단·작업 전후 STATE/WORKLOG/NEXT 저장·별도 승인 경계는 유지한다. 아래는 전체 맥락용 참고 프롬프트이며 전체 작업의 무제한 Goal 실행 지시가 아니다.

```text
/home/roma/IRC_SOOMAC3.0/docs/IRC LLM Final handoff.md를 처음부터 읽고,
§1의 순서대로 최종 plan과 오늘 Codex·Claude·Astra 검수 및 조정자 재판정을 읽어라.
브랜치는 llm_final. 목표는 이전 대화+state에 근거한 자연스러운 직원형 대화이다.
잘 된 멀티턴은 보존하고 history 삭제/요약/축소와 새 Python 자연어 규칙을 만들지 않는다.
데이터는 단답으로 획일화하지 않으며, 잘못된 target·중복은 활성본에서 제외한다.
session/멀티턴 few-shot, history를 보존하는 토큰 예산 증액 검증, 엄격한 조건부 편향 감사를 포함하라.
계획·검수·실제수정·GPU검증·학습완료를 구분해 W0의 현재상태와 W1/W2의 첫 수정범위를 보고하라.
이 인계만으로 학습·원격취소·adapter교체·ROS실행을 시작하지 않는다.
필요한 Codex 작업자는 gpt-6-astra/high를 실제 launch 설정으로 확인한다.
```

### 11.2 기존 회귀 재실행

```bash
cd /home/roma/IRC_SOOMAC3.0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/soomac_irc /home/roma/miniconda3/envs/gemma4_env/bin/python -m pytest -p no:cacheprovider src/soomac_irc/test/test_natural_multiturn_contract.py src/soomac_irc/test/test_section_question_events.py src/soomac_irc/test/test_evaluation_v43.py src/soomac_irc/test/test_scenario_context_evaluation.py -q
```

직전 실측160 passed·2 subtests·2 warnings, exit0. 위 명령은 shell이며 환경변수는 그 명령에서 정의한다. 새 목표 테스트를 추가하기 전 같은 결과는 기존 코드 보존 확인일 뿐 의미 합격이 아니다. 최신 기록 재생 도구는 `training/learning_first_v43/reports/2026-10-26_latest_sessions/verify_logs.py`이며 GPU/ROS 실행이 아니다.

## 12. 최종 완료 정의

최종 plan의 G0–G4를 모두 증거와 함께 충족하고, 원문 시나리오 외 표현·메뉴·선호·제약·history 조합에서도 정해진 동결평가를 통과해야 한다. 기존 성공 멀티턴·즉시실행·철회·skip을 망가뜨리지 않는다. 마지막으로 말·주문·pending·실행·VLM 증거가 일치해야 한다. “한 번 자연스럽게 대답함”이나 “학습 loss가 낮음”은 완료가 아니다.

## 13. 이번 인계 작성의 진행 기록

2026-10-26 GL04 추가 요구: 복합 멀티턴과 모델 판단 자유도를 필수검수로 확정했다. plan §8.3과 final/goals/GL04/MULTITURN_REVIEW.md를 읽는다. 같은 최신 발화라도 앞선history/state/pending에 따라 달라지는 지시대상·수량·수락·실행·제약을 검증한다. mock 통과와 실제LLM closed-loop 성공은 구분하며, 새 Python 자연어 규칙/역사삭제는 금지한다. 전행 데이터 정제는 GL07~09이며 현재 수행했다고 쓰지 않는다.

2026-10-26 최신 인계: GL01·GL02 완료, GL03 Codex 보완 완료(384 passed·최종Codex찬성). 현재 상태는 외부 작업장 `final/goals/GOAL_INDEX.md`와 `GL03/CODEX_FOLLOWUP_STATE/WORKLOG/NEXT.md`를 읽는다. 예전 `USAGE_LIMIT_HANDOFF.md`의 GL02 미완료/GL03 미시작은 과거 시점 기록이다. 이후 작업/독립 검수는 Codex only. 사용자 정정: “나머지만 담아줘”는 이전 대화·state·직전 질문·pending으로 판단하고, 수량 생략 시 기존 확정량을 유지하며 미확정 새 항목에만 normal을 쓴다. 수량 기본값과 실행 승인은 별개이다. [코드 수정 명세](<IRC LLM Final code changes.md>) R07과 plan §3.2/8.0이 후속 구현 기준이다.

2026-10-26: 초기 release 감사, 후보/평가기 개선, 3담당+Claude/Codex 교차검수, future-scope, active-stack 재판정, Oct11 GPU, 계약reset 조정,225109,최종계획B1/B2,최신34턴 결과를 통합했다. 원문 보고의 시점·분모·과잉판정 정정을 명시했다. 이 인계 작업은 문서만 작성/갱신하며 Runtime·학습자료·model·ROS를 변경하지 않는다. 새 인계 문서 자체의 교차검수와 파일 묶음 검증 결과는 별도 `HANDOFF_VALIDATION.md`에 기록한다.

2026-10-26 추가: 사용자 요청에 따라 최종 plan §5.6/5.7/6.6과 이 인계 §5.1/6.4에 session·멀티턴 few-shot, history를 희생하지 않는 입력/출력/학습 토큰 증액 검증, 조건부 교차 분포·누수·그룹별 편향 기준을 추가했다. W1~W5 담당과 G0~G4 산출물에 연결했다. 이전 인계 봉인 묶음은 수정하지 않고 새 문서 보강본으로 보관한다. 이 추가 작업도 문서 변경이며 실제 prompt/설정/data 변경은 아니다.

## 용어 정리

| 용어 | 의미 |
|---|---|
| 정본 상태(canonical state) | 현재 검증된 주문·선호·제약·pending이며 과거 assistant의 말 자체가 아니다 |
| 검수 범위(coverage) | 어떤 행/파일을 읽거나 검사했는지. 정확성이나 채택 여부와 다르다 |
| 정답 주입(gold injection) / 기록 재생(replay) | 정답 또는 저장모델출력을 코드에 넣는 시험. 새 모델 추론이 아니다 |
| 정책상 진행(policy_success) | 실패 재시도 뒤 제어 진행. 관측PASS나 실물성공이 아니다 |
| 채택 준비(dataset_ready/release gate) | 판정 회수 이후 오류 교정·독립 oracle·회귀·운영 증거까지 갖춘 상태 |

(내가 의역해보자면, 대화를 기억하는 기능을 버리는 것이 아니라, 기억한 것과 지금 하는 말과 실제 행동을 서로 맞추는 작업이다.)
