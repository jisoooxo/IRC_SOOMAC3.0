# IRC LLM Final — 진행 목록

**현재 우선: GL01~06 명시 범위 완료, BATCH02 종료.** GL05 코드 계약과 GL06 prompt/few-shot·serializer·token·알려진 로컬 누수 검증, 최종Codex sol/high찬성. 전체612pass·8subtests, 원본prep24개보존/새32후보검수·system23개, Decision oracle replay17개. S04/S09의근거없는진행/없는실행질문을교정하고추가12회귀와독립검수를반영했다. 실제새모델성능NOT_RUN·전체data정제/새학습/Docker미실행. 다음은승인후GL07이며자동시작하지않는다. 정본: final/goals/GL06/RESULT·manifest·RECURRING_FAILURE_CONTRACT·TOKEN_BUDGET_REPORT. 아래는과거진행이력이다.

**최신 우선(2026-10-26): 판정 기준 문서 정정 완료, GL05 자체는 미완료다.** 프롬프트 개선은 계속하되 기존 adapter 성적으로 앞으로 학습할 모델을 판정하지 않는다. GL05 코드/안전 계약 → GL06 oracle 기반 prompt/few-shot → GL07~10 데이터 → GL11 기존 모델 비교 → GL12 학습 판단 → GL13 승인 후 새 모델 재평가를 구분한다. 기존 6-demo A/B는 기본/후보 각20/22·미채택이며 새 adapter 성능은 NOT_RUN이다. Codex 문서 수정안 검수는 조건부 찬성으로 학습 비교/채택 시점 분리를 요구했고 plan/goals/handoff에 반영했다. 아래 수치/실행 중 표기는 각 시점 이력이며 최신 기준은 GL05/ACCEPTANCE_POLICY.md·STATE·NEXT이다. 이번 작업은 문서만 변경했다.

**최신 GL05: 구현/검수 보완 진행 중, 미완료.** 네 대화 경로의 Response 소유권과 이유 보존·user/event 입력·단계 이벤트 연결을 구현했다. LLM 회귀520건·8subtests와 기존어댑터GPU12사용자턴/2모의robot event 구조 증거를 저장했다. 전체 source 검사는 병렬 VLM prompt 변경으로false/exit2이며 원본을 유지하고, 별도 검사에서 해당 GPU snapshot의 LLM 필수13파일/나머지28파일 불변을 확인했다. 이후 publisher 실패 수정으로 현재 node는 그 GPU snapshot 이후 버전이다. 질문 의미와 전달 경계에 대한 Sol 검수 지적은 아직 최종 해소 전이다. GL06은 후보24개·CPU측정/harness 등 준비만 완료했다. **담당 범위는 LLM 전용이며 VLM은 수정 금지(AGENTS.md).** 아래 GL05 미착수 표기는 과거 이력이다.

최신 GL04: **상태·실행 권한 범위 완료, Codex astra/high 최종 찬성.** 수량 문법·강한 제한·현재 실행과 미래 추천·명확화 권한을 보강했다. 사용 가능한 테스트493개·8 subtests 통과, 20개 controlled episode/63턴 전체 필드 대조 저장. 실제 기존 어댑터 GPU12턴 상태검사12통과. **모델 의미 성능·자연스러움 완료는 아니다.** 상세는 final/goals/GL04/RESULT.md.

[Goal 계획](<IRC LLM Final goals.md>) · [최종 plan](<IRC LLM Final plan.md>) · [코드 수정 명세](<IRC LLM Final code changes.md>)
→ **GL01~04 완료. GL05~15 미착수. GL04 완료는 상태 계약 범위이다.** 2026-10-26 KST.

## 1. 완료한 목록

| 작업 | 실제 완료 범위 | 증거 |
|---|---|---|
| GL01 현황 동결 | llm_final/HEAD·입력698파일·설정된 adapter3개·데이터/로그 identity 기록. 실행 중 모델과 원격 학습 상태는 UNKNOWN | final/goals/GL01/INPUT_MANIFEST.json·RESULT.md |
| GL02 정답 기준 | ORACLE r1 동결, 독립 검수9건 반영. Runtime 출력을 무조건 gold로 삼지 않음 | GL02/ORACLE.md·ORACLE_REVIEW.md |
| GL03 평가기 | 실제 재료 값의 제한 검사, skip/빈commit 구분, 구조 오류 분리. Response 문구 규칙은 검토 표시로 분리 | GL03/RESULT.md·evaluation_v43.py |
| GL03 Codex 보완 | 승인/실행 및 정답의 bool 타입, null 루트·깨진 pattern·무효 pending 대상 검증. 전체384 passed·2subtests·exit0, 최종Codex찬성 | GL03/CODEX_FOLLOWUP_RESULT.md·MANIFEST.json·REVIEW2.md |
| 정답 비교·계약 인계 | release8230+candidate8500의 판정 변화를 재비교. 같은 Response100개ID 외 추가 변화0, 데이터30파일 원본hash 유지. 문맥 기반 나머지·기존 수량 보존·Codex only로 plan/handoff 정정 | GL03/codex_followup_gold_comparison.json, docs plan §3.2/8.0 D2 |

위 증거의 `final/goals/` 정본은 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/final/goals/`이다. 저장소 staging은 `training/learning_first_v43/final/goals/`이다.

## 2. 완료로 해석하면 안 되는 것

384개는 코드·계약 테스트 통과이며 실제 LLM의 자유로운 대화 성능이나 실제 로봇 성공률이 아니다. 16,730행은 중복되는 release와 candidate의 합계이며 의미 전수검수 완료가 아니다. 알려진100개ID 외 판정 변화가 없다는 것을 나머지 정답의 의미가 모두 맞다는 증명으로 쓰지 않는다.

Runtime·최소 prompt는 GL04 후보를 반영했다. 학습 데이터·adapter는 변경하지 않았다. 기존 학습 자료의 FIX/DROP 집행, Agent 재학습·모델 교체·Docker·실물 ROS·원격 취소·commit/push는 하지 않았다. 이전 감사 자료는 재사용하며 새 의미 전수감사를 수행했다고 주장하지 않는다.

## 3. 확정한 사용자 계약

“나머지만 담아줘”는 이전 대화·현재 주문·유효 후보·직전 질문·pending을 보고 해석한다. 특정 문구를 소시지 normal 즉시 실행으로 치환하지 않는다.

수량을 새로 말하지 않으면 기존 확정량을 유지한다. 수락한 유효 후보의 수량도 유지하며, 새 명시 수량이 우선한다. 이들 근거가 없는 새 선택만 normal을 기본값으로 쓴다. 수량 기본값은 실행 승인이 아니다.

history와 성공 멀티턴·안전 검증·ROS topic·busy STT 차단은 보존한다. Python 자연어 키워드 판단과 새 고정 발화는 추가하지 않는다. 이후 구현·독립 교차검수는 Codex만 사용한다.

## 4. 현재 GL04

GL04는 실제 상태 처리 코드에서 수량·강한 제한·복합 의도·pending이 위 계약대로 움직이도록 고치는 단계이다. GL05의 자연스러운 발화 경로와 GL06의 전체 prompt/few-shot 확정은 뒤에 이어진다. GL04 때문에 필요한 최소 schema/serializer/prompt 정합성 교정은 같이 검증하되 GL06 전체 완료로 표시하지 않는다.

실행 상세는 `final/goals/GL04/BRIEFING.md`, `STATE.md`, `WORKLOG.md`, `NEXT.md`를 읽는다. 구현 순서는 공통 상태 계약 → 실패 재현 테스트 → 최소 코드 교정 → 통합 회귀이다. 계약 동결 뒤 schema/domain·독립 테스트·읽기 전용 검수는 파일 소유권을 나누어 병렬 준비할 수 있다. 같은 llm_langgraph.py의 여러 상태 분기를 동시에 수정하지 않는다.

## 5. 다음 한 단계

사용자 추가: GL04는 복합 멀티턴·모델 자유도 검수를 포함한다. history 전달·mock 상태처리·실제LLM closed-loop 결과를 구분하고, 실제 모델 검증 미수행을 성공으로 보고하지 않는다. 상세는 GL04/MULTITURN_REVIEW.md와 plan §8.3. Agent별 전행 의미 정제는 GL07~09에서 수행한다.

초기 실패 재현4건/7건/복합요청4건은 각각 로그를 보존하고 후보 수정 뒤 회귀했다. 16,730행 기존 target의 새 schema 위반0, GL03의 동일한 Response100 ID 외 판정 변화0·원본30파일 hash 동일이다. 이는 의미 전수검수 완료가 아니다. 전체 테스트 수집은 ROS의 rclpy가 없는 STT2파일에서 막혔고, 그 두 파일을 제외한493개가 통과했다.

실제 GPU run2/run3은12턴 중 상태검사11턴 통과: 직전 명확화를 받은 `응`을 모델이 옛 실행 수락으로 오판했다. 예시-only 교정은 실패하여 채택하지 않았다. pending 질문 목적과 1회 repair를 반영한 run4는 오실행0·명시 low/low 실행 성공이나 진단기의 새 메타데이터 기대값1건이 오래되어 재실행했다. Response의 명확화/실행 질문 충돌도 교정한 **run5는12턴 상태검사12통과·source hash불변**이다. `응`에서는 명확화 유지, 후속 low/low 명시실행은 성공했다. 실제 ROS 발행0, 실제 어댑터 교체0이다. 전체 의미·자연스러움 합격은 아니며 최종 독립 검수 판정은 GL04/STATE와 RESULT를 따른다.

다음 한 단계: 사용자 GL05 지시 후 고정 발화·추천 이유·이벤트 Response를 교정한다. GL06은 전체 few-shot·토큰, GL07~09는 Agent별 의미/언어/편향 데이터 전행 정제이다. 한 줄 저장 보고·추천 실패가 남아 있으므로 자연스러운 전체 시나리오 완성이라고 보고하지 않는다. 자동 학습·모델 교체·실물 ROS·commit/push는 하지 않았다.
