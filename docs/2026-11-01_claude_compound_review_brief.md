# Claude 단발 독립 검수 — 추천 의미와 복합 즉시 실행

## Target / 범위

사용자는 Claude와 Codex가 함께 의미를 제대로 검증하고 「바로 해줘/바로 넣어줘」를 선택·추천수락·수량변경·제약 추가와 한 번에 처리하도록 데이터 보강을 승인했다. 담당 범위는 LLM뿐이다. 실제 기본 시연은 면 얇은면normal+느끼한소스추천 → 크림선택+즉시실행 → 양파/버섯low → 버섯만normal+즉시실행 → 게살allergy+소시지high+즉시실행 → 페퍼론치노제외/치즈low+즉시실행이다.

현재 검수 기반은 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/candidates/conversation_supplement_20261101_r4_a2/dataset`이다. r3가 최신이 아니다. 전체8230행/train6566행이며 현 모델은 이 자료를 학습하지 않았다. 직전 교정은 A06 느끼함비선호→토마토(사용자 지정 메뉴 기준), F08 조사 교정과 실제 다음턴history를 포함한14행이다. 나머지8216행은 동일record/system 영수증을 재결속했다. CPU gold 재생은 실제 모델 성공이 아니다.

참고: `docs/2026-11-01_dataset_rebuild_worklog.md`의 최신 절만 먼저 읽는다. `training/learning_first_v43/conversation_supplement_20261101_r1/`의 replay_dialogues.py, correct_meaning_r4.py, package_meaning_r4.py, test_meaning_r4.py와 `src/soomac_irc/soomac_irc/`의 agent_contract.py, agent_prompts.py, response_contract.py, llm_langgraph.py를 필요한 분기만 읽는다. 소스·schema의 실제 구조를 확인한 뒤 판정한다.

## Change / 할 일

독립적인 반례 검수 보고서를 **오직** `/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/compound_semantics_20261101/CLAUDE_REVIEW.md`에 작성한다. root가 별도 literal 보강과 실제 Graph 연결을 작성 중이다. 생성 후 이 세션으로 정확한 파일 경로를 추가 전달할 예정이다. 이 최초 검수에서는 아래 기준의 빈틈과 현재 기반 대표 사례를 대조하고, 초안 보고서를 먼저 작성한다. 다음 입력 전에는 추가 에이전트를 만들거나 무한 대기하지 말고 완료 응답으로 돌아온다.

검수 기준:

1. 추천 의미: 넓은 식사 요청은 현재부터 미래 코스 전체를 고려한다. 비매운 요청에 면만 바꿔 해결했다고 하지 않는다. 선택 가능한 안전 소스를 후보와 응답에 함께 연결한다. 명시적으로 면만/채소만 요청하면 그 범위를 지킨다. 기존 확정량·선호·강한제한 보존. 느끼함비선호는 A06에서토마토, 느끼한소스원함은크림이다. 오일은 보편적으로가볍다고 단정하지 않는다.
2. 복합 실행: 추천수락+바로해줘/넣어줘, 수량번복+바로해줘/넣어줘, allergy+새주문+실행, 제외+새수량+실행을 같은 D target에 넣는다. 명시적 긍정 실행 의도가 분명하면 raw commit=true이다. Python이 keyword로강제하면 안 된다. 실행이 허가되는지는 실제 현재주문·물리단계·안전조건을 별도 확인한다.
3. 맥락 대조: 같은 「바로넣어줘」도 기존 execution pending, recommendation pending, pending없는 명확한 주문, 대상없는 모호한 대화에서 구분한다. 부정/인용/조건/과거회상은 자동실행하지 않는다. 「현재주문변경+실행보류」는 최신order를보존하고execute=false이다. pending이 남을 수 있으므로 pending=null만정상이라고 가정하지 않는다.
4. P grounding: 이번선택/변경/기존량유지/실제완료를 구분한다. 「이미그양」을 applied변경이있는데말하면틀림. execution pending과next_prompt가실제로일치해야실행질문가능, 질문/취향저장만으로고정질문반복금지. 실행시작안내는 현재deterministic 경로이므로 거기에가짜P행을만들지 않는다. Q가없는데P가추천을창작하는오류주입분기는정상학습D정답으로내보내지 않는다.
5. 실제 before→D→Q→policy→after→P→다음history를 확인한다. source ID/파일경로/줄번호/짧은반례로 KEEP/FIX/HOLD를 구별한다. 정규식은후보탐색용이지의미합격/일괄삭제기준이아니다. 실제 메뉴/물리scope 예외를 확인한다.

## Constraints / Ownership

읽기 전용 검수+지정보고서1개만작성. 원본/학습데이터/Runtime/prompt/cache/가속/env/VLM/ROS/adapter/원격/commit/push/기존Claude속도실험을 수정·실행·중단하지 않는다. GPU 실행·학습 금지. 추가 Claude/Codex CLI·하위 worker·브라우저/API 호출 금지. 전역 「항상Codex교차검수」는 이미root가별도소유하므로중복실행하지않는다. 세션에오류/인증제한이발생하면명시하고다른모델/계정으로재시도하지않는다.

Orca 감독Run 생성은 root의자기terminal식별자없음으로거부됐다. **이작업은 Orchestration Task/Dispatch가없는단발터미널검수**다. worker_done/heartbeat/임의run-create를보내거나다른패널handle을사칭하지않는다. 마지막에 실제 작성한 경로·판정·미확인범위를 답하고 종료한다.

## Observable acceptance / 완료 증거

첫 응답은 5분 이내에 기준 검토와 기존 대표 행 판정을 보고서로 남긴다. 전체전수의미검수/모델성공을주장하지않는다. 전체Q706행의범위분포/초기소스누락후보는 `/tmp/soomac_recommendation_meaning_r4/`에있지만, 초기한후보QF28-narrow는채소한가지만요청한정상사례다. Q train617중initialnoodle27, 현재단계만출력진단525는대부분명시적좁은요청이라전체폐기대상이아니다. 기분/스트레스/우울의현재발화표면변형0은의미동의어0이나모델실패원인증명이아니다. 기존감사를다시전수시작하지않는다.
