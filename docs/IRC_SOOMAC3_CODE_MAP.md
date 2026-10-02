# IRC_SOOMAC3.0 코드 지도

> 기준 브랜치: `llm_langgraph`  
> 기준 커밋: `5215b5102336a838916ecb0c71477fa0d377a345` (`판단 필드 추가 전 백업`)  
> 확인일: 2026-10-02  
> 목적: 다음 작업에서 저장소 전체를 다시 읽지 않고, 변경 대상과 상태 소유자를 빠르게 찾기 위한 현재 코드 스냅샷이다.

**이 문서는 설계안이 아니라 현재 코드가 실제로 하는 일을 기록한다.** 기준 커밋이 바뀌면 먼저 이 문서와 코드가 같은지 확인한다.

## 0. 먼저 읽을 다섯 파일

1. `src/soomac_irc/soomac_irc/domain.py` — 메뉴와 공정 단계의 정본이다.
2. `src/soomac_irc/soomac_irc/agent_contract.py` — Decision·Recommendation·Response JSON 계약이다.
3. `src/soomac_irc/soomac_irc/llm_langgraph.py` — 세션 상태와 한 사용자 턴의 처리 순서이다.
4. `src/soomac_irc/soomac_irc/llm_policy.py` — 모델 출력에 대한 Python 안전·물리 정책이다.
5. `src/soomac_irc/soomac_irc/llm_langgraph_node.py` — ROS와 LangGraph를 연결하고 실제 작업을 시작한다.

(내가 의역해보자면, `agent_contract.py`가 모델의 답안지 양식이고, `llm_langgraph.py`가 처리 순서이며, `llm_policy.py`가 실행 허가 담당이다.)

## 1. 전체 실행 구조

```text
브라우저 UI
  └─ /ui/start, /ui/reset
          ↓
STT ── /stt_question ──→ LLMLangGraphNode
                              │
                              ├─ Decision Agent
                              ├─ LangGraph workers
                              ├─ Python policy
                              ├─ Recommendation Agent (필요한 턴만)
                              └─ Response Agent
                                      │
                 /llm_response ←──────┤
                 /llm/plan     ←──────┘
                                      ↓
                             MAIN 공정 조정 노드
                                      ↓
                      Rail → Vision → Arm control
                                      ↓
                                 VLM 확인
                                      ↓
                              /llm/next 또는 retry
```

LLM 계층과 로봇 계층 사이의 실제 실행 계약은 `/llm/plan`이다.

```json
{"class":"mushroom","repeat_count":2}
```

- `class`: 로봇이 이해하는 영문 작업명이다.
- `repeat_count`: `low=1`, `normal=2`, `high=3`으로 변환된 투입 횟수이다.

## 2. 상태 소유자

| 상태 | 정본 소유자 | 설명 |
| --- | --- | --- |
| 메뉴·공정 정의 | `domain.py` | 소스, 면, 토핑, 제한 카테고리, `SECTION_ORDER` |
| 실제 누적 주문 | `SessionState.order` | 여러 사용자 턴에 걸쳐 유지되는 canonical order |
| 대화·추천·확인 | `SessionState` | `history`, `action_history`, `recommendation`, `pending_confirmation` |
| 한 턴의 모델 해석 | `Decision` | 현재 발화에만 유효한 변경·질문·실행 의도 |
| 로봇 진행 상태 | `LLMLangGraphNode` | `section`, `task_queue`, `active_task`, `completed_tasks` |

**`SessionState.order`만 실제 주문 사실이다.** 모델 출력은 Python 검증과 정책 적용 전까지 후보이다.

## 3. 한 사용자 발화가 처리되는 순서

현재 `llm_langgraph.py::build_graph()`는 다음 순서로 항상 직렬 실행한다.

```text
interpret_decision
→ validate_and_repair
→ resolve_confirmation
→ apply_workers
→ run_recommendation
→ check_policy
→ generate_response
→ END
```

1. `interpret_decision`은 Decision 모델 또는 deterministic override를 호출한다.
2. `validate_and_repair`는 잘못된 field/query를 발견하면 모델을 한 번만 다시 호출한다.
3. `apply_workers`는 원본이 아닌 복사본에 주문·제한·취향 변경을 적용한다.
4. `check_policy`가 통과·경고·재질문·사람 확인·차단을 결정한다.
5. `generate_response`가 확정된 사실만 Response 모델에 전달하고 history를 누적한다.

Cause: 모델 출력과 실제 상태 변경이 같은 단계에 있지 않다.  
Effect: 모델이 잘못된 값을 말해도 Python 정책에서 실행을 막거나 되돌릴 수 있다.

## 4. LLM·대화 계층 파일

### 4.1 계약과 의미

| 파일 | 현재 역할 | Decision Routing 작업 시 변경점 |
| --- | --- | --- |
| `domain.py` | 지원 메뉴, 양, 제한, 공정 순서 | alias·canonical mapping을 둘 경우 정본 후보 |
| `agent_contract.py` | 세 JSON Schema와 sparse→full Decision 변환 | `route`, `mentions` 추가와 기본값 결정 |
| `agent_prompts.py` | Decision·Recommendation·Response system prompt | route contrast와 task/general/mixed 응답 규칙 추가 |
| `decision_overrides.py` | 짧은 확인 답변을 모델 없이 처리 | override가 항상 `route=task`를 갖게 해야 함 |
| `decision_model.py` | history 구성, XGrammar, Decision 추론·정규화 | route 계약 전달과 repair trace 유지 |

`decision_model.py`의 XGrammar는 현재 `any_order=True`이다. 계획대로 유지해야 한다.

### 4.2 그래프와 정책

| 파일 | 현재 역할 | Decision Routing 작업 시 변경점 |
| --- | --- | --- |
| `llm_langgraph.py` | `SessionState`, `Decision`, `TurnState`, 그래프 단계 | focus 상태, route guard, general 조건부 edge 추가 |
| `llm_policy.py` | section, restriction, 완료 작업 보호, 실행 판정 | general mutation 0 보장과 route consistency 검사 |
| `recommendation_model.py` | 요청이 있을 때만 추천 JSON 생성 | mixed도 기존 task 경로를 사용하므로 원칙상 유지 |
| `response_model.py` | 확정 상태를 자연어 JSON reply로 변환 | task/general/mixed prompt 선택 또는 호출 분리 |
| `llm_runtime_logger.py` | 턴 trace와 runtime event를 JSONL로 저장 | route, mentions, focus 전후를 trace에 남길 위치 |

### 4.3 모델과 VLM

| 파일 | 현재 역할 | 주의점 |
| --- | --- | --- |
| `model_runtime.py` | Gemma 4 12B 로딩, int8/nf4/bf16, VLM 호출 | adapter 경로를 받을 수 있으나 현재 호출자는 `None` 전달 |
| `vlm.py` | 이미지 요청 구성, verdict parsing, retry 정책 | 주문 대화 routing과 별개인 작업 검증 계층 |
| `vlm_prompts.py` | 재료·소스·뚜껑 판정 prompt | 일반대화 prompt와 섞지 않음 |
| `vlm_rag.py` | ChromaDB에서 대표 reference crop 조회 | 지원 재료의 시각 reference 전용 |
| `vlm_reference_embedding.py` | reference embedding collection 생성·검증 | 운영 턴이 아닌 사전 준비 스크립트 |

## 5. ROS 사용자 입출력 계층

| 파일 | 역할 | 핵심 입출력 |
| --- | --- | --- |
| `stt_node.py` | Clova streaming STT | 마이크 → `/stt_question` |
| `stt_nemotron_node.py` | 로컬 Nemotron STT와 VAD | 마이크 → `/stt_question` |
| `stt_hotword.py` | STT 문자열 정규화 | 순수 함수 |
| `tts_node.py` | `/llm_response`를 음성으로 재생 | `/stt_stop`, `/tts_done`으로 STT gate 제어 |
| `ui_node.py` | Flask·Socket.IO UI와 ROS 연결 | 시작·초기화·대화·상태·VLM snapshot |

STT 구현은 둘 중 하나를 실행하는 대안 관계이다. 둘 다 같은 `/stt_question` 계약을 사용한다.

## 6. `LLMLangGraphNode`의 역할

`llm_langgraph_node.py`는 대화 코드와 로봇 실행 사이의 어댑터이다.

1. 같은 base model에서 Decision·Recommendation·Response·VLM 호출 함수를 만든다.
2. ROS callback을 최대 32개 job queue에 넣고 worker thread 하나가 순서대로 처리한다.
3. 사용자 발화를 `graph.invoke()`에 전달하고 성공한 `result["session"]`만 저장한다.
4. `policy.execute=True`이면 현재 section의 주문을 실제 task 목록으로 변환한다.
5. task 하나씩 `/llm/plan`으로 보내고 완료 신호 뒤 다음 section으로 이동한다.

현재 `active_task`가 있거나 `task_queue`가 비어 있지 않으면 STT 발화를 무시한다. 따라서 **현재 코드의 자유대화 가능 시점은 로봇 작업 사이의 주문 대화 구간뿐이다.**

## 7. 로봇 제어 계층

### 7.1 공정 조정

| 파일 | 역할 | 현재 계약 |
| --- | --- | --- |
| `main_irc.py` | 단순 MAIN 상태 머신 | `/llm/plan` → rail → `/control/plan` |
| `main_vlm.py` | VLM 재확인을 포함한 MAIN 상태 머신 | `/llm/plan` → 세부 motion → VLM → retry/완료 |
| `point_pose_node.py` | 현재 setup에 등록된 IK·작업 계획기 | `/control/start`, `/control/plan` |
| `dual_grip_planner_node.py` | 별도 dual-grip 계획기 구현 | `/llm/start_control`, `/llm/task_plan` |
| `motor_motion_control_node.py` | waypoint를 Dynamixel·공압·그리퍼 명령으로 실행 | `/arm/joint_*` → `/arm/motion_done` |

### 7.2 비전과 좌표

| 파일 | 역할 | 핵심 입출력 |
| --- | --- | --- |
| `detect_yolo_sam_obb_pub.py` | RealSense+YOLO+SAM2로 pick pose 검출 | `/control/motion_done` → `/vision/raw_pick_pose` |
| `transform_node.py` | 카메라 raw pose를 로봇 base 좌표로 변환 | `/vision/raw_pick_pose` → `/vision/pick_pose` |
| `point_pose_node.py` | pick/place/CP 경로 생성 | `/vision/pick_pose` → arm waypoint |
| `motor_motion_control_node.py` | 실제 관절 trajectory 실행 | waypoint → motor → 완료 신호 |
| `main_vlm.py` | 작업 단계와 재시도 조정 | control·rail·VLM 신호를 상태 머신으로 연결 |

## 8. 핵심 ROS topic 지도

### 8.1 대화

| Topic | 발행자 | 구독자 | 의미 |
| --- | --- | --- | --- |
| `/ui/start` | UI | LLM, MAIN | 새 주문 시작 |
| `/stt_question` | STT | LLM, UI | 사용자 발화 |
| `/llm_response` | LLM | TTS, UI | 사용자에게 말할 답변 |
| `/stt/enable` | LLM | STT, UI | 마이크 입력 허용 |
| `/agent/status` | LLM | UI | 주문·로봇·화면 상태 snapshot |

### 8.2 실행

| Topic | 발행자 | 구독자 | 의미 |
| --- | --- | --- | --- |
| `/llm/plan` | LLM | MAIN | 영문 class와 반복 횟수 |
| `/llm/next` | MAIN | LLM | 첫 주문 시작 또는 현재 task 완료 |
| `/llm/reset` | MAIN | LLM | 마지막 공정 뒤 주문 완료 처리 |
| `/llm/done` | LLM | MAIN | LLM 완료 처리 종료 |
| `/reset` | MAIN | UI 등 | 전체 화면·노드 초기화 신호 |

### 8.3 VLM·Vision

| Topic | 발행자 | 구독자 | 의미 |
| --- | --- | --- | --- |
| `/llm/confirm_start` | MAIN | LLM | 현재 task의 VLM 확인 시작 |
| `/vlm/confirm` | LLM | MAIN | success 또는 retry 결과 |
| `/vision/raw_pick_pose` | detector | transform | 카메라 좌표 pick pose |
| `/vision/pick_pose` | transform | planner | base 좌표 pick pose |
| `/vision/overlay_image` | vision 계층 | LLM | VLM 판정용 최근 이미지 |

## 9. 테스트·로그·보조 파일

| 경로 | 역할 | 현재 상태 |
| --- | --- | --- |
| `logs/llm/<date>/<session>/agent_turns.jsonl` | 모델 input/raw/final, policy, 상태 전후 | 새 routing eval의 실제 trace 근거 |
| `logs/llm/<date>/<session>/runtime_events.jsonl` | task/VLM/세션 사건 | ROS full-cycle 분석 근거 |
| `evaluation_any_order_true_20260930/` | any-order 평가 결과와 full-cycle 자료 | 과거 평가 증거; 새 suite와 구분 필요 |
| `decision_adapter_v225_root_cause_review_20260930/` | v225 학습·평가·계약 보관 bundle | 운영 소스가 아니라 비교·재현 자료 |
| `src/soomac_irc/test/` | CPU 회귀 테스트 | 일부 테스트가 현재 없는 이전 모듈을 import함 |

웹 GPT가 dataset을 생성하더라도, 넘겨줄 기준 묶음은 최소한 `DECISION_SCHEMA`, `DECISION_SYSTEM`, family별 정답 예시, 금지 규칙, validator 계약을 포함해야 한다.

## 10. 현재 코드에서 확인된 불일치

### 10.1 Decision adapter가 runtime에 연결되지 않음

`LLMLangGraphNode.__init__()`은 현재 다음과 같이 base model만 로드한다.

```python
self.model, self.processor = load_model(None)
```

`model_runtime.load_model(adapter_path)`는 adapter를 붙일 수 있지만 실제 호출 경로는 `None`이다.

Cause: 현재 코드에는 runtime adapter 경로를 전달하는 설정이나 호출이 없다.  
Effect: 계획서의 “기존 Decision Adapter”를 전제로 구현하기 전에 실제 adapter 로딩 방식을 확정해야 한다.

### 10.2 등록됐지만 없는 ROS entry point

`src/irc_control_pkg/setup.py`는 아래 모듈을 등록하지만 현재 저장소에 파일이 없다.

```text
irc_control_pkg.point_pose_vlm_node
irc_control_pkg.rail_bridge
```

Cause: 원인 미확인.  
Effect: 해당 console script를 실행하면 module import 단계에서 실패한다.

### 10.3 현재 테스트와 현재 LLM 구조가 맞지 않음

`test_runtime.py`는 현재 없는 `soomac_irc.agent`, `order`, `order_change`, `recommendation`, `reply`, `validation`을 import한다.

Cause: 테스트가 이전 구조를 대상으로 남아 있다.  
Effect: routing/focus 구현 전에 현재 `llm_langgraph.py` 기준의 CPU 테스트 기반을 다시 세워야 한다.

### 10.4 제어 계약이 세 계열로 나뉨

```text
main_irc.py              ↔ /control/start, /control/plan
main_vlm.py              ↔ /control/motion, /control/vision_request, /main/confirm
dual_grip_planner_node.py ↔ /llm/start_control, /llm/task_plan
```

Cause: 원인 미확인. 서로 다른 시점의 제어 구현이 같은 브랜치에 함께 있다.  
Effect: 어떤 MAIN·planner 조합이 실제 배포 조합인지 정하지 않으면 ROS full-cycle 기준을 하나로 만들 수 없다.

### 10.5 로봇 동작 중 대화가 차단됨

`LLMLangGraphNode._process_turn()`은 `active_task` 또는 `task_queue`가 있으면 발화를 무시하고 STT를 끈다.

Cause: 기존 주문 시스템이 동작 중 새 주문 변경을 받지 않도록 설계됐다.  
Effect: “자유대화 멀티턴”이 로봇 동작 중에도 가능해야 한다면 routing만 추가해서는 동작하지 않는다.

## 11. Decision Routing + Focus 구현 접점

1. `agent_contract.py`: external sparse schema에 required `route`, optional `mentions`를 추가한다.
2. `llm_langgraph.py`: internal Decision과 `SessionState`에 route·focus를 추가하고 conditional edge를 만든다.
3. `dialogue_focus.py`: mention canonicalization, supported 여부, current/recent, TTL을 순수 Python으로 분리한다.
4. `response_model.py`와 `agent_prompts.py`: task/general/mixed 표현 계약을 분리한다.
5. `llm_runtime_logger.py` 경로: route·focus의 입력, 해석, 상태 변화를 eval 가능한 형태로 기록한다.

`general` 경로는 canonical order, recommendation, pending confirmation, robot state를 변경하지 않아야 한다. `mixed` 경로는 task 부분을 기존 pipeline에 그대로 통과시킨 뒤 일반 질문까지 답해야 한다.

## 12. 구현 전에 고정할 계약

1. unsupported 주문에서 `clarify=true`는 “지원 메뉴로 다시 선택 요청”을 뜻한다.
2. `아까 그거`가 current를 제외한 직전 focus 하나를 뜻하는지 확정한다.
3. focus TTL을 턴 수로 자를지 event 개수로 자를지 확정한다.
4. 로봇 작업 중 general 대화를 받을지 확정한다.
5. 실제 Decision adapter 경로와 실제 MAIN·planner 배포 조합을 확정한다.

## 용어 정리

- **정본(canonical state)**: 실제 주문·실행 판단의 유일한 기준 상태이다.
- **희소 Decision(sparse Decision)**: 현재 발화에 관련된 필드만 모델이 출력하는 형식이다.
- **대화 초점(dialogue focus)**: `그거`가 가리키는 대상을 해석하기 위한 임시 기억이다.
- **정책(policy)**: 모델이 낸 후보를 실제 상태나 로봇 실행으로 허용할지 결정하는 Python 규칙이다.
- **공정 조정자(orchestrator)**: LLM plan, rail, vision, arm, VLM의 순서를 연결하는 MAIN 노드이다.

