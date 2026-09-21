# 2026-09-21 주문 완료 후 시작 화면 자동 복귀

## 범위

작업 브랜치는 `llm_vlm`이다. 로봇 토픽 기준은 `main` 브랜치의 `src/irc_control_pkg/irc_control_pkg/main_vlm.py`이다. MAIN 코드, Tool 필드, 모델 프롬프트와 어댑터는 변경하지 않았다. 커밋과 push는 사용자가 진행한다.

원인: MAIN은 `/reset`에 문자열 `reset`을 보내지만, 기존 UI는 `/ui/reset`의 JSON `{"action":"complete"}`를 기다렸다. 기존 HTML도 완료 시 시작 화면을 숨겼다.

결과: 로봇 완료, 마지막 안내 처리 종료, LLM 초기화를 모두 확인한 뒤 다음 손님용 시작 화면을 표시한다.

## 완료 순서

1. 마지막 소스 작업 후 MAIN이 `/llm/reset`을 보낸다. LLM은 기존 마지막 안내와 `/llm/done`을 발행한다.
2. MAIN이 `/reset` 문자열 `reset`을 발행하고 다음 시작을 기다린다.
3. UI는 MAIN 완료와 `/agent/status`의 완료 상태, 마지막 안내의 `/tts/utterance_done`을 모두 기다린다. 세 토픽의 수신 순서는 무관하다.
4. 조건이 모이면 UI가 기존 `/ui/reset`에 `{"action":"complete"}`를 한 번 발행한다. LLM의 `work_state=idle`, `reset_allowed=true` 응답을 기다린다.
5. UI는 현재 대화, 재료 그림, VLM 팝업과 화면 상태를 비우고 시작 화면으로 돌아간다. 영구 기록은 삭제하지 않는다.

## 파일별 변경

| 파일 | 변경 |
| --- | --- |
| `src/soomac_irc/soomac_irc/llm_node.py` | 마지막 안내를 `ORDER_COMPLETE_REPLY` 상수로 공유하고, UI 상태에 `completion_reply`를 추가한다. Tool 입력·출력 필드가 아니다. |
| `src/soomac_irc/soomac_irc/tts_node.py` | 기존 `/tts_done`을 유지하고 `/tts/utterance_done`을 추가한다. 출력 버퍼 재생이 끝난 뒤 원문과 종료 상태를 보낸다. 합성 예외도 실패 종료로 알린다. |
| `src/soomac_irc/soomac_irc/ui_node.py` | MAIN의 `/reset`을 수신한다. 이전 발화의 종료 신호는 완료로 보지 않는다. LLM 초기화 확인 전에는 다음 시작 화면을 열지 않는다. |
| `src/soomac_irc/soomac_irc/templates/index.html` | `work_complete`에서도 기존 화면 초기화 함수를 호출한다. |
| `src/soomac_irc/test/test_order_completion.py` | 순서 교환, 중복, 잘못된 메시지, TTS 실패, 연속 주문, 출력 버퍼 종료 검사를 추가한다. |

새 토픽은 `std_msgs/msg/String`이며 데이터는 다음과 같다. `text`는 TTS 정규화 전 `/llm_response` 원문을 strip한 값이다.

```json
{"text":"소스까지 모두 담았어요. 이용해 주셔서 감사합니다.","status":"finished"}
```

`status=failed`도 발화 처리가 끝난 것으로 간주해 화면 복귀를 허용한다. 재생 성공으로 기록하지 않으며 경고를 남긴다. TTS가 아예 실행되지 않거나 프로세스가 종료돼 종료 신호가 오지 않으면 자동 복귀하지 않는다. 임의 타이머로 완료를 추정하지 않는다.

## 검증

- 기존 22개 회귀 검사와 신규 9개 검사, 총 31개 통과. 신규 검사에는 세 신호 수신 순서 6가지와 시작 버튼 동시 요청 8개의 단일 발행 검사가 포함된다.
- `main` 제어기와 실제 LLM 메서드로 소스 3종 × PASS/FAIL/UNCERTAIN 및 VLM OFF, 총 10가지 모의 사이클 통과.
- 같은 MAIN·LLM·UI 인스턴스로 토마토 주문 9작업 → 자동 복귀 → 크림 주문 9작업 → 자동 복귀 통과.
- 실제 Flask·Socket.IO 테스트 클라이언트의 시작 → 완료 → 다음 시작 통과. HTML inline JavaScript 문법과 완료 핸들러의 시작 화면 표시 검사 통과.
- 실제 TTS 합성 처리 메서드에 모의 오디오를 연결해 정상, 빈 청크, 중간 합성 실패, 출력 실패, 정규화 후 빈 문장 5가지 종료 처리를 확인했다. 로봇·카메라·GPU·실제 스피커는 실행하지 않았다. ROS 전송은 모의 검사이며 실기 전체 사이클 검증은 별도다.

재검사 명령:

```bash
cd /home/roma/IRC_SOOMAC3.0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/soomac_irc /home/roma/miniconda3/envs/gemma4_env/bin/python -m unittest discover -s src/soomac_irc/test -p 'test_*.py'
```

읽기 전용 Codex 교차검증에서 시작 요청의 확인과 활성화 사이 경쟁 조건을 지적했다. `handle_start`의 확인·세션 열기·발행을 같은 `ros_node_lock` 안으로 이동하고 동시 요청 8개가 시작 메시지 1개만 만드는 회귀 검사를 추가해 통과했다.

## 적용 주의

LLM·UI·TTS 세 노드를 모두 이 소스로 재시작하고 브라우저도 새로고침해야 한다. 예전 TTS는 `/tts/utterance_done`을 발행하지 않으므로 새 UI와 혼용하면 복귀를 기다리게 된다.

이번 변경은 `main` 제어기와의 정상 주문 완료 경로에 한정한다. 현재 `llm_vlm` 브랜치의 제어기는 `main`과 별개이며 수정하지 않았다. 시작 직후 수동 뒤로가기·홈으로 취소한 뒤 MAIN을 다시 시작하는 문제도 이번 수정 범위가 아니다.
