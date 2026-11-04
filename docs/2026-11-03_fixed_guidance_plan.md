# 고정 안내 자연화 — 적용 계획과 검증

사용자 승인: 2026-11-03, “plan md 세우고 한번 적용”. 이전 고정 안내 제거 방침보다 이번 실행·단계 안내 유지 및 표현 개선 요청을 우선한다.

## 1. 목적과 역할

시연에서 확정된 수량을 정확히 말하면서 안내를 자연스럽게 한다. Decision은 의도·주문 변경을 해석하고, Recommendation은 후보를 제안하며, Response는 일반 대화·추천 설명·선택 변경 설명을 담당한다. 실행 시작과 단계 안내는 기존 Python 경로를 유지한다. 이 변경으로 모델 전반의 수량 오류가 해결됐다고 주장하지 않는다.

Cause: P가 최신 버섯 low 입력에도 high라고 발화한 반례가 있으며, 실행 안내는 직접 학습 예시가 없는 영역이었다.
Effect: 확정 주문의 재료·수량을 표현하는 문장만 개선하고 실행 권한·상태·계획은 보존한다.

## 2. 적용할 문장

| 경우 | 문구 |
| --- | --- |
| 한 재료 low / normal / high | 버섯을 적은 양으로 / 보통 양으로 / 많이 담을게요. |
| 두 재료 같은 양 | 양파와 버섯을 적은 양으로 / 보통 양으로 / 많이 담을게요. |
| 두 재료 다른 양 | 양파는 적게, 버섯은 많이 담을게요. |
| 미리 선택한 재료 확인 | 미리 고르신 양파와 버섯을 적은 양으로 담을까요? |
| 다음 단계 선택 | 이제 야채를 골라 주세요. 양파와 버섯 중 어떤 걸 얼마나 담을까요? 원하지 않으면 ‘다음 단계’라고 말씀해 주세요. |

면 2종·토핑 6종에 적용한다. 을/를·은/는·와/과를 받침에 맞춰 사용한다. high는 ‘많이’로 통일한다. 게살을 육류로 분류하지 않도록 해당 단계는 재료 이름으로 안내한다. 뚜껑·소스·최종 완료 문구는 유지한다. 제외된 재료는 기존 실행 대상 추출 결과에 따라 발화하지 않는다. 빈 실행 대상의 기존 오류 처리를 유지한다.

## 3. 변경 범위

`llm_langgraph.py`의 실행/미리 선택 확인 문장 생성과 `llm_langgraph_node.py`의 세 가지 선택 단계 문구, 관련 CPU 테스트만 변경한다. 사용자 발화를 Python 키워드로 판단하는 로직은 추가하지 않는다. history에는 실제 발행 문장이 기존 방식대로 저장된다.

현재 작업 트리는 llm_final_2이며 기존 미커밋 변경을 보존한다. GPU·모델 로딩·학습·adapter 교체·ROS 기동·원격 작업·VLM·checkpoint·원본 학습/평가 자료·과거 증거는 변경하지 않는다.

## 4. 작업 순서와 완료 조건

1. [x] 계획 저장 및 실제 문장 생성/호출 지점 확인.
2. [x] 문장 생성과 단계 문구 적용. 현재 fixed/vLLM 설정 유지.
3. [x] CPU로 모든 면/토핑 수량 조합, 미리 선택, 제외/빈 대상, 오류 fallback, 실제 노드의 계획 발행·history·모델 미호출 회귀 확인.
4. [x] 검사 결과와 출력 예시를 이 문서 및 worklog에 기록.

실물 및 새 Decision 통합 성공은 이 CPU 검사로 보증하지 않는다. 기존 실패 기록은 당시 문구와 함께 보존한다.

## 5. 결과

적용 완료. `test_fixed_guidance_wording.py`에서 한 재료 24조합(면2종+토핑6종 × 3수량), 두 재료 27조합(3단계 × 9수량쌍), 제외/다른 단계/빈 대상 및 기존 pending 형태를 확인했다. 기존 실행 모드·단계 전환·멀티턴 계약 검사와 합쳐 **173 passed, 6 subtests passed (2.15초)**이다. GPU나 ROS 실물을 실행한 결과가 아니다.

최초 검사는 172 passed / 1 failed / 6 subtests passed였다. 실패 위치는 `test_section_question_events.py`의 빈 단계 실행 요청 검사였으며, 테스트가 과거 안내의 ‘육류’라는 단어를 요구했다. 새 문구의 실제 재료명 ‘소시지와 게살’을 기대하도록 수정한 뒤 동일 범위를 재검사해 통과했다. 단계 유지·history·실행 권한 검사는 그대로 보존했다.

변경 후 CPU 회귀에서 버섯 high→low 정정 뒤 고정 안내는 **“양파와 버섯을 적은 양으로 담을게요.”**이다. Response 호출 없이 onion repeat_count=1 발행 및 버섯 repeat_count=1 대기 계획이 유지됐다. 계획 발행 실패 시 시작 안내를 말하지 않는 검사, 모델 안내 오류 시 고정 안내 fallback, 일반 비실행 대화의 모델 경로 검사도 통과했다.

변경 위치: `llm_langgraph.py`의 `_describe_portioned_items`, `build_preselected_confirmation_reply`, `build_execution_started_reply`; `llm_langgraph_node.py`의 `_section_prompt`. 실행 모드·backend·adapter·정책·VLM은 이번 변경 대상이 아니다. 실제 모델에 새 안내 history가 들어간 후의 후속 생성과 TTS 청취는 미검증이다.

재현 명령:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/soomac_irc /home/roma/miniconda3/envs/soomac_llm/bin/python -m pytest -q src/soomac_irc/test/test_fixed_guidance_wording.py src/soomac_irc/test/test_execution_reply_mode.py src/soomac_irc/test/test_section_question_events.py src/soomac_irc/test/test_natural_multiturn_contract.py
```
