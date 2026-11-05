# Qwen STT·TTS·D6 vLLM 동시 상주 VRAM — 16K와12K 비교

**추가 이미지 크기 시험 완료(2026-11-05):** [7장 해상도별 결과](2026-11-05_vlm_resolution_vram_review.md). 운영domain17의이미지발행자0/45초수신0으로실제토픽해상도는UNKNOWN이다. 로컬640×480설정기반fixture와FHD에서7장VLM추론을실행했고각peak26.54GiB, FHD VLM+Qwen30초STT+긴TTS동시부하는27.83GiB/여유4.03GiB로완주했다. 앞선더높은12K/.50 peak28.97GiB기록은유지한다. 운영16K/.58미변경·시험노드종료.

**3/3 측정 완료. 현재16K/.58은 동시 부하에서 여유512MiB 경계에 도달해 중단했다. 12K/.50은 같은 부하를 완주했고 관측 최고28.97GiB, 최소 여유2.89GiB였다. 12K/.50을 운영 후보로 권고하지만 운영 기본값은 아직16K/.58이다.**

## 1. 실제 올린 구성

| 구성 | 이번 실행 |
|---|---|
| STT | **Qwen3-ASR-1.7B-hf, BF16 CUDA**, gemma4_env. `/home/roma/models/audio/stt/Qwen3-ASR-1.7B-hf` |
| TTS | 실제 CosyVoice3 TTSNode, fp16=False/load_trt=False, token_hop_len25, soomac_speaker 출력 |
| LLM/VLM | 실제 LLMLangGraphNode, vLLM NF4/BF16, D6/Q6/Pbest150 별도 LoRA, 전체 보강 프롬프트, 고정 실행 안내·VLM 실패 이유 ON |
| UI/MAIN | 실제 ROS 노드, ROS_DOMAIN_ID217/localhost 격리, MAIN WAIT_START. 제어 노드는 실행하지 않음 |
| 입력 장치 | 사용자 확인: 카메라를 직접 쓰지 않고 이미지 토픽을 받는 구성. 마이크는 현재 없음 |

STT 클래스가 NemotronSttNode를 상속하지만 **이번에 로딩한 음성 모델은 Qwen**이다. Nemotron 가중치는 로드하지 않았다. Silero VAD는 CPU에 상주만 했고 VAD 판단/마이크 수음은 실행하지 않았다. 저장 음성10초와30초를 Qwen의 실제 `_recognize_utterance`로 보냈다.

카메라/YOLO/SAM/모터는 이번 PC 측정에 넣지 않았다. 기존 VLM 함수에 저장 이미지7장을 넣어 GPU 연산을 측정했고, 외부 이미지 토픽 수신과 실제 로봇 판정은 미검증이다. 사용 이미지 크기는 참고 crop202×415 +512×512 이미지6장이다. 임의로 실제 완료를 발행하지 않았다.

## 2. GPU 전체 관측값

NVML50ms 표본, 데스크톱과 모든 모델을 합친 **동일 시점 장치 사용량**이다. 총 메모리는31.86GiB로 보고됐으며, 표의 여유는 동일NVML 기준이다. 이 NVML v1 사용량은 드라이버 예약 약0.49GiB를 포함한다. 종료 후 v1/v2 동시 조회에서 v1.used = v2.used + v2.reserved를 확인했다(NVML_ACCOUNTING.json). 따라서 nvidia-smi의 모델/앱 사용량 표기보다 약0.49GiB 높게 보일 수 있으며, 표의 여유값에는 이 차이가 없다.

| 단계 |16K / GPU예약.58|12K / GPU예약.50|
|---|---:|---:|
| 모든 노드 로드 후 대기 최고 |26.65GiB|24.67GiB|
| 필수17턴 추론 최고 |30.47GiB|27.54GiB|
| 이미지7장 VLM 추론 최고 |30.47GiB|27.54GiB|
| STT30초·긴TTS·LLM 동시 부하 최고 |**31.37GiB — 중단 전 관측치**|**28.97GiB — 완주**|
| 동시 부하 최소 여유 |**0.49GiB**|**2.89GiB**|

16K에서는512MiB 미만4연속 표본으로 자원 guard가 작동했다. **CUDA OOM이나 PC 정지를 관측했다는 뜻은 아니다.** 이때 모니터도 종료되어 이후 종료 과정의 최고값은 미측정이다. 따라서31.37GiB는 완주 피크가 아니라 중단 전까지의 하한이다. 종료 중 TTS가 WAV는 생성했으나 ROS context 종료 뒤 finished 발행에 실패한 로그를 보존했다. 이를 독립 모델 추론 오류로 오인하지 않는다.

12K에서는 모든 작업과 후속 대기를 완주했다. RAM available도 약40GiB 이상이었다. 각 시험의 최대 표본 간격은16K50.27ms,12K353.71ms다. 짧은 순간의 실제 최고값을 모두 잡았다는 보장은 없다.

동시 부하 단계에서 **vLLM 프로세스 자체의 관측 최대**는18.67→16.72GiB였다. 같은 TTS 입력/출력이더라도 TTS 프로세스 최대는6.44→5.96GiB로 달랐다. 따라서 전체2.40GiB 차이를 토큰 상한 단독 효과로 해석하지 않는다. 길이·maxbatched·예약률을 함께 바꾼 조합 비교이고, 반복 기동/장시간 누수 시험은 아니다. 프로세스별 서로 다른 시점의 최고값을 합산한 수치도 아니다.

## 3. 12K에서 입력·출력이 유지됐는가

전체 보강 프롬프트 SHA는 기존 시험본과 같다. 어댑터·dtype·출력 한도·최근20메시지 정책을 유지했다. 클라이언트는 이 시험 프로세스에서만 입력 예산을 D/Q11,264+출력1,024, P11,776+출력512로 맞췄다. 서버도 max_model_len/max_num_batched_tokens12,288이다.

- 두 설정 모두 필수 compound6 + split11 = **17/17 상태·계획·전체 계약 통과**. 실제 D17/Q2/P9회 호출, 실행 안내8턴은 Python 고정문장이다.
- 12K의 추가 history trim은 **0건**이다. 기본20메시지 창 자체를 해제한 것은 아니다.
- 12K HTTP 텍스트29건(필수28+긴 입력1)은 모두 길이 예산 안, finish_reason=stop이다. VLM1회는 별도 이미지 요청이며 텍스트29건에 포함하지 않는다.
- 가장 긴 재사용 실제 입력은9,067토큰 + 예약출력1,024 = **10,091**이다. 10,240이면149토큰,12,288이면2,197토큰 여유다. 이 근거로10K에 바로 붙이지 않고12K를 권고한다.
- 17턴 통과로 자유로운 대화 전체의 성공을 보장하지 않는다. 앞선82턴의 분류/짧은 지시 오류와 서버 재실행 간 변동은 여전히 남으며, 이번에 전부 재평가하지 않았다.

12K에서 STT10초0.85초, STT30초2.56초, 필수17턴32.48초, VLM7장2.13초였다. TTS 긴 응답32.88초는 **합성+실제 재생**까지 포함한다. 단회/캐시 조건이므로 정밀 속도비교로 사용하지 않는다.

## 4. 교차검수·종료·현재 적용 상태

읽기 전용 Codex 검수는 조건부였다. 감시 오류 전파, 오래된run 재사용 금지, 정리 예외 처리, UI 시작 경로와 실제 입력 범위 표기를 지적했다. 16K 당시에도 localhost ROS217에서 제어 노드 없음·UI/MAIN 준비로그를 확인했고 실제 시작/로봇 토픽은 보내지 않았다. 12K 측정기는 추가로 UI localhost 바인딩·start/reset 이벤트 차단, monitor 오류 전파, stale-run 거부를 넣었다. 운영 UI와 모델 소스는 변경하지 않았다.

원본 runtime/source SHA 비교 결과 변경0개, 프롬프트 동일. 두 시험 소유scope12개 모두inactive, GPU compute빈목록,5000/8000 포트 반환을 확인했다. 첫 시험 STT scope의 stop_code5는 이미 자연 종료되어 unit이 없던 경우이며 후속 inactive와 GPU반환으로 확인했다. 모든 실패 로그를 유지했다.

**운영 설정은 아직16K/.58이다.** 12K/.50은 임시 시험만 했다. 실제 채택 시 server/client를 함께 변경해야 하며 서버 환경변수만12K로 낮추면 현재16K 클라이언트의 시작 검사에서 거부된다. 별도 LoRA 병합·학습·원격·VLM 정책·ROS topic 변경은 없다.

Cause:16K/.58 구성은 STT/TTS 및 첫 추론의 추가 할당까지 겹치면 GPU 여유가 매우 작아졌다.
Effect: 전체 보강을 유지하는12K/.50 후보는 같은 관측 부하를 완주하며 약2.89GiB 여유를 남겼다. 이 범위의 권고이며 모든 해상도·장시간 부하 보장은 아니다.

[수치·요청·작업별 원문](../training/learning_first_v43/final/goals/GL10/stack_vram_12k_20261105/COMPARISON.json) · [12K 실제 답변](../training/learning_first_v43/final/goals/GL10/stack_vram_12k_20261105/TURNS.jsonl) · [16K 중단 근거](../training/learning_first_v43/final/goals/GL10/stack_vram_20261105/GUARD.json) · [종료 확인](../training/learning_first_v43/final/goals/GL10/stack_vram_12k_20261105/RESOURCE_CHECK.json)

최종 수치·범위 교차검수: Codex 조건부 찬성(REVIEW_RESULT.log). 수치와17턴·임시적용 범위는 일치하나 전체82턴/장시간/실물 안전 보장은 아니다. 검수에 제공한4개 파일만으로 시험용 runner 보완 구현 자체를 독립 확인한 것은 아니며, root가 해당 소스와 실제 준비·종료 로그를 직접 대조했다.
