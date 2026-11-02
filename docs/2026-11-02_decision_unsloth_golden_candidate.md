# Decision Unsloth Golden 후보 — 이미지·TAR·원격 smoke 준비 (2026-11-02)

상태: **GPU 미검증 후보. production-ready 아님. 원격 제출 보류.**
코드: [tools/unsloth_remote_golden/](../tools/unsloth_remote_golden/)

## 세 질문에 대한 답

| 질문 | 답 |
|---|---|
| 1. 로컬 Response Unsloth 환경과 새 Decision Docker가 일치하는가? | **파일·CPU 수준에서 일치.** 필수 134개 배포판의 34,261개 파일이 Golden 디스크와 바이트 동일. Python은 같은 conda 빌드(3.10.20). CPU 인코딩 결과도 동일. **GPU 생성 코드·수치 일치는 미확인**(smoke에서 확인할 항목) |
| 2. TAR를 만들고 로드 검증했는가? | c3 후보 TAR: 제작·blob 해시·`docker image load` 완료. smoke용 s1 TAR: 아래 표 |
| 3. 원격 GPU 사용 전 확인할 것은? | ① 원격 드라이버 ≥ 580.65.06 (현재 **UNKNOWN**) ② 기존 HF 작업과 같은 물리 GPU를 쓰는 문제 ③ 2-job smoke(저장→실패→재개) ④ 영수증 관문 전부 통과 |

## 원격 제출 판정: 보류

- 원인: 원격 서버의 GPU는 RTX PRO 6000 **1장**이다. `/api/resources` 기준 total 102.6GB이고, 단독 모드가 슬롯 2개를 모두 쓴다. 비어 있는 "공유" 슬롯은 실행 중인 HF Decision 작업(build05-speed)과 같은 GPU를 소프트웨어 예산으로 나눈 것이며 MIG가 아니다(규격서 317-325줄).
- 결과: 사용자 조건 2("기존 HF 학습과 GPU·VRAM을 공유하지 않음을 확인")를 충족하지 못해 제출하지 않았다. 기존 HF 학습은 그대로 둔다.
- 추가 위험: production의 `acquire_gpu_lock`([production.py:444-455](../tools/unsloth_training/production.py#L444-L455))은 `nvidia-smi`에 다른 GPU 프로세스가 보이면 즉시 종료한다. 컨테이너 안에서 다른 컨테이너의 프로세스가 보이는지는 **UNKNOWN**이다. 보인다면 HF 작업이 도는 동안에는 smoke와 train 모두 수 초 안에 종료된다.

## Golden 대 Docker

| 항목 | Golden (PID 1405238) | Docker c3 / s1 | 판정 |
|---|---|---|---|
| Python | conda python-3.10.20-h17756b0_1 | 같은 아카이브(SHA 일치) | 동일 |
| torch / triton | 2.12.1+cu130 / 3.7.1 | 동일 휠 | 동일 |
| unsloth / zoo | 2026.10.11 | 동일 | 동일 |
| transformers / peft / bnb | 5.13.1 / 0.19.1 / 0.49.2 | 동일 | 동일 |
| accelerate / trl / safetensors | 1.14.0 / 1.8.0 / 0.8.0 | 동일 | 동일 |
| xformers | 0.0.35 PyPI cu128 빌드, torch 2.12에서 로드 실패 → unsloth가 `xformers=None` | 같은 휠(같은 실패 상태 재현) | **호환 합격이 아님**. 실패 상태만 같다 |
| 배포판 수 | 230(상속 포함) | 필수 134 | 96개 의도적 제외(langchain·chromadb·fastapi 등) |
| sentence-transformers | 있음 → unsloth import 시 ST 클래스 4곳 패치 | 없음 | 패치 대상이 ST 클래스뿐이라 Decision Trainer 경로에는 영향 없음(코드 확인). import 부수효과는 미확인 |
| gcc / glibc | 11.4.0 / 2.35-0ubuntu3.15 | 동일 apt 버전 | 동일 |
| locale | en_US.UTF-8 | C.UTF-8 | 영향 없음(데이터 I/O가 UTF-8 명시) |
| 캐시 경로 | /tmp/soomac-training-accel-20261101 | /tmp/soomac-unsloth-container | 생성 코드 SHA로 GPU에서 확인 |
| 생성 코드 `unsloth_compiled_module_gemma4_unified.py` | `a8f33a56…` | **smoke에서 비교** | 미확인 |
| Decision 수치(loss) | 같은 입력의 Golden Decision 실행 없음 | — | **UNKNOWN** |

## 이미지와 TAR

| 이름 | 이미지 ID | 용도 | TAR | SHA-256 |
|---|---|---|---|---|
| c3 후보 | `sha256:86bb8967d128840004f31d392eb67f90e090066c8f112cbc3cbe165324e22a2c` | 기준 후보(드라이버 검사 fail-open 버전) | `~/Downloads/soomac-decision-unsloth-golden-candidate-20261102-c3.tar` (3,495,497,216 B) | `d9688f8d…d955fa` |
| s1 smoke | `sha256:b354fad3ff8c90cad939293292df2968f514b18435884d3ce23cbcc95b0f8cf7` | 원격 smoke 전용(train 불가) | `~/Downloads/soomac-decision-unsloth-golden-smoke-20261102-s1.tar` (3,495,517,696 B; blob 18개 해시 일치, OCI 매니페스트 digest = 이미지 ID) | `03fd1fe04a9f3cc4d6b98ab42daff36a71d87b7ea2db35d7daedfad66690c245` |

s1은 c3의 RootFS 레이어 12개 위에 4개 레이어만 얹었다(앞 12개가 동일함을 확인). 바뀐 파일은 다음 셋뿐이다.
- `/app/run_training.sh`: 드라이버 검사를 fail-closed로 변경. UNKNOWN·형식 오류·580.65.06 미만이면 exit 69.
- `/app/smoke_entry.sh`: 새 ENTRYPOINT, CMD는 `smoke-validation`.
- `/app/gpu_validate.py`

`production.py` SHA는 빌드 중에 `86638bad…`로 재확인했다.

## 원격 smoke 절차 (사용자가 GPU 공유를 승인했을 때만)

1. 웹에서 s1 TAR를 업로드하고 검사한다. 설정은 **공유**, CPU 8, RAM 32 GiB 이상(여유가 있으면 40), 공유 메모리 4 GiB다.
   - 근거: Golden의 최대 RSS는 25.8GB였다. Golden smoke의 GPU 최대 사용량은 12.8GB였다.
2. job1: smoke 10 step으로 돌린다. step 5를 저장한 뒤 production이 일부러 실패한다.
   - 로그에 `SMOKE_PHASE1 EXPECTED_FAILURE … after committed step 5`가 나와야 정상이다.
   - `SOOMAC_DRIVER_CHECK status=FAIL/UNKNOWN`이 나오면 그 즉시 이 후보를 중단한다.
3. job1 상세에서 **재개 설정 → 새 작업으로 재개**를 누른다. 이것이 job2다.
   - 체크포인트 5에서 복원한 뒤 6~10 step을 진행한다.
   - 이어서 같은 컨테이너에서 `gpu_validate.py`가 one-row loss/gradient/env 증거를 `/checkpoints/gpu-validation/`에 쓴다.
4. 두 작업의 결과 TAR를 받아 압축을 풀고, 로컬에서 아래 명령을 실행한다.
   ```bash
   python3 -I tools/unsloth_remote_golden/make_receipt.py --job1 <job1 checkpoints> --job2 <job2 checkpoints> \
     --tested-image sha256:b354fad3ff8c90cad939293292df2968f514b18435884d3ce23cbcc95b0f8cf7 --out <new context>/gpu-validation
   ```
5. PASS가 나오면 [Dockerfile.final](../tools/unsloth_remote_golden/Dockerfile.final)로 s1 위에 영수증을 얹은 train 이미지를 만든다. ENTRYPOINT는 `run_training.sh`, CMD는 `train`이다. 이 이미지의 TAR가 최종본이다.

## 영수증 관문 ([make_receipt.py](../tools/unsloth_remote_golden/make_receipt.py))

모든 관문은 증거 파일을 직접 읽고 수치를 다시 계산한다. 하나라도 실패하면 아무 파일도 쓰지 않는다.

| 관문 | 내용 |
|---|---|
| job1 실패·step 5 마커 | `FAILURE.json`이 있고 resume.json이 step 5의 실제 디렉터리를 가리킴 |
| identity | release·manifest·production·model config·tokenizer·model builder SHA가 Golden 값과 같음. 9개 패키지 버전도 같음 |
| 재개 | 재개 후 10 step 완료. `RESTORE_EVIDENCE` 4종(adapter/optimizer/scheduler/rng) equal |
| optimizer 갱신 | step 5 어댑터의 lora_B가 모두 0이 아님. step 5와 최종 어댑터 텐서가 다름 |
| GPU 메모리 | job1·job2·검사 프로세스의 NVML 최대 사용량 ≤ 32e9, 예산 위반 없음 |
| 드라이버 | ≥ 580.65.06, torch CUDA 13.0 |
| loss | native selected CE와 model loss의 차이 ≤ min(atol + rtol·|CE|, 1e-3). 선택 라벨이 인코딩 라벨과 같음 |
| Golden 생성 코드 | compiled SHA = `a8f33a56…` |
| gradient | 모든 LoRA 텐서에 grad가 있고 유한함. lora_B grad가 0이 아님 |

고정 데이터 시험 결과는 다음과 같다. 로컬 Golden Response smoke 출력을 심볼릭 링크로만 읽었다.
- 정상 1건: PASS. `production.verify_validation_receipt`도 통과했다.
- 오류 주입 5건(생성 코드 불일치, CE 차이, 드라이버 575, 예산 초과, lora_B grad 0): 각각 해당 관문 하나에서만 FAIL했다.

## 어댑터 변환·vLLM 증거의 적용 범위 (조건 6)

| 기존 증거 | 범위 | 새 Decision 어댑터에 적용? |
|---|---|---|
| PEFT 재로드(텐서 656개 일치, forward 유한) | 로컬 Golden의 **Response** smoke 어댑터 | 아니오. 구조(타깃 모듈, r16/α32)는 같지만 Decision 어댑터로는 실행하지 않음 |
| 변환기(전역 K→V 16개 복제) + vLLM 0.32 로딩 + JSON 스키마 3건 | Response smoke 어댑터, 격리 포트 18001 | 아니오 |

새 Decision 어댑터에 필요한 것:
1. 원격 결과 어댑터의 키·shape이 Golden Response 어댑터와 같은지 비교(CPU).
2. 기존 변환기를 실행하고 텐서 수·config를 대조(CPU).
3. PEFT 재로드 forward를 유한 값으로 확인(GPU).
4. 격리 vLLM 0.32에서 Decision 스키마 디코드 확인(GPU).

3·4번은 로컬 GPU가 필요하다. Response 학습이 끝난 뒤에만 할 수 있다.

## 최종 판정

- c3·s1 모두 **GPU 미검증**이며 production-ready가 아니다.
- 원격 드라이버: **UNKNOWN**.
- Golden 대비 Decision 수치 동등성: **UNKNOWN**. 생성 코드 동일성은 smoke로 확인 가능하다.
- 원격 train 투입: **불가**. smoke 영수증이 없고, train 이미지는 영수증 없이 시작을 거부한다(fail-closed 유지).

## 갱신: 원격 공유 슬롯 smoke 결과와 train TAR

### 바뀐 점 (사용자 지시: 슬롯 2에서 HF와 병렬 실행)
- s2 = c3 + 다음 4가지 변경이다.
  - [production_shared_slot.py](../tools/unsloth_remote_golden/production_shared_slot.py)를 이미지 안의 production.py로 사용한다. 원본과 다른 곳은 `acquire_gpu_lock` 함수와 그 호출부 두 곳뿐이다. 공유 슬롯에서만 다른 테넌트의 GPU 프로세스를 허용한다. 로컬 production.py는 수정하지 않았다.
  - 예산 ENV를 32e9로 설정했다. 영수증 예산과 같은 값이다.
  - 드라이버 검사를 fail-closed로 바꿨다.
  - smoke entrypoint를 추가했다.
- 실제로는 컨테이너 안에서 HF 프로세스가 보이지 않았다. 로그에 `shared_slot_other_gpu_processes`가 없다. 따라서 이 예외는 쓰이지 않았다.

### 원격 결과 (s2 `sha256:ab3bf16e…`)
| 작업 | 결과 |
|---|---|
| job1 `5e5605eb` (RAM 8g) | step 5 저장 후 의도적 실패, exit 1, OOM 아님. 결과 TAR SHA `846b35c5…` 서버 값과 일치 |
| job2 `caae7370` (RAM 8g) | step 6에서 **RAM OOM**(exit 137). 영수증에서 제외 |
| job3 `8c48edbf` (RAM 32g) | job1에서 재개해 5→10 완료, gpu_validate PASS, exit 0. 결과 TAR SHA `8cf54db5…` 일치 |

[make_receipt.py](../tools/unsloth_remote_golden/make_receipt.py) 판정은 **PASS**다.
- 원격 드라이버 **595.99.02**, CUDA 13.0, sm_120.
- GPU 최대 사용량 13.7GB(예산 32e9).
- native CE와 model loss의 차이 0.
- lora_B 328/328 gradient가 0이 아님. step 5→10 사이 텐서 656개 변경.
- 재개 4종(adapter·optimizer·scheduler·rng) 모두 일치.
- 생성 코드는 Golden과 `compile_threads` 외 바이트 동일하다.
  - 원인: Unsloth가 `min(32, max(4, os.cpu_count()))`를 파일에 직접 넣는다. 로컬은 CPU 32개, 원격은 24개다.
  - 결과: 원격 SHA `01cb73dd…`가 Golden 파일에서 이 값만 24로 바꾼 SHA와 정확히 같다. Codex도 조건부 찬성했다.
- Golden 대비 Decision 수치 동등성: 여전히 **UNKNOWN**.

### train TAR
- 이미지 t1 `sha256:4e7279e7cd23288d4040fa6b5da8626aaa0996dd6be4ea7cb785c059483e804e`.
  - s2 레이어 17개 위에 영수증 레이어 1개를 얹었다. [build_final_golden.py](../tools/unsloth_remote_golden/build_final_golden.py)로 계보를 검사했다.
- TAR `~/Downloads/soomac-decision-unsloth-golden-train-20261102-t1.tar`: 3,495,539,200 B, SHA `1a449c3adf2fe023589f8d96eb4710ea33f7cc467d544c380bc4654466cd0ef1`, blob 20개 해시 일치.
- CPU train 경로 시험: 영수증 검증이 통과했다(`ENTRY_GATES` 영수증 SHA `36f9cfe1…`). identity는 train, 1,638 step, 생성 코드 `01cb73dd…`, 예산 32e9다. GPU가 없어서 GPU 잠금 단계에서 종료했다(정상).
- 원격 설정: **공유**, CPU 8, **RAM 32 이상**, 공유 메모리 4.
  - 원인: RAM 8g에서는 재개 작업이 OOM으로 죽었다. 검사 프로세스의 최대 RSS는 25.9GB였다.
- 남은 위험(추측): 1,638 step 동안의 GPU 최대 사용량은 smoke로 증명되지 않았다. 32e9를 넘으면 Budget이 학습을 멈춘다.
