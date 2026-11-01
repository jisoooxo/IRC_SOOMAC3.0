# 로컬 학습 가속 비교 도구

참고: [봉인된 r5 프로파일러](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed/common/profiling.py), [토큰 영수증](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed/validation/TOKEN_MASK_ROWS.jsonl), [Unsloth FastModel 구현](/tmp/soomac-training-accel-20261101/venv/lib/python3.10/site-packages/unsloth/models/loader.py)
→ 동일한 사전 인코딩 입력으로 baseline과 Unsloth의 로컬 일회성 optimizer update 시간을 비교한다.

## CPU 준비

`benchmark.py prepare --agent response|recommendation --output NEW_DIR`는 최신 MANIFEST와 봉인 processor를 확인한다. 토큰 영수증에서 학습·검증의 최장 행과 상위 15행을 고르고 **16행만 인코딩**한다. `encoded.json`과 그 SHA-256을 기록한다. 첫 실험 workload는 최장 행을 16번 반복한다. 검증 행이 선택될 수 있으나 이는 폐기용 벤치마크에서만 사용하며 본 학습 데이터로 옮기지 않는다. Response 상한은 9,216, Recommendation은 5,120이며 history를 줄이지 않는다. CPU 준비 실측은 Response 최장 8,430토큰, Recommendation 최장 4,332토큰이었다.

## 별도 프로세스 실행

각 엔진은 `benchmark.py run --agent AGENT --engine baseline|unsloth --encoded PREPARED/encoded.json --output NEW_DIR`로 **서로 다른 프로세스**에서 실행한다. 기본 `smoke`는 warmup 없이 optimizer update 1회만 잰다. 통과 후 `--mode full`을 명시하면 warmup 2회와 timed update 3회를 잰다. 각 timed update의 평균 loss도 동기화 후 계측 밖에서 기록한다. full은 계측 밖에서 추가 단일 microbatch의 attention operator trace를 내보내며 smoke에서는 `--trace`로 요청한다. `--save-snapshot`은 별도 검증용 adapter·optimizer 사본만 남긴다. 어떤 출력도 본 학습 완료나 resume 호환성 증거가 아니다.

동일한 최장 인코딩 행 16회, microbatch 1·유효 배치 16, seed 42, LoRA rank 16/alpha 32/dropout 0.05, NF4 double quant/BF16, Response LR `5e-5`·Recommendation LR `1e-4`, PagedAdamW8bit을 유지한다. baseline은 봉인 프로파일러의 PEFT·reentrant 체크포인팅·SDPA 로더와 scored suffix loss를 따른다. Unsloth는 로컬 기반 모델에서 `FastModel`과 `use_gradient_checkpointing="unsloth"`를 사용하며 원본 processor 외 토크나이저를 인코딩에 쓰지 않는다. trainable count, decoder LoRA 범위, 설정, vocab, 4bit 상태가 달라지면 중단한다. 동일 행 반복이므로 이 workload의 샘플별 loss 가중치는 같다. 이후 서로 다른 행을 섞을 경우 HF의 토큰 가중 평균과 비교 검증해야 한다.

GPU 실행은 `/tmp/soomac-r5-gpu.lock`을 독점 획득하고 다른 compute process가 있으면 시작하지 않는다. 프로세스 요청 상한은 32,000,000,000바이트, allocator 외 reserve는 512MiB이다. `RESULT.json`은 감시 종료와 마지막 강제 예산 확인이 성공한 뒤 시간, samples/s, nonpad·padded tokens/s, allocator peak, NVML 관측 process peak를 분리한다. trace의 dtype 표기는 `input_ids`와 요청한 BF16 autocast일 뿐 residual 텐서의 실제 dtype 검증이 아니다. attention 이름을 찾지 못하면 `INCONCLUSIVE`로 표시하고 trace를 직접 조사한다. `--check-loss`는 **별도 프로세스**에서 Unsloth의 `UNSLOTH_RETURN_LOGITS=1`을 import 전에 설정하고, 동일 한 행의 선택된 logits에 대한 native cross entropy와 model loss를 비교한다. logits가 없거나 shape가 맞지 않으면 `UNVERIFIED`로 실패한다. 이 진단은 timed update를 실행하지 않으며, 일반 타이밍 실행은 logits 반환을 끈다. **GPU 진단·품질 동등성 결과는 아직 없다.** 저장·재개 호환성도 미검증이다. 단일 smoke 결과로 품질 통과나 장기 ETA를 주장하지 않는다.

## 용어 정리

- **사전 인코딩(pre-encoding)**: 두 엔진에 같은 `input_ids`·`labels`를 공급하는 CPU 단계이다. `(내가 의역해보자면 토큰화 차이를 비교에서 제거하는 단계이다.)`
- **계측 밖 trace(profiler trace)**: timed update와 별도로 attention 연산자를 확인하는 기록이다. profiler 자체 메모리 증가량은 기본 peak와 구분한다.
