# Decision Unsloth 학습 TAR 명세서 (t1)

작성: 2026-11-02 KST. 대상: 원격 학습 서버(`http://100.112.9.90:8787`, spec 2.9)에 이 TAR를 올리고 실행하는 사람.
근거 파일: [Dockerfile](../tools/unsloth_remote_golden/Dockerfile), [Dockerfile.smoke](../tools/unsloth_remote_golden/Dockerfile.smoke), [Dockerfile.final](../tools/unsloth_remote_golden/Dockerfile.final), [run_training.sh](../tools/unsloth_remote_golden/run_training.sh), [production_shared_slot.py](../tools/unsloth_remote_golden/production_shared_slot.py), [lock/](../tools/unsloth_remote_golden/lock/).

## 1. 파일과 이미지

| 항목 | 값 |
|---|---|
| TAR | `/home/roma/Downloads/soomac-decision-unsloth-golden-train-20261102-t1.tar` |
| TAR 크기 / SHA256 | 3,495,539,200 B / `1a449c3adf2fe023589f8d96eb4710ea33f7cc467d544c380bc4654466cd0ef1` |
| 이미지 태그 | `soomac/decision-unsloth:golden-train-20261102-t1` |
| 이미지 ID | `sha256:4e7279e7cd23288d4040fa6b5da8626aaa0996dd6be4ea7cb785c059483e804e` |
| 계층 | c3(12층) → s2 smoke(17층) → t1(18층). t1은 s2에 GPU 검증 영수증 1층만 더한 것 |
| 검증된 원본(s2) | `sha256:ab3bf16e9790d5b6afdf90d39818cc18df7442ed6202df89b53a19be63abdaa1` |
| ENTRYPOINT / CMD / USER | `/app/run_training.sh` / `train` / `1000:1000` |
| 라벨 | `lab.training.spec=2.9`, `lab.training.resume=output-dir-v1`, `soomac.candidate.status=GPU_VALIDATED_RECEIPT_ATTACHED`, `soomac.tested_candidate=<s2 ID>` |

## 2. 실행 환경과 라이브러리 버전

로컬 Response 학습 환경(golden: r5 venv + `gemma4_env`)을 바이트 단위로 재현했다. 패키지는 resolver 없이 hash 고정 설치했고, 빌드 중 `golden_record.py verify`로 golden 설치본과 대조했다.

**기반**

| 항목 | 값 |
|---|---|
| OS 이미지 | `ubuntu@sha256:076e7bf8…5e11` (22.04 계열, libc6 2.35-0ubuntu3.15) |
| C 컴파일러 (Triton 런타임 JIT용) | gcc 11.4.0-1ubuntu1~22.04.3 |
| Python | conda `python-3.10.20-h17756b0_1` (`/opt/soomac/py`, conda 패키지 62개) |
| conda 기반 라이브러리 | libstdcxx 15.2.0, openssl 3.5.7 |

**학습 핵심 (빌드 중 버전 단언으로 고정)**

| 패키지 | 버전 |
|---|---|
| torch | 2.12.1+cu130 |
| triton | 3.7.1 |
| transformers | 5.13.1 |
| peft | 0.19.1 |
| trl | 1.8.0 |
| accelerate | 1.14.0 |
| bitsandbytes | 0.49.2 |
| unsloth / unsloth_zoo | 2026.10.11 / 2026.10.11 |
| safetensors | 0.8.0 |
| xformers | 0.0.35 (설치돼 있으나 로드 실패 → PyTorch attention 사용. golden과 같은 상태) |

**CUDA 사용자 공간 (pip wheel)**

| 패키지 | 버전 |
|---|---|
| cuda-toolkit | 13.0.2 |
| nvidia-cuda-runtime | 13.0.96 |
| nvidia-cublas | 13.1.1.3 |
| nvidia-cudnn-cu13 | 9.20.0.48 |
| nvidia-nccl-cu13 | 2.29.7 |
| torchao / torchvision / torchaudio | 0.17.0+cu130 / 0.27.1+cu130 / 2.11.0+cu130 |

**그 밖의 주요 패키지:** tokenizers 0.22.2, datasets 4.8.5, huggingface_hub 1.33.0, cut-cross-entropy 25.1.1, numpy 2.2.6, sentencepiece 0.2.1.

전체 목록:
- pip 134줄: [requirements-golden.lock](../tools/unsloth_remote_golden/lock/requirements-golden.lock)
- conda: [conda-explicit.txt](../tools/unsloth_remote_golden/lock/conda-explicit.txt), 내려받을 URL: [conda-urls.txt](../tools/unsloth_remote_golden/lock/conda-urls.txt)
- apt: [apt-runtime-versions.txt](../tools/unsloth_remote_golden/lock/apt-runtime-versions.txt)

## 3. 호스트 요구 조건

| 항목 | 조건 | 실측 (원격) |
|---|---|---|
| NVIDIA driver | ≥ 580.65.06 (CUDA 13.0 런타임 최소). 버전을 읽을 수 없거나 낮으면 exit 69로 시작을 거부 | 595.99.02, PASS |
| GPU | CUDA 13 지원 GPU | RTX PRO 6000 Blackwell Max-Q (sm_120) |
| 모델 마운트 | `/model` (서버가 제공하는 Gemma 4 base). 이미지에 없음 | 서버 기본값 |
| 데이터셋 | 이미지 안 `/app/assets/dataset` (MANIFEST SHA `78509482…`). 웹에서 데이터셋을 따로 고르지 않음 | — |
| 출력 | `/checkpoints` (쓰기 가능해야 함, 아니면 exit 73) | — |

## 4. 웹 제출 설정

RAM·CPU·shm·GPU 모드는 모두 **웹 제출 화면에서 고른다.** 이미지와 학습 코드에는 RAM 상한이 없다(`run_training.sh`와 production에 RAM 관련 제한 코드가 없음).

| 웹 항목 | 권장값 | 근거 |
|---|---|---|
| GPU 모드 | 공유(shared) 1슬롯 | 서버 슬롯 예산 50,246,713,344 B. 학습이 실제로 쓰는 GPU 메모리는 약 13.7 GB(smoke 실측 peak) |
| RAM | **32g 이상** | 8g: smoke resume 작업이 step 6에서 OOM-kill(exit 137). 32g: smoke 통과, 현재 학습도 32g에서 정상 진행 중. 로컬 측정 process peak RSS는 약 25.9 GB |
| CPU | 8 | 현재 학습 설정. 속도 영향은 측정하지 않음 |
| shm | 4g | 현재 학습 설정 |
| 이어하기 | 실패/취소된 작업 상세 → "재개 설정" → 새 작업으로 재개 | 서버가 `LAB_RESUME=1`, `RESUME_CHECKPOINT_DIR`, `RESUME_STEP`을 넣는다 |

RAM은 웹에서 32g보다 크게(서버 1작업 최대 48 GiB) 줘도 된다. 이미지를 다시 만들 필요가 없다. 32g 미만은 검증되지 않았다.

**⚠ 이미지 안의 "32GB"는 RAM이 아니라 GPU 메모리 예산이다.**
- 위치: `ENV SOOMAC_REMOTE_REQUESTED_BUDGET_BYTES=32000000000`
- 적용 방식: 실제 상한은 min(32e9, GPU 전체, 서버 슬롯 예산, 전체 × 서버 비율)이다. 이를 넘으면 학습이 스스로 멈춘다.
  - 앞의 두 값은 `run_training.sh`가 계산한다(68-74줄).
  - 나머지는 이미지 안의 sealed `common/budget.py`(11-22줄, 50-59줄)가 적용한다.
- smoke는 이 32e9로 검증됐다(영수증 생성 관문 `gpu_budget_is_receipt_budget`).
- ⚠ 학습 시작 때는 영수증 값이 상수 32e9인지만 본다(`production_shared_slot.py` 314-315줄). 실행 ENV 값과 영수증을 대조하지는 않는다.
  - 그래서 ENV를 바꿔도 시작은 막히지 않는다.
  - 하지만 검증되지 않은 설정이 되므로 바꾸지 않는다. 바꿔야 하면 새 이미지를 만들고 smoke를 다시 돌린다.

## 5. 학습 설정 (이미지에 고정, 웹에서 바꿀 수 없음)

| 항목 | 값 |
|---|---|
| 데이터 | train 4,360 / validation 309, max_length 9,216 |
| 에폭 / step | 6 에폭, 1 에폭 = 273 step, 총 1,638 step |
| 배치 | batch 1 × accumulation 16 = 16 |
| 최적화 | paged_adamw_8bit, lr 1e-4, cosine, warmup 3%(50 step), weight decay 0.01, max_grad_norm 1.0 |
| LoRA | r 16, alpha 32, dropout 0.05, text decoder 전용 |
| 양자화 / 정밀도 | NF4 double quant, BF16 연산, TF32 허용 |
| gradient checkpointing | Unsloth 방식 |
| 평가·저장 | 50 step마다 validation과 checkpoint |
| seed | 42 (data_seed 42) |

## 6. 시작 관문 (하나라도 실패하면 학습을 시작하지 않음)

1. 서버가 `GPU_MEMORY_BUDGET_BYTES`, `GPU_MEMORY_FRACTION`을 줬는지 확인한다. 없으면 exit 64.
2. driver ≥ 580.65.06인지 확인한다. 아니면 exit 69.
3. `/app/gpu-validation/VALIDATION_RECEIPT.json`이 PASS인지 확인한다. 함께 확인하는 값: 영수증 안 이미지 ID의 형식, GPU 예산 32e9, peak ≤ 예산.
   - 영수증과 s2 이미지 ID의 결합은 **빌드 때** 확인했다(`build_final_golden.py`, `Dockerfile.final`).
   - 실행 시점에는 현재 이미지 ID를 다시 대조하지 않는다.
4. 영수증 identity와 현재 실행 값이 같은지 확인한다(`production_shared_slot.py` 294-310줄). 대상: release, 데이터 manifest, 모델 설정, tokenizer, production 코드, model builder, 패키지 버전.
5. GPU 잠금을 건다. 공유 슬롯(`GPU_MODE=shared` + 서버 예산)에서는 다른 컨테이너의 GPU 프로세스를 허용한다.

## 7. 검증 근거 (2026-11-02 원격 smoke)

| 항목 | 결과 |
|---|---|
| 의도적 실패 후 이어하기 | step 5 저장 → 의도적 실패 → step 10까지 완료. 복원 4/4 일치 |
| loss | native CE 재계산과 차이 0 (4.0373) |
| gradient / update | LoRA B 328/328 nonzero, step 5→10 사이 텐서 656개 변경 |
| GPU peak | 13,735,297,024 B (예산 32e9 이내) |
| 생성 코드 | golden과 `compile_threads`(32→24, CPU 수 차이)만 다름. SHA `01cb73dd…` |
| golden 수치 동등성 | **UNKNOWN** (원격에서 golden과 같은 입력으로 대조하지 않음) |

## 8. 실측 속도 (원격 RTX PRO 6000, Decision 1,638 step 학습)

step당 토큰 수는 두 엔진이 같다(약 129,700~130,000 tok/step).

| | 혼자 사용 | 공유 중 |
|---|---|---|
| HF build05 | 약 176초/step, 약 738 tok/s (step 0→222) | 약 371초/step (step 222→263) |
| Unsloth t1 | 약 128초/step, 약 1,012 tok/s (step 62→70, **8 step뿐인 중간값**) | 약 278초/step, 약 466 tok/s (step 5→62) |

- 전체 1,638 step 환산(혼자): HF 약 80시간, Unsloth 약 58시간 (중간값 기준, 추정)
- Unsloth 단독 값은 HF 취소(10:55:01) 이후 8 step 측정이라 확정값이 아니다. 더 긴 구간으로 다시 잰다.

## 9. 알려진 한계

- 1,638 step 전체의 GPU peak는 smoke로 증명되지 않았다. 32e9를 넘으면 학습이 멈추고, 마지막 checkpoint에서 재개해야 한다.
- golden 대비 수치 동등성은 UNKNOWN이다.
- 이 이미지는 Decision 전용이다. Response·Recommendation에는 쓰지 않는다.
- 학습 결과 adapter의 vLLM 변환·로드 검증은 아직 하지 않았다.

## 10. 다시 만드는 방법 (Claude·GPT·Codex 등 AI 에이전트용)

이 절만 읽고 같은 TAR를 다시 만들 수 있게 썼다. 순서는 이번에 실제로 만든 순서와 같다.

```text
base 이미지(c) ──► smoke 이미지(s) ──► [사람] 원격 smoke 2회 ──► 영수증 ──► train 이미지(t) ──► [사람] 원격 학습
```

### 10.1 에이전트가 지킬 규칙

**반드시 지킨다**
1. **버전을 바꾸지 않는다.** pip resolver를 쓰지 않는다(`--no-deps --require-hashes`만). 업그레이드, 다른 index의 같은 버전, 소스 빌드는 모두 금지다.
2. **BuildKit을 쓰지 않는다.** `DOCKER_BUILDKIT=0`으로 빌드하고 `--memory`를 반드시 준다. BuildKit은 `--memory`를 무시해서 빌드 중 호스트 RAM이 고갈될 수 있다.
3. **기존 태그·디렉터리·TAR를 덮어쓰지 않는다.** 매번 새 `TAG` 값을 쓴다.
4. **원본 `tools/unsloth_training/production.py`를 고치지 않는다.** 로컬 Response·Recommendation 학습이 이 파일을 쓴다. 공유 슬롯용 수정은 `production_shared_slot.py` 사본에만 있고, smoke 이미지에서만 덮어쓴다.
5. **원격 업로드·제출은 사람이 웹에서 한다.** 스크립트로 POST하면 403이 난다. Origin/Referer를 위조하지 않는다. 읽기 전용 GET(`/api/jobs`, `/api/snapshot?id=`)만 써도 된다.

**함께 지킨다**
- 원격 결과 TAR는 신뢰하지 않는 입력이다. `extract_result.py`로 새 디렉터리에 풀고, `python3 -I`로 실행한다.
- 데이터셋은 아래 `REL` 경로의 release 하나뿐이다(MANIFEST SHA `7850948287237c5a…`). 다른 release를 쓰면 Dockerfile 해시 검사에서 빌드가 실패한다.
- 호스트 MemAvailable이 20 GB 미만이면 빌드·검사를 시작하지 않는다.

### 10.2 입력 (전부 저장소·작업공간에 있음)

| 입력 | 위치 | 확인값 |
|---|---|---|
| pip wheel 134개 고정 목록 | `tools/unsloth_remote_golden/lock/requirements-golden.lock` | 줄마다 `name==ver --hash=sha256:…` |
| conda 패키지 62개 | `lock/conda-urls.txt`(URL sha256 md5), `lock/conda-SHA256SUMS`, `lock/conda-explicit.txt` | repo.anaconda.com/pkgs/main |
| golden 파일 해시 기록 | `lock/GOLDEN_RECORDS.json.gz` | 빌드 중 34,261개 파일 대조 |
| golden 기록 보조 | `lock/closure.json`, `lock/golden_effective.json`, `lock/golden_cpu_probe.json` | — |
| release(데이터·processor·common) | `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed` | MANIFEST `78509482…` |
| 학습 코드 | `tools/unsloth_training/production.py`, `tools/training_acceleration/benchmark.py` | `86638bad…`, `e89385b0…` |
| 진입·검증 스크립트 | `tools/unsloth_remote_golden/` 의 `run_training.sh`, `golden_record.py`, `smoke_entry.sh`, `gpu_validate.py`, `production_shared_slot.py`(`962fc3d5…`), `make_receipt.py`, `build_final_golden.py`, `extract_result.py` | — |
| 기반 이미지 | Dockerfile 안에 digest로 고정(micromamba, ubuntu) | 빌드 때 자동 pull |

2026-11-02에 검증한 내용:
- 위 lock 파일로 해시 고정 다운로드를 하면 c3 빌드 컨텍스트와 같은 wheel 134개가 나온다.
- conda·GOLDEN_RECORDS·학습 코드·release도 c3 빌드 컨텍스트와 바이트 단위로 같다.

### 10.3 실행 방식과 공통 변수

**실행 방식.** 아래 블록은 각각 새 셸에서 아래처럼 실행한다. 명령이 하나라도 실패하면 그 자리에서 멈춘다. 멈추면 다음 블록으로 가지 않는다.

```bash
bash -euo pipefail -c "$(cat vars.sh block.sh)"
```

- `vars.sh`: 이 절의 변수 블록을 저장한 파일
- `block.sh`: 실행할 블록을 저장한 파일

**변수 블록** (`vars.sh`, 대입만 있음)

```bash
REPO=/home/roma/IRC_SOOMAC3.0
G=$REPO/tools/unsloth_remote_golden
L=$G/lock
REL=/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed
RUNS=/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs
PY=$REPO/training/learning_first_v43/training_release_20261101_r5/env/bin/python   # 다운로드용. pip 있는 python3면 무엇이든 됨
TAG=20261102-x1                                   # 매번 새 값
CPUS=24-31                                        # 빌드에 쓸 CPU. 이 PC(32코어) 기준
W=/tmp/soomac-golden-wheels-$TAG                  # wheel 내려받는 곳
C=/tmp/soomac-golden-conda-$TAG                   # conda 내려받는 곳
CTX=/tmp/soomac-golden-context-$TAG               # base 이미지 빌드 컨텍스트
SC=/tmp/soomac-golden-smoke-context-$TAG          # smoke 이미지 빌드 컨텍스트
TC=/tmp/soomac-golden-train-context-$TAG          # train 이미지 빌드 컨텍스트
X=$RUNS/unsloth_golden_smoke_$TAG                 # 원격 smoke 결과와 영수증
BASE_TAG=soomac/decision-unsloth:golden-candidate-$TAG
SMOKE_TAG=soomac/decision-unsloth:golden-smoke-$TAG
TRAIN_TAG=soomac/decision-unsloth:golden-train-$TAG
SMOKE_TAR=/home/roma/Downloads/soomac-decision-unsloth-golden-smoke-$TAG.tar
TRAIN_TAR=/home/roma/Downloads/soomac-decision-unsloth-golden-train-$TAG.tar
```

**시작 전 검사** (처음 한 번만 실행)

```bash
for p in $W $C $CTX $SC $TC $X $SMOKE_TAR $TRAIN_TAR; do
    if [ -e "$p" ]; then echo "중단: 이미 있음 $p (TAG를 바꿀 것)" >&2; exit 1; fi
done
for t in $BASE_TAG $SMOKE_TAG $TRAIN_TAG; do
    if docker image inspect "$t" >/dev/null 2>&1; then echo "중단: 태그 있음 $t (TAG를 바꿀 것)" >&2; exit 1; fi
done
avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if [ "$avail_kb" -lt 20971520 ]; then echo "중단: MemAvailable < 20GB" >&2; exit 1; fi
echo "시작 전 검사 OK"
```

### 10.4 wheel 134개 (약 30분, 수 GB)

```bash
mkdir -p $W/pypi-xformers $CTX/wheels
grep -v '^xformers==' $L/requirements-golden.lock | sed 's/ --hash=.*//' > $W/req-no-xformers.txt
$PY -m pip download --no-deps --only-binary=:all: -r $W/req-no-xformers.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 -d $W
$PY -m pip download --no-deps --only-binary=:all: xformers==0.0.35 \
    --index-url https://pypi.org/simple -d $W/pypi-xformers
# 해시 관문: lock의 hash와 다른 파일이 하나라도 있으면 여기서 실패한다
$PY -m pip download --no-deps --only-binary=:all: --require-hashes --no-index \
    --find-links $W --find-links $W/pypi-xformers -r $L/requirements-golden.lock -d $CTX/wheels
n=$(ls $CTX/wheels | wc -l)
if [ "$n" -ne 134 ]; then echo "중단: wheel $n개 (134개여야 함)" >&2; exit 1; fi
echo "wheel 134 OK"
```

- 원인: xformers 0.0.35는 cu130 index에도 같은 버전이 있다. 하지만 golden 설치본은 **PyPI 빌드**다.
- 결과: xformers만 PyPI에서 따로 받는다. 잘못 받아도 해시 관문에서 멈춘다.

### 10.5 conda 62개와 나머지 컨텍스트 (약 5분)

```bash
mkdir -p $C $CTX/conda $CTX/golden $CTX/package $CTX/source/unsloth_training $CTX/source/training_acceleration $CTX/bundle
while read -r url sha md5; do curl -fsSL -o "$C/$(basename "$url")" "$url"; done < $L/conda-urls.txt
cp $C/*.conda $CTX/conda/
cp $L/conda-SHA256SUMS $CTX/conda/SHA256SUMS
cp $L/conda-explicit.txt $CTX/conda/explicit.txt
(cd $CTX/conda && sha256sum -c --quiet SHA256SUMS)
cp $L/requirements-golden.lock $L/closure.json $L/golden_effective.json $L/golden_cpu_probe.json $CTX/golden/
gunzip -c $L/GOLDEN_RECORDS.json.gz > $CTX/golden/GOLDEN_RECORDS.json
cp $G/run_training.sh $G/golden_record.py $CTX/package/
cp $REPO/tools/unsloth_remote/test_processor_encoding.py $CTX/package/
cp $REPO/tools/unsloth_training/production.py $CTX/source/unsloth_training/
cp $REPO/tools/training_acceleration/benchmark.py $CTX/source/training_acceleration/
cp -a $REL/. $CTX/bundle/
if [ -n "$(find $CTX/bundle -type l -print -quit)" ]; then echo "중단: bundle에 링크 있음" >&2; exit 1; fi
cp $G/Dockerfile $CTX/Dockerfile
sha256sum -c --quiet - <<SUMS
86638bad3614a633b6fa023694f5c7b37ca840e9e37177be0eeff9d6a27e3208  $CTX/source/unsloth_training/production.py
e89385b0e4257d3c0e7b7947a4a8c1bb7e67e8ce2e27873efd2afac8ed45394c  $CTX/source/training_acceleration/benchmark.py
c1df5b92d6a85a9c31c472af486863ced25687c6c8195437fabd3e45976c1ace  $CTX/package/test_processor_encoding.py
SUMS
echo "컨텍스트 OK"
```

- 원인: base Dockerfile은 학습 코드 SHA를 확인하지 않는다. smoke Dockerfile에서야 확인한다.
- 결과: 여기서 전체 SHA를 대조한다. 코드가 바뀌었으면 이 블록에서 멈춘다.

### 10.6 base 이미지 빌드 (약 20~40분)

```bash
LAB_INPUTS="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])),separators=(",",":")))' $CTX/bundle/remote/lab-inputs.json)"
DOCKER_BUILDKIT=0 nice -n 15 docker build --platform linux/amd64 --memory 12g --memory-swap 12g \
    --cpuset-cpus $CPUS --cpu-shares 256 --build-arg "LAB_INPUTS=$LAB_INPUTS" \
    -f $CTX/Dockerfile -t $BASE_TAG $CTX
docker image inspect $BASE_TAG --format '{{.Id}}'
```

- `CPUS`(10.3)는 이 PC 기준이다. 다른 PC에서는 실제 있는 코어 범위로 바꾼다.
- 빌드 안에서 자동으로 확인하는 것:
  - conda sha256, Python 3.10.20, conda 패키지 62개
  - `pip check`, golden 파일 34,261개 바이트 대조, 핵심 11개 버전 단언
  - 데이터 MANIFEST와 processor 해시, production `--help` 플래그
- 하나라도 틀리면 빌드가 실패한다. **실패를 우회하려고 Dockerfile 검사를 지우지 않는다.**

### 10.7 smoke 이미지와 TAR (약 5분)

```bash
BASE_ID=$(docker image inspect $BASE_TAG --format '{{.Id}}')
mkdir -p $SC/package
cp $G/run_training.sh $G/smoke_entry.sh $G/gpu_validate.py $G/production_shared_slot.py $SC/package/
cp $G/Dockerfile.smoke $SC/
sed -i "s|soomac.smoke.base_c3=\"sha256:[0-9a-f]*\"|soomac.smoke.base_c3=\"$BASE_ID\"|" $SC/Dockerfile.smoke
grep -q "soomac.smoke.base_c3=\"$BASE_ID\"" $SC/Dockerfile.smoke
cd $SC
DOCKER_BUILDKIT=0 nice -n 10 docker build --memory 4g --memory-swap 4g \
    --cpuset-cpus $CPUS --cpu-shares 256 -f Dockerfile.smoke \
    --build-arg C3_IMAGE=$BASE_ID -t $SMOKE_TAG .
if [ -e $SMOKE_TAR ]; then echo "중단: TAR 있음 $SMOKE_TAR" >&2; exit 1; fi
docker save -o $SMOKE_TAR $SMOKE_TAG
sha256sum $SMOKE_TAR
docker image inspect $SMOKE_TAG --format 'SMOKE_ID={{.Id}}'
```

- `sed` 줄은 사본(`$SC`)의 정보용 라벨만 새 base ID로 바꾼다. 저장소의 `Dockerfile.smoke`는 그대로 둔다.
- 출력된 TAR SHA256과 `SMOKE_ID`를 기록해 둔다. 웹 업로드 후 서버가 보여주는 SHA와 대조한다.

### 10.8 원격 smoke 2회 — 사람이 웹에서 함

1. 웹에서 smoke TAR를 업로드한다. 설정: **GPU 공유 1슬롯, RAM 32g, CPU 8**.
2. 1차 작업은 **실패로 끝나는 것이 정상이다.** 로그에 아래 3줄이 모두 있어야 한다.
   - `SOOMAC_DRIVER_CHECK status=PASS`
   - `intentional smoke failure after committed step 5`
   - `SMOKE_PHASE1 EXPECTED_FAILURE`
3. 1차 작업 상세에서 "재개 설정 → 새 작업으로 재개"를 누른다. **RAM은 다시 32g로 고른다.**
   - 이번에 8g로 재개했다가 step 6에서 OOM이 났다.
4. 2차 작업은 성공(exit 0)으로 끝나야 한다. 로그에 아래 3줄이 모두 있어야 한다.
   - `SOOMAC_DRIVER_CHECK status=PASS`
   - `SMOKE_PHASE2 resumed smoke reached step 10`
   - `"event": "SOOMAC_GPU_VALIDATION", "pass": true`
5. 두 작업의 결과 TAR(`<jobid>-result.tar`)를 내려받는다.

위 문자열은 `make_receipt.py`(105-111줄)가 검사하는 문자열과 같다. 하나라도 없으면 10.9에서 영수증이 거부된다.

### 10.9 결과 추출과 영수증 (약 5분)

`JOB1_TAR`, `JOB2_TAR`는 실제 파일명으로 바꾼다.

```bash
JOB1_TAR=/home/roma/Downloads/1차jobid-result.tar
JOB2_TAR=/home/roma/Downloads/2차jobid-result.tar
SMOKE_ID=$(docker image inspect $SMOKE_TAG --format '{{.Id}}')
mkdir $X
python3 -I $G/extract_result.py $JOB1_TAR $X/job1
python3 -I $G/extract_result.py $JOB2_TAR $X/job2
python3 -I $G/make_receipt.py --job1 $X/job1/results/checkpoints --job2 $X/job2/results/checkpoints \
    --job1-results $X/job1/results --job2-results $X/job2/results \
    --tested-image $SMOKE_ID --out $X/receipt/gpu-validation
```

- `mkdir $X`는 디렉터리가 이미 있으면 실패한다. 예전 결과가 섞이지 않게 하기 위해서다. `extract_result.py`도 새 디렉터리에만 쓴다.
- `make_receipt.py`는 관문 약 30개를 모두 통과해야 `{"status": "PASS"}`를 출력하고 파일을 만든다. 하나라도 실패하면 아무것도 쓰지 않는다.
  - 대상: 이미지 ID, exit/OOM, step 5 의도적 실패, 복원, loss 재계산, gradient, GPU 예산 등
- 실패하면 **다음 단계로 가지 않는다.** 출력된 실패 관문 이름으로 원인을 찾는다.
- `compile_threads` 값이 golden(32)과 다른 것은 정상이다. 원격 CPU 수 때문이며, 영수증 관문이 이 차이만 허용한다.

### 10.10 train 이미지와 TAR (약 5분)

```bash
SMOKE_ID=$(docker image inspect $SMOKE_TAG --format '{{.Id}}')
python3 -I $G/build_final_golden.py --tested-tag $SMOKE_TAG --tested-image $SMOKE_ID \
    --receipt-dir $X/receipt/gpu-validation --context $TC --final-tag $TRAIN_TAG --cpuset-cpus $CPUS
if [ -e $TRAIN_TAR ]; then echo "중단: TAR 있음 $TRAIN_TAR" >&2; exit 1; fi
docker save -o $TRAIN_TAR $TRAIN_TAG
sha256sum $TRAIN_TAR
```

- `build_final_golden.py`가 빌드 전후로 확인하는 것:
  - smoke 이미지 ID와 영수증 결합
  - 계층이 smoke 이미지를 그대로 잇는지
  - ENTRYPOINT/CMD/USER
- 결과물을 웹에 올리고 4절 설정(공유 1슬롯, RAM 32g 이상)으로 학습을 시작한다.

### 10.11 이번에 실제로 막혔던 곳

| 증상 | 원인 | 대응 |
|---|---|---|
| 빌드 중 `--memory`가 안 먹음 | BuildKit은 `--memory`를 무시 | `DOCKER_BUILDKIT=0` |
| golden 파일 대조에서 xformers 불일치 | cu130 index의 xformers를 받음 | PyPI 빌드 사용(10.4) |
| CPU 전용 컨테이너에서 torch import 시 `KeyError` | UID 1000의 passwd 항목 없음 | Dockerfile이 사용자 `soomac`을 만든다(수정 완료) |
| 재개 작업 OOM (exit 137) | RAM 8g | RAM 32g |
| 생성 코드 SHA가 golden과 다름 | `compile_threads` = CPU 수(로컬 32, 원격 24) | 영수증이 이 한 값만 허용해 정규화 |
| `build_final`이 Dockerfile을 못 찾음 | 상대 경로 `-f` | 절대 경로로 수정 완료 |
| 스크립트 업로드 403 | 서버가 웹 페이지 요청만 허용 | 사람이 웹으로 업로드 |
| CPU 시험에서 학습이 GPU 잠금 단계에서 멈춤 | GPU가 없음 | 정상. CPU 시험은 거기까지만 확인한다 |
