# Gemma-4 Unified 12B attention ablation — Claude 담당 계획

## 2026-11-01 최신 사용자 범위 — root 실제 실행

목표는 VRAM 절약이 아닌 optimizer update 시간 단축이다. 기존 `attention_ablation.py`와 encoded Response8,430토큰을 재사용한다. GC OFF/부분 GC/batch2/native GC는 다시 실행하지 않는다. 원본 production/checkpoint/데이터/원격 Decision을 변경하지 않는다.

1. 현재 smart GC·native CE·batch1의 **실제 CUDA 프로파일**을 먼저 기록한다. Global8층(head512)과 sliding40층(head256)을 실제 attention operator query shape 및 연결된 CUDA kernel로 구분한다. 카운터0이나 FA2/Xformers 문자열만으로 미사용을 확정하지 않는다.
2. 설치된 sliding router는 `unsloth_zoo/temporary_patches/gemma4_flash_sliding.py`에 있다. `gemma4_banded_attention.py:85`는 compile 중 mask 검사를 거절하고, padding 없는 순수 band만 허용한다. HD512 router도 compiled masked call을 SDPA로 보낸다. 실제 입력 mask에 해당하는지 확인한 뒤 판정한다.
3. 정상 설정 후보는 fresh process의 공식 `UNSLOTH_COMPILE_DISABLE=1` 경로 하나로 제한한다. rewritten forward는 유지하며 causal/window/padding/softcap/dtype/LoRA dropout0.05는 바꾸지 않는다. 실패/정확성 불일치면 후보를 중단하고 새로운 kernel 개발·무한 설정 탐색을 하지 않는다.
4. 실제 다른 CUDA 경로가 확인될 때만 동일16개 입력·warmup 이후 반복 optimizer update를 비교한다. profiler는 시간 밖에 둔다. **기존 compare의 Python counter 기반 timing gate는 compiled kernel 판정으로 쓰지 않는다.** 실제 활성화가 안 되면 B의 시간/배수를 만들지 않는다.

진행 원본: `/tmp/soomac-training-accel-20261101/attention_current_a1/`. 첫 native profiler 결과는 후속 intrusive probe가 실패하더라도 별도 `NATIVE_PROFILE.json`으로 보존한다. layer/Flex probe는 production 실행 증거가 아니며 이번 bounded check에서는 Flex를 생략한다. 아래 구버전 계획의 'counter0이면 전용 kernel 미선택' 종료 기준은 폐기하고 CUDA proof를 우선한다.

Cause: compile된 경로는 Python 카운터를 누락할 수 있으며, 층별 kernel과 mask 의미를 함께 봐야 정확하다.
Effect: 카운터에 의존한 잘못된 미사용 판정 및 의미를 바꾼 속도 비교를 하지 않는다.

폐기 판단용. production 학습/checkpoint/데이터/weights/gemma4_env/STT/VLM/원격 Decision 미변경.
학습용 prefix 공유 보류. 이 세션은 GPU 미실행(root PID1019116 점유). CPU 준비·소스 검수·스크립트
작성만 수행. 토큰/.aws/원본 학습 행/가중치 미열람·미전송.

소유 산출물(신규 파일만):
- `tools/training_acceleration/attention_ablation.py`
- `tools/training_acceleration/test_attention_ablation.py`
- 본 문서

root 소유 `benchmark.py`/`test_benchmark.py`는 **읽기만** 했고 수정하지 않았다. 재사용은 import로만 한다.

---

## 1. 소스에서 확정한 사실 (일차 근거, 줄번호)

파일 해시는 실행 시 `reviewed_sources`로 기록된다(아래 줄번호가 drift하면 해시로 드러남).

### 1.1 transformers Gemma4 (`modeling_gemma4_unified.py`)
- 층 dispatch는 `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager)` 1곳(456줄).
  즉 `attn_implementation`이 모든 층의 attention 함수를 정한다. 층별로 다른 구현을 쓰지 않는다.
- global(full_attention) 8층: `head_dim = global_head_dim = 512`, `use_alternative_attention = attention_k_eq_v and not is_sliding` → KV head = `num_global_key_value_heads = 1`, V = `v_norm(raw k_proj)`(372~375줄).
- sliding 40층: `head_dim = 256`, KV head 8(372줄, config).
- `scaling = 1.0` 고정(378줄). 모든 경로가 scale=1.0을 받는다(q_norm이 스케일 흡수).
- 학습 시 `dropout = attention_dropout if training else 0` = **0.0**(466줄, config `attention_dropout=0.0`). LoRA dropout 0.05는 adapter용이며 attention dropout이 아니다.
- `_supports_flex_attn = True`(570줄). flex는 모델 차원에서 허용된다(실측은 별도).
- 텍스트 학습 forward는 `full_attention`=`create_causal_mask`, `sliding_attention`=`create_sliding_window_causal_mask` 두 마스크만 만든다(664~667줄). blockwise overlay가 섞인 복합 sliding 마스크(1239줄)는 **멀티모달 생성 경로 전용**이라 텍스트 Response 입력에는 쓰이지 않는다.

**config 확정값** (`/home/roma/Desktop/sLLM/gemma-4-12B-it/config.json`): 48층, full_attention index = (5,11,17,23,29,35,41,47) 8개, 나머지 40개 sliding(window 1024). head512/256, heads16, kv8, global_kv1, kv_shared0, k_eq_v true.

### 1.2 Unsloth head_dim-512 커널 (`kernels/large_head_attention.py`, PR #13869)
- `_SUPPORTED_HEAD_DIMS = (512,)`(27줄). 라우터 게이트 `query.shape[-1] > 256`(886줄).
  → **sliding 40층(head_dim 256)은 이 커널을 영구적으로 쓰지 않는다. global 8층(512)만 후보.**
- 라우터가 SDPA로 **되돌아가는** 조건(= 등록됐어도 미선택): dropout≠0(887), autocast dtype 불일치(899),
  math-first SDPA 우선순위(900), **compiled/capturing 중 mask≠None**(912), `_shapes_qualify` 불통과, self-check 실패(770).
- 실제 커널 실행 시 `LARGE_HEAD_ENGAGED["calls"]++`(937), 폴백 시 `["fallback"]++`(924). → **등록 vs 선택을 카운터로 실측.**
- 설치는 `ALL_ATTENTION_FUNCTIONS["sdpa"]`를 래핑(1022줄). `_unsloth_forced_float32 is True`면 설치 자체를 생략(985줄).
- on/off 스위치: `UNSLOTH_HD512_ATTENTION`(716줄, 기본 on).

### 1.3 Unsloth loader (`models/vision.py`)
- `do_forced_float32`는 `UNSLOTH_FORCE_FLOAT32=1`일 때만 True(3669줄). **기본 bf16+4bit 로드에선 False** → 1.2의 985줄 가드에 안 걸림 → 라우터 설치됨.
- `_explicit_flex`는 호출자가 `attn_implementation="flex_attention"`을 줄 때만(3760줄). root `model_and_optimizer`(benchmark.py 237~242)는 안 주므로 Unsloth가 자동 해결(설명상 xformers import 실패 → PyTorch SDPA로 fallback).
- `patch_large_head_attention(model)`은 `FastModel.from_pretrained` 끝에서 호출(4943줄).

**결론(소스 기준, 미실측)**: bf16+4bit Unsloth 로드에서 HD512 라우터는 **등록된다**. 그러나 global 8층에서
**실제 선택되는지**는 (a) 그 층에 전달되는 mask가 None인지 materialized인지, (b) forward가 compile/capture
중인지에 달려 있어 **런타임 카운터로만 확정**된다. 이것이 사용자 요구 2의 "등록 vs 선택" 지점이다.

---

## 2. 스크립트가 하는 일 (요구 1~5 대응)

`attention_ablation.py`는 benchmark.py를 import해 sealed loader(`model_and_optimizer`), encoded artifact
(`validate_payload`), GPU lock(`acquire_gpu_lock`), budget, `scored_suffix`, `one_update`를 **그대로 재사용**한다.
모델 그래프·LoRA(r16/α32/dropout0.05)·NF4 double/BF16·max_length·토큰 마스킹이 baseline과 동일해진다.
**측정은 두 층위로 분리한다**: (A) production 대표 경로 = ALL_ATTENTION_FUNCTIONS 미개입 `model.train()`
forward+backward를 torch.profiler로 찍는다, (B) 침습 probe = `ALL_ATTENTION_FUNCTIONS["sdpa"]` 래핑으로
층별 라벨·텐서 캡처(`finally` 복원). (B)에서 커널이 engage돼도 production 학습이 그 커널을 쓴다고 주장하지 않는다.
실험 전체는 `UNSLOTH_RETURN_LOGITS=1`(native loss)로 돌려 attention 변화와 fused-CE delta(fused .79728025 vs
native .79451740)를 혼동하지 않는다.

- **prepare** (CPU, GPU 없음): 환경 manifest(라이브러리 버전·소스 해시·검수 줄번호·CPU RSS) 기록. torch import 안 함.
  encoded artifact는 **기존 것을 재사용**한다(아래 4절). 새 prepare 전체 반복은 하지 않는다.
- **check** (GPU, timing 없음):
  - **(A) native_train_profile (production 대표)**: 침습 없이 `model.train()` 1 microbatch forward+backward를
    `record_shapes=True` profiler로 찍어 실제 attention CUDA kernel·입력 shape(실제 8k)·HD512 engaged/fallback
    카운터·resolved config를 기록. 실제 8k shape로 head_dim 512/256 kernel이 구분되므로 synthetic seq512 probe는
    제거했다(길이-독립이라는 근거 없는 주석도 삭제).
  - **(B) 침습 per-layer recorder (production 아님)**: 48층 각각 layer_type/head_dim/mask-None/KV head와 HD512
    engaged delta를 기록하고 global/sliding 각 1층의 q/k/v/mask를 캡처. `per_layer_attention_is_intrusive_probe=True`.
  - **층 probe(HD512 vs SDPA 수치)**: 캡처 텐서로 HD512 커널 출력 vs SDPA 출력 absolute/relative/cosine/nonfinite.
    bool mask는 그대로(True=keep) SDPA에 주고, float mask는 0/-inf bias, **mask=None은 causal dense interval로 비교**
    (더 이상 SKIPPED 아님). scale=1·GQA·pad mask 보존. 캡처 텐서는 detach clone이라 production peak와 별개로 표시.
  - **flex probe(layer-wise만)**: 캡처된 **실제 mask를 BlockMask로 반영**(새 causal/window 날조 금지)해 flex 출력+backward를
    SDPA와 비교. eager/compiled 둘 다 시도(공식 권장은 compiled). 전체 모델 flex는 별도 fresh-process native1
    `attn_implementation=flex_attention` 로드(40 sliding+8 global forward/backward)로만 확정 — 이 스크립트는 안 한다.
- **compare** (GPU): 모델 1회 로드.
  - **bounded 기본(수 분)**: optimizer.step 없이 off/on 각각 forward+backward(동일 seed42·동일 weights·dropout 0.05)로
    loss 합·LoRA grad 스냅샷 비교. `_set_hd512`는 `_ENABLED[0]`을 직접 초기화하고 `_enabled()` 값을 **되읽어 검증**(토글
    no-op이면 raise). off/on calls로 **토글이 실제로 커널을 바꿨는지** 판정.
  - **timing은 opt-in(`--run-timing`)이고 토글이 커널을 바꾼 경우에만** 실행. 바꾸지 않으면 `timing_skipped_reason`
    기록 후 생략(노이즈 측정 금지). 실행 시 off/on 각각을 **동일 initial weights/optimizer/RNG로 복원**한 뒤 warmup+timed로
    측정(ON이 OFF의 step 변화를 상속하지 않음). 325초(cold 1-update)와 배수 비교하지 않는다.

---

## 3. 실행한 것 / 아직 안 한 것

**한 것** (이 세션, CPU, 수정만):
- 소스 4개 read-only 재검수(위 줄번호). `_enabled`/`_ENABLED=[None]` 규약과 `intervals_from_mask`(bool True=keep,
  float 0=keep) 규약을 설치 소스로 직접 확인.
- GPU 검수 전 5개 오류를 국소 수정(아래). CPU 테스트 2개 추가(`_set_hd512` ON/OFF 토글, `--run-timing` 기본 off).

**GPU 검수 전 수정 5건**:
1. `_set_hd512`: lru 아님 → `_ENABLED[0]=None` 직접 초기화 + `_enabled()` 되읽어 검증(no-op이면 raise). 예외 삼킴 제거.
   compare `finally`에서 원래 env·`_ENABLED` 복원.
2. check: 침습 wrapper 전에 **무계측 `model.train()` forward+backward profiler**(native_train_profile)를 먼저 찍고
   production 대표로 표시. 침습 per-layer는 `is_intrusive_probe=True`로 분리. synthetic seq512 probe 제거 + 길이-독립 주석 삭제.
   실험 전체 `UNSLOTH_RETURN_LOGITS=1`로 변경.
3. 수치 probe: `zeros_like` bool mask(차단 제거) → bool은 그대로/float은 0-inf bias. mask=None은 causal interval로 비교(SKIPPED 해소).
4. flex probe: 날조 mask 제거 → 캡처 실제 mask를 BlockMask로. 출력+backward 비교. eager/compiled 구분. layer-wise 범위 명시.
5. compare: timing을 opt-in(`--run-timing`)·토글 변화 확인 후로 제한(기본은 bounded 수-분 probe). timing 시 off/on을
   동일 initial weights/optimizer/RNG로 복원.

**안 한 것** (실행 수단 없음 — 이 세션에 shell 도구 없음, 실행 불가):
- CPU 테스트 미실행(코드 수정만). root가 `"$VENV/bin/python" tools/training_acceleration/test_attention_ablation.py`로 확인.
- GPU check/compare 미실행(root가 lock 확인 후 실행).
- Codex 교차검증 미실행(shell 없음). 수치 결론/공식 경로는 root의 `gpt-6-sol/high` read-only 검수
  (`/tmp/soomac_claude_attention_harness_codex_review.out`)와 실제 GPU 이후 보고한다.

---

## 4. root가 실행할 명령

전제: root가 GPU lock(PID1019116 smoke 종료, `/tmp/soomac-r5-gpu.lock` 해제) 확인. 메모리 상한은
`systemd-run --scope -p MemoryMax=…` + `MAX_JOBS=2`로 감싼다(FlashInfer/Triton 컴파일 OOM 방지).

```bash
cd /home/roma/IRC_SOOMAC3.0
VENV=/tmp/soomac-training-accel-20261101/venv
OUT=/tmp/soomac-training-accel-20261101/claude_attention_$(date +%H%M%S)
# 기존 encoded artifact 재사용 (새 prepare 전체 반복 금지)
ENCODED=/tmp/soomac-benchmark-p-cpu-prep-20261101-a1/encoded.json

# 0) CPU 테스트 (GPU 불필요)
"$VENV/bin/python" tools/training_acceleration/test_attention_ablation.py

# 1) native profile + 층별 경로/kernel/수치/flex probe (GPU, timing 없음)
#    prepare는 안 돈다(encoded 재생성 = prepare 반복). reviewed_sources·library_versions는
#    CHECK.json/COMPARE.json 안에 이미 함께 기록된다.
systemd-run --scope -p MemoryMax=30G --setenv=MAX_JOBS=2 \
  "$VENV/bin/python" tools/training_acceleration/attention_ablation.py check \
  --agent response --engine unsloth \
  --encoded "$ENCODED" --output "$OUT/check"
```

**compare — 기본은 bounded(수 분)**: off/on loss·LoRA grad 동치 + 토글이 커널을 바꿨는지(off/on calls)만 확인.
timing은 돌리지 않는다. 이걸로 "토글이 실제 다른 커널인가"를 먼저 본 뒤에만 timing으로 넘어간다.

```bash
systemd-run --scope -p MemoryMax=30G --setenv=MAX_JOBS=2 \
  "$VENV/bin/python" tools/training_acceleration/attention_ablation.py compare \
  --agent response --encoded "$ENCODED" --output "$OUT/compare"
```

산출: `$OUT/compare/COMPARE.json` — `numeric_equivalence`(loss delta, lora_grad_diff, off/on calls,
enabled_observed), `toggle_changed_kernel`. `toggle_changed_kernel=true`로 확인되고 수치가 정상일 때만 아래 timing:

```bash
# warmup 3 + timed 3 (수 분~약 1h). toggle_changed_kernel=false면 스크립트가 timing을 생략한다.
systemd-run --scope -p MemoryMax=30G --setenv=MAX_JOBS=2 \
  "$VENV/bin/python" tools/training_acceleration/attention_ablation.py compare \
  --agent response --encoded "$ENCODED" --output "$OUT/compare_timed" --run-timing
```

(선택) Codex 교차검증 — 사용자 지침상 권장. 검수 모델은 `gpt-6-sol`(reasoning `high`):

```bash
codex exec -s read-only --skip-git-repo-check -m gpt-6-sol -c model_reasoning_effort=high \
  < docs/2026-11-01_claude_gemma4_attention_plan.md > /tmp/codex_attn_review.txt 2>&1
```

---

## 5. 예상 비용 / 중단 기준

- check: 모델 1회 로드(~수십초) + native train forward/backward 1회 + 작은 probe. GPU 수 분.
- compare **기본(bounded)**: 모델 1회 로드 + no-step forward×2뿐. **수 분**. 이게 먼저 돌아야 할 짧은 probe다.
- compare `--run-timing`: 위 + (warmup3+timed3)×2 = update 12회. cold 1-update ≈ 325초 기준 **수 분~약 1시간**(warm은
  더 빠를 수 있음, 실측 전 추정). **토글이 커널을 바꾼 게 확인될 때만** 돈다(아래 2).
- **중단 기준**:
  1. `peak_reserved_bytes`가 budget(32GB−512MiB)에 근접하거나 budget.check가 raise → 즉시 중단.
  2. `check`의 `native_train_profile.hd512_kernel_engaged == 0`(production 경로에서 global층이 전부 SDPA) →
     HD512 커널이 이 설정에서 선택조차 안 됨. compare의 `toggle_changed_kernel=false`면 스크립트가 timing을 자동
     생략하고 `timing_skipped_reason`을 남긴다. 폐기 결론은 "이 로드 경로에선 Gemma4 전용 커널 미선택"으로 난다.
     (침습 `global_layers_hd512_engaged`는 보조 지표일 뿐, production 판정은 native_train_profile로 한다.)
  3. `hd512_vs_sdpa_layer_probe`가 DRIFT(cosine<0.999 또는 max_relative 큼) 또는 nonfinite → 수치 불일치를
     그대로 보고하고 compare 수치는 신뢰하지 않는다.
  4. `flex_probe`가 global에서 UNSUPPORTED(eager·compiled 둘 다) → flex 경로는 거기서 종료, 추가 적용 금지.
     전체 모델 flex 확정은 별도 fresh-process 로드가 통과한 뒤에만.

증거 없는 속도/수치합격 주장 금지. 모든 판정은 `CHECK.json`/`COMPARE.json`의 실측 필드와 1절 줄번호로 뒷받침한다.
