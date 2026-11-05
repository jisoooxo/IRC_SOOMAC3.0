#!/usr/bin/env bash
# Gemma 4 12B (NF4) + Agent LoRA 3개를 vLLM OpenAI 호환 서버로 띄운다.
# LLM 노드(soomac_llm 환경)는 LLM_BACKEND="vllm"일 때 이 서버에 HTTP로 요청한다.
# 로봇 스택을 쓸 때만 켜고, 끄면 GPU를 전부 돌려준다. 인터넷 없이 동작하도록 오프라인 변수를 고정한다.
#
# 어댑터는 vLLM용 변환본(글로벌 k_eq_v 층의 k_proj LoRA를 v_proj로 복제)을 쓴다.
# 어댑터를 새로 학습하면 model_runtime.py 경로를 바꾸고 tools/vllm_server/convert_adapters_for_vllm.py로 변환한 뒤 서버를 다시 시작한다.
# 학습 후 한 번에 확인: tools/vllm_server/check_after_training.sh
set -euo pipefail

ENV_PY_BIN=/home/roma/miniconda3/envs/soomac_vllm_server/bin
MODEL_PATH=${MODEL_PATH:-/home/roma/Desktop/sLLM/gemma-4-12B-it}
ADAPTER_ROOT=${ADAPTER_ROOT:-/home/roma/IRC_SOOMAC3.0/models/vllm_adapters}
HOST=${VLLM_HOST:-127.0.0.1}
PORT=${VLLM_PORT:-8000}
# 2026-11-05 전체 Decision 보강 문구/16K 적용: .45는 KV 부족, .58은 단독 기동·추론 통과.
# 이전10K/.45의 STT·TTS 동시 상주 측정은 이16K 설정의 동시 실행 보장이 아니다.
GPU_MEM=${VLLM_GPU_MEM:-0.58}
# 클라이언트 input_budget.SERVING_MAX_MODEL_LEN과 함께 맞춘다.
MAX_LEN=${VLLM_MAX_LEN:-16384}
MAX_LORA_RANK=${VLLM_MAX_LORA_RANK:-16} # 어댑터 rank가 커지면 같이 올린다
# 서버 기본값(약 2.5k)이면 6~8k 입력을 여러 조각으로 나눠 계산해 HF와 수치가 갈라진다(오프라인 실측 12/14 → 9/14).
# 한 요청을 한 번에 계산하도록 최대 입력 길이 이상으로 둔다.
MAX_BATCHED=${VLLM_MAX_BATCHED:-$MAX_LEN}
# 서버 기본 동시 처리 수(256)만큼 CUDA graph·작업 공간을 잡는다. 노드는 한 번에 1개씩만 보내므로 작게 둔다.
MAX_SEQS=${VLLM_MAX_SEQS:-2}
# batch invariant는 기본 켬. 2026-11-05 동일 요청의 서버 재기동 간 차이는 남아 완전 결정성을 보장하지 않는다.
# 0.45 + STT·TTS 상주 실측: 17턴 2회 필드 17/17 일치, 25.8GiB, 17턴 36.0초(끔 37.2초). 끄려면 VLLM_BATCH_INVARIANT=0
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}
# 실험용(기본 꺼짐). MTP 추측 디코딩: VLLM_SPEC_MODEL=/home/roma/models/gemma-4-12B-it-assistant VLLM_SPEC_K=2
SPEC_MODEL=${VLLM_SPEC_MODEL:-}
SPEC_K=${VLLM_SPEC_K:-2}
EXTRA_ARGS=()
if [ -n "$SPEC_MODEL" ]; then
    # 보조 모델은 bf16 원본 그대로 올린다. 지정하지 않으면 본 모델의 bitsandbytes 로더를 물려받아 시작에 실패한다.
    EXTRA_ARGS+=(--speculative-config "{\"method\": \"mtp\", \"model\": \"$SPEC_MODEL\", \"num_speculative_tokens\": $SPEC_K, \"draft_load_config\": {\"load_format\": \"auto\"}}")
fi
# 기반 모델 양자화. 기본 bitsandbytes(NF4, 어댑터 학습 기반과 같음).
# 실험용: VLLM_QUANT=auto + MODEL_PATH=<이미 양자화된 체크포인트>(예: NVFP4) → 체크포인트 설정을 그대로 쓴다.
QUANT_MODE=${VLLM_QUANT:-bitsandbytes}
if [ "$QUANT_MODE" = "bitsandbytes" ]; then
    QUANT_ARGS=(--quantization bitsandbytes --load-format bitsandbytes)
else
    QUANT_ARGS=()
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTHONPATH  # ROS·예전 워크스페이스 경로가 섞이지 않게 한다

exec "$ENV_PY_BIN/vllm" serve "$MODEL_PATH" \
    --served-model-name gemma4-base \
    --dtype bfloat16 \
    "${QUANT_ARGS[@]}" \
    --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras 3 \
    --lora-modules \
        decision="$ADAPTER_ROOT/decision" \
        recommendation="$ADAPTER_ROOT/recommendation" \
        response="$ADAPTER_ROOT/response" \
    --enable-prefix-caching --enable-prompt-tokens-details \
    --max-model-len "$MAX_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$GPU_MEM" \
    --limit-mm-per-prompt '{"image": 8, "audio": 0}' \
    --structured-outputs-config '{"backend": "xgrammar"}' \
    "${EXTRA_ARGS[@]}" \
    --host "$HOST" --port "$PORT"
