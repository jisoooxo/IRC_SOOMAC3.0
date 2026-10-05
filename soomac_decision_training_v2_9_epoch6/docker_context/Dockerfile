FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LAB_INPUTS
ARG TRAIN_QUANTIZATION=int4
ARG TRAIN_MODE=full

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates python3 python3-dev python3-pip unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchvision==0.23.0

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python3 -c "from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer; from huggingface_hub import HfApi, snapshot_download; from huggingface_hub.utils import filter_repo_objects; import bitsandbytes, peft; print('PASS: training imports')"

WORKDIR /app
COPY assets/dataset /app/assets/dataset

COPY preflight.py lab_progress.py gpu_budget.py train_in_container.py run_training.sh /app/
COPY --chmod=0555 lab-entrypoint.sh /usr/local/bin/lab-entrypoint
RUN python3 /app/preflight.py \
    && chmod 0555 /app/run_training.sh /app/train_in_container.py \
    && chmod -R a+rX /app

RUN test -n "$LAB_INPUTS" || (echo 'Pass lab-inputs.json as LAB_INPUTS' >&2; exit 1)

LABEL lab.training.spec="2.9" \
      lab.training.progress="1" \
      lab.training.resume="output-dir-v1" \
      lab.training.inputs="${LAB_INPUTS}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    TRAIN_QUANTIZATION=${TRAIN_QUANTIZATION} \
    TRAIN_MODE=${TRAIN_MODE}

ENTRYPOINT ["/usr/local/bin/lab-entrypoint"]
CMD ["/app/run_training.sh"]
