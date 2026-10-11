FROM soomac/decision-v3.1:v2.9

ARG LAB_INPUTS
ARG TRAIN_QUANTIZATION=int4
ARG TRAIN_MODE=full

WORKDIR /app
RUN rm -rf /app/assets/dataset
COPY assets/dataset /app/assets/dataset
COPY preflight.py lab_progress.py gpu_budget.py train_in_container.py token_length_audit.py run_training.sh /app/
COPY README.md training_config.json environment_manifest.json SHA256SUMS requirements.txt /app/
COPY --chmod=0555 lab-entrypoint.sh /usr/local/bin/lab-entrypoint
RUN python3 -c "from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer; from huggingface_hub import HfApi, snapshot_download; from huggingface_hub.utils import filter_repo_objects; import bitsandbytes, peft; print('PASS: inherited training imports')" \
    && python3 /app/preflight.py \
    && chmod 0555 /app/run_training.sh /app/train_in_container.py /app/token_length_audit.py \
    && chmod -R a+rX /app

RUN test -n "$LAB_INPUTS" || (echo 'Pass lab-inputs.json as LAB_INPUTS' >&2; exit 1)

LABEL lab.training.spec="2.9" \
      lab.training.progress="1" \
      lab.training.resume="output-dir-v1" \
      lab.training.task="decision-v3.2-adapter" \
      lab.training.inputs="${LAB_INPUTS}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    TRAIN_QUANTIZATION=${TRAIN_QUANTIZATION} \
    TRAIN_MODE=${TRAIN_MODE}

ENTRYPOINT ["/usr/local/bin/lab-entrypoint"]
CMD ["/app/run_training.sh"]
