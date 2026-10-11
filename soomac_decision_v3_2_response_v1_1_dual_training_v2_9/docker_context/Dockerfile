FROM soomac/decision-v3.2:v2.9

ARG LAB_INPUTS
ARG TRAIN_QUANTIZATION=int4
ARG TRAIN_MODE=full

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
      lab.training.task="decision-v3.2-response-v1.1-dual-adapter" \
      lab.training.inputs="${LAB_INPUTS}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    TRAIN_QUANTIZATION=${TRAIN_QUANTIZATION} \
    TRAIN_MODE=${TRAIN_MODE}

ENTRYPOINT ["/usr/local/bin/lab-entrypoint"]
CMD ["/app/run_training.sh"]
