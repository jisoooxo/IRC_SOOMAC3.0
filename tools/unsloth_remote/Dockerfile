# Separate future TAR: candidate build checks the shared Decision CLI;
# GPU correctness and save/resume validation remain separate gates.
ARG TESTED_CANDIDATE_IMAGE=scratch
ARG TESTED_CANDIDATE_DIGEST=unknown
FROM --platform=linux/amd64 unsloth/unsloth:core-nightly-2026.10.28@sha256:852e00064749edbaecb57f8388a6776915bfe744cb5238d34b809d152623d87f AS candidate

USER 0:0
ARG LAB_INPUTS
WORKDIR /app
COPY --chown=1000:1000 bundle/ /app/
COPY --chown=1000:1000 bundle/dataset/ /app/assets/dataset/
COPY --chown=1000:1000 bundle/assets/model-processor/ /app/assets/model-processor/
COPY --chown=1000:1000 source/unsloth_training/production.py /app/tools/unsloth_training/production.py
COPY --chown=1000:1000 source/training_acceleration/benchmark.py /app/tools/training_acceleration/benchmark.py
COPY --chown=1000:1000 --chmod=0555 package/run_training.sh /app/run_training.sh
COPY --chown=1000:1000 package/test_processor_encoding.py /app/test_processor_encoding.py

RUN test -n "$LAB_INPUTS" && \
    test "$LAB_INPUTS" = "$(/opt/unsloth-venv/bin/python -c 'import json; print(json.dumps(json.load(open("/app/remote/lab-inputs.json")),separators=(",",":")))')" && \
    /opt/unsloth-venv/bin/python -c 'import hashlib,json; from pathlib import Path; r=Path("/app"); a=Path("/app/assets"); m=a/"dataset/MANIFEST.json"; h="7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa"; digest=lambda q: hashlib.sha256(q.read_bytes()).hexdigest(); assert m.is_file() and not m.is_symlink() and digest(m)==h; d=json.loads(m.read_text()); assert all((a/"dataset"/n).is_file() and not (a/"dataset"/n).is_symlink() and digest(a/"dataset"/n)==v for n,v in d["file_sha256"].items()); p=a/"model-processor"; x=json.loads((r/"RELEASE.json").read_text()); assert x["source_dataset_manifest_sha256"]==h; assert all((p/n).is_file() and not (p/n).is_symlink() and digest(p/n)==v for n,v in x["processor_files"].items())' && \
    /opt/unsloth-venv/bin/python -m py_compile /app/tools/unsloth_training/production.py /app/tools/training_acceleration/benchmark.py && \
    /opt/unsloth-venv/bin/python /app/tools/unsloth_training/production.py --help > /tmp/soomac-production-help && \
    for flag in decision --execution-environment --release --model --dataset --processor --budget-bytes --validation-receipt; do grep -Fq -- "$flag" /tmp/soomac-production-help || exit 1; done && \
    chmod -R a+rX /app/common /app/remote /app/dataset /app/assets /app/tools

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp \
    HF_HOME=/tmp/soomac-hf HUGGINGFACE_HUB_CACHE=/tmp/soomac-hf \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    UNSLOTH_SKIP_NOTEBOOK_SYNC=1 \
    DATASET_DIR=/app/assets/dataset MODEL_DIR=/model \
    PROCESSOR_DIR=/app/assets/model-processor CHECKPOINT_DIR=/checkpoints \
    SOOMAC_REMOTE_REQUESTED_BUDGET_BYTES=44000000000 \
    SOOMAC_NON_ALLOCATOR_RESERVE_BYTES=1073741824

LABEL lab.training.spec="2.9" \
      lab.training.progress="1" \
      lab.training.resume="output-dir-v1" \
      lab.training.inputs="${LAB_INPUTS}" \
      lab.training.task="decision-unsloth-latest-compound-semantics" \
      lab.training.release="compound_semantics_20261101_r5"
USER 1000:1000
ENTRYPOINT ["/app/run_training.sh"]
CMD ["train"]

# Only the final image carries the candidate-image GPU validation receipt.
# Use the exact tested candidate image ID, not a possibly rebuilt stage/tag.
# This last layer cannot quote its own final image digest (self-reference).
FROM ${TESTED_CANDIDATE_IMAGE} AS final
USER 0:0
ARG TESTED_CANDIDATE_IMAGE
ARG TESTED_CANDIDATE_DIGEST
COPY --chown=1000:1000 validation/ /app/validation/
RUN TESTED_CANDIDATE_DIGEST="$TESTED_CANDIDATE_DIGEST" /opt/unsloth-venv/bin/python -c 'import hashlib,json,os,re; from pathlib import Path; root=Path("/app/validation"); p=root/"VALIDATION_RECEIPT.json"; d=json.loads(p.read_text()); e=d.get("evidence_files"); tested=os.environ["TESTED_CANDIDATE_DIGEST"]; assert re.fullmatch(r"sha256:[0-9a-f]{64}",tested) and d.get("tested_candidate_image_digest")==tested; assert d.get("status")=="PASS" and d.get("agent")=="decision" and isinstance(e,dict) and set(e)=={"loss","gradient","restore"}; names={"VALIDATION_RECEIPT.json"}; assert all(isinstance(x,dict) and set(x)=={"path","sha256"} and Path(x["path"]).name==x["path"] and x["path"].endswith(".json") for x in e.values()); names.update(x["path"] for x in e.values()); assert {q.name for q in root.iterdir()}==names; assert all((root/n).is_file() and not (root/n).is_symlink() and (root/n).stat().st_size<=65536 for n in names); assert all(hashlib.sha256((root/x["path"]).read_bytes()).hexdigest()==x["sha256"] for x in e.values())'
USER 1000:1000
