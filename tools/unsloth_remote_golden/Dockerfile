# Decision Unsloth remote candidate rebuilt from the GOLDEN local Response environment
# (r5 venv over gemma4_env, observed in running PID 1405238 on 2026-11-02).
# Written for the legacy builder (DOCKER_BUILDKIT=0) so --memory/--cpuset-cpus apply;
# therefore no COPY --chmod and no heredocs.
# Candidate = GPU-UNVERIFIED. Decision `train` refuses to start until a GPU validation
# receipt is added by Dockerfile.final in /app/gpu-validation.

# ---- 1. golden conda Python (exact builds from repo.anaconda.com/pkgs/main) ----
FROM mambaorg/micromamba@sha256:4292706e8f440aace45678ac27d99448576b2798dec1746fbeb3285213509376 AS pyenv
USER root
COPY conda/ /conda-pkgs/
RUN cd /conda-pkgs && sha256sum -c SHA256SUMS > /dev/null && \
    micromamba create -y --offline --no-rc -p /opt/soomac/py -f /conda-pkgs/explicit.txt && \
    /opt/soomac/py/bin/python -c "import sys; assert sys.version.split()[0] == '3.10.20', sys.version" && \
    test "$(ls /opt/soomac/py/conda-meta/*.json | wc -l)" = 62

# ---- 2. golden wheels, no resolver, byte-verified against the golden install ----
FROM pyenv AS wheels
COPY wheels/ /wheels/
COPY golden/ /opt/soomac/golden/
COPY package/golden_record.py /opt/soomac/golden_record.py
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN /opt/soomac/py/bin/python -m pip install --no-deps --no-index --require-hashes \
        --find-links /wheels -r /opt/soomac/golden/requirements-golden.lock && \
    /opt/soomac/py/bin/python -m pip check && \
    /opt/soomac/py/bin/python /opt/soomac/golden_record.py verify \
        --records /opt/soomac/golden/GOLDEN_RECORDS.json \
        --allow-extra pip wheel packaging \
        --output /opt/soomac/golden/GOLDEN_VERIFY.json && \
    /opt/soomac/py/bin/python -c "import importlib.metadata as m; want={'torch':'2.12.1+cu130','triton':'3.7.1','transformers':'5.13.1','peft':'0.19.1','bitsandbytes':'0.49.2','accelerate':'1.14.0','unsloth':'2026.10.11','unsloth-zoo':'2026.10.11','safetensors':'0.8.0','trl':'1.8.0','xformers':'0.0.35'}; got={k:m.version(k) for k in want}; assert got==want, got"

# ---- 3. runtime image ----
FROM ubuntu@sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11 AS candidate
USER 0:0
# Triton JIT-compiles small C launchers at runtime with the system C compiler (host golden: gcc 11.4).
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/* && \
    dpkg-query -W -f='${Package}=${Version}\n' gcc gcc-11 libc6 libc6-dev > /opt/apt-runtime-versions.txt && \
    groupadd -g 1000 soomac && useradd -u 1000 -g 1000 -M -d /tmp -s /usr/sbin/nologin soomac
# The lab runs UID:GID 1000:1000. Without a passwd entry torch's default inductor cache dir
# (getpass.getuser) raises KeyError at torch._dynamo import (found by the c1 CPU probe).
COPY --from=wheels /opt/soomac /opt/soomac
RUN ln -s /opt/soomac/py/bin/python3 /usr/local/bin/python3 && \
    ln -s /opt/soomac/py/bin/python3 /usr/local/bin/python

ARG LAB_INPUTS
WORKDIR /app
COPY --chown=1000:1000 bundle/ /app/
COPY --chown=1000:1000 bundle/dataset/ /app/assets/dataset/
COPY --chown=1000:1000 source/unsloth_training/production.py /app/tools/unsloth_training/production.py
COPY --chown=1000:1000 source/training_acceleration/benchmark.py /app/tools/training_acceleration/benchmark.py
COPY --chown=1000:1000 package/run_training.sh /app/run_training.sh
COPY --chown=1000:1000 package/test_processor_encoding.py /app/test_processor_encoding.py

RUN chmod 0555 /app/run_training.sh && \
    test -n "$LAB_INPUTS" && \
    test "$LAB_INPUTS" = "$(/opt/soomac/py/bin/python -c 'import json; print(json.dumps(json.load(open("/app/remote/lab-inputs.json")),separators=(",",":")))')" && \
    /opt/soomac/py/bin/python -c 'import hashlib,json; from pathlib import Path; r=Path("/app"); a=Path("/app/assets"); m=a/"dataset/MANIFEST.json"; h="7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa"; digest=lambda q: hashlib.sha256(q.read_bytes()).hexdigest(); assert m.is_file() and not m.is_symlink() and digest(m)==h; d=json.loads(m.read_text()); assert all((a/"dataset"/n).is_file() and not (a/"dataset"/n).is_symlink() and digest(a/"dataset"/n)==v for n,v in d["file_sha256"].items()); p=a/"model-processor"; x=json.loads((r/"RELEASE.json").read_text()); assert x["source_dataset_manifest_sha256"]==h; assert all((p/n).is_file() and not (p/n).is_symlink() and digest(p/n)==v for n,v in x["processor_files"].items())' && \
    /opt/soomac/py/bin/python /opt/soomac/golden_record.py verify \
        --records /opt/soomac/golden/GOLDEN_RECORDS.json --allow-extra pip wheel packaging \
        --output /tmp/golden-verify-runtime.json && \
    /opt/soomac/py/bin/python -m py_compile /app/tools/unsloth_training/production.py /app/tools/training_acceleration/benchmark.py && \
    /opt/soomac/py/bin/python /app/tools/unsloth_training/production.py --help > /tmp/soomac-production-help && \
    for flag in decision --execution-environment --release --model --dataset --processor --budget-bytes --validation-receipt; do grep -Fq -- "$flag" /tmp/soomac-production-help || exit 1; done && \
    rm -f /tmp/soomac-production-help /tmp/golden-verify-runtime.json && \
    find /app/tools -name '__pycache__' -type d -prune -exec rm -rf {} + && \
    mkdir -p /app/gpu-validation && chown 1000:1000 /app/gpu-validation && \
    chmod -R a+rX /app /opt/soomac

ENV PATH=/opt/soomac/py/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp \
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
      lab.training.task="decision-unsloth-golden-response-env" \
      lab.training.release="compound_semantics_20261101_r5" \
      soomac.candidate.status="GPU_UNVERIFIED" \
      soomac.golden.python="conda python-3.10.20-h17756b0_1" \
      soomac.golden.torch="2.12.1+cu130" \
      soomac.golden.min_driver="580.65.06"
USER 1000:1000
ENTRYPOINT ["/app/run_training.sh"]
CMD ["train"]
