#!/usr/bin/env python3
"""Local disposable candidate testing; no toolkit/daemon/system-env modifications.

Host driver files are read-only TEST mounts, never copied into the shipping image.
Remote deployment uses the server's NVIDIA runtime instead.
"""
import argparse
from pathlib import Path
import subprocess

IMAGE = 'soomac/decision-unsloth:candidate-20261101-v3'
IMAGE_ID = 'sha256:bf465ad8543b8a280f8322980a879fb46cba09d2356ee8d49581c082a95c2d3a'
BASE = Path('/home/roma/Desktop/sLLM/gemma-4-12B-it')
REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/tmp/soomac-candidate-v3-audit-20261101')
PREPARED = Path('/tmp/soomac-training-accel-20261101/quality_prepared_v2')
CACHE = Path('/tmp/soomac-unsloth-d-candidate-cache-20261101')


def command(extra_env=None):
    info = subprocess.check_output(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], text=True).strip()
    if info != IMAGE_ID:
        raise RuntimeError('candidate tag no longer identifies reviewed image')
    ROOT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    cmd = ['docker', 'run', '--rm', '--network', 'none', '--user', '1000:1000',
           '--cpus', '8', '--memory', '32g', '--memory-swap', '32g', '--shm-size', '4g',
           '--device', '/dev/nvidia0', '--device', '/dev/nvidiactl', '--device', '/dev/nvidia-uvm',
           '--security-opt', 'no-new-privileges',
           '--mount', f'type=bind,src={BASE},dst=/model,readonly',
           '--mount', f'type=bind,src={ROOT},dst=/evidence',
           '--mount', f'type=bind,src={PREPARED},dst=/prepared,readonly',
           '--mount', f'type=bind,src={REPO}/tools/unsloth_remote,dst=/diagnostics,readonly',
           '--mount', f'type=bind,src={CACHE},dst=/tmp/soomac-unsloth-container',
           '--mount', 'type=bind,src=/tmp/soomac-r5-gpu.lock,dst=/tmp/soomac-r5-gpu.lock',
           '--mount', 'type=bind,src=/usr/bin/nvidia-smi,dst=/usr/local/bin/nvidia-smi,readonly']
    for name in ('libcuda.so.1', 'libnvidia-ml.so.1', 'libnvidia-ptxjitcompiler.so.1', 'libnvidia-nvvm.so.4'):
        source = (Path('/lib/x86_64-linux-gnu') / name).resolve(strict=True)
        cmd += ['--mount', f'type=bind,src={source},dst=/tmp/host-driver/{name},readonly']
    env = {'NVIDIA_VISIBLE_DEVICES': 'void', 'LD_LIBRARY_PATH': '/tmp/host-driver:/usr/local/cuda/lib64',
           'HOME': '/tmp', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'UNSLOTH_DISABLE_STATISTICS': '1', 'UNSLOTH_RETURN_LOGITS': '1',
           'UNSLOTH_COMPILE_LOCATION': '/tmp/soomac-unsloth-container/compiled',
           'TORCHINDUCTOR_CACHE_DIR': '/tmp/soomac-unsloth-container/inductor',
           'TRITON_CACHE_DIR': '/tmp/soomac-unsloth-container/triton',
           'MAX_JOBS': '2', 'TORCHINDUCTOR_COMPILE_THREADS': '2', 'OMP_NUM_THREADS': '4',
           'GPU_MEMORY_BUDGET_BYTES': '32000000000', 'GPU_MEMORY_FRACTION': '0.97'}
    env.update(extra_env or {})
    for key, value in env.items():
        cmd += ['--env', f'{key}={value}']
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('cuda-check', 'hf', 'native', 'smart', 'timing'))
    parser.add_argument('--gc', choices=('native', 'smart'), default='smart')
    args = parser.parse_args()
    if args.mode == 'cuda-check':
        cmd = command() + ['--entrypoint', '/opt/unsloth-venv/bin/python', IMAGE, '-c',
            'import torch; print(torch.__version__,torch.cuda.is_available()); '
            'assert torch.cuda.is_available(); print((torch.ones(1,device="cuda")+1).item())']
    else:
        selected = args.gc if args.mode == 'timing' else args.mode
        label = ('timing-' + selected) if args.mode == 'timing' else selected
        cmd = command() + ['--entrypoint', '/opt/unsloth-venv/bin/python', IMAGE,
            '/diagnostics/validate_candidate.py', '--mode', selected, '--prepared', '/prepared',
            '--output', '/evidence/' + label]
        if selected != 'hf':
            cmd += ['--initial', '/evidence/hf/INITIAL_LORA.safetensors']
        if args.mode == 'timing':
            cmd += ['--timing', '--equivalence', '/evidence/GC_EQUIVALENCE.json']
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
