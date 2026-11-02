"""Bind timing to the exact comparison evidence, not two editable status fields."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_timing_proof(path, mode, prepared_sha, initial_sha):
    # Candidate v3 failed. Hash binding alone cannot turn operator-edited PASS
    # fields into numerical evidence. Keep timing closed until independent
    # replay of all raw FP64 comparisons is integrated and reviewed.
    raise ValueError('timing disabled: raw numerical evidence replay is required before authorization')


def verify_bindings_only(path, mode, prepared_sha, initial_sha):
    """CPU-tested bookkeeping, NOT timing/production authorization."""
    path = Path(path)
    proof = json.loads(path.read_text())
    candidate = proof.get('candidates', {}).get(mode, {})
    issues = proof.get('runtime_control_issues', {})
    if (proof.get('status') != 'GC_EQUIVALENCE_PASS' or proof.get('selected') != mode
            or candidate.get('passed') is not True or candidate.get('numeric_pass') is not True
            or any(m in issues for m in ('hf', mode))):
        raise ValueError('no passing selected GC proof')
    identity = proof.get('identity', {})
    if identity.get('prepared_sha256') != prepared_sha or identity.get('initial_lora_sha256') != initial_sha:
        raise ValueError('timing input/initial identity differs from proof')
    expected = {f'{m}/{name}' for m in ('hf', 'native', 'smart') for name in
                ('CANDIDATE_RESULT.json', 'INITIAL_LORA.safetensors',
                 'gradient0.pt', 'gradient1.pt', 'logits0.pt', 'logits1.pt')}
    evidence = proof.get('evidence_sha256', {})
    if set(evidence) != expected:
        raise ValueError('incomplete or unsafe evidence paths')
    for relative, digest in evidence.items():
        item = path.parent / relative
        if item.is_symlink() or item.parent.is_symlink() or not item.is_file() or sha(item) != digest:
            raise ValueError(f'evidence changed: {relative}')
    for m in ('hf', 'native', 'smart'):
        report = json.loads((path.parent / m / 'CANDIDATE_RESULT.json').read_text())
        if any(report.get(key) != value for key, value in identity.items()):
            raise ValueError(f'bound report identity differs: {m}')
    return proof
