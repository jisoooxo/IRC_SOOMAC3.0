import json
from pathlib import Path
import tempfile
import unittest
from .candidate_proof import sha, verify_bindings_only, verify_timing_proof


class CandidateProofTest(unittest.TestCase):
    def fixture(self, root):
        identity = {'prepared_sha256': 'prepared', 'initial_lora_sha256': 'initial'}
        evidence = {}
        for m in ('hf', 'native', 'smart'):
            (root/m).mkdir()
            for name in ('CANDIDATE_RESULT.json', 'INITIAL_LORA.safetensors',
                         'gradient0.pt', 'gradient1.pt', 'logits0.pt', 'logits1.pt'):
                p = root/m/name
                p.write_text(json.dumps(identity) if name.endswith('.json') else 'bounded synthetic')
                evidence[f'{m}/{name}'] = sha(p)
        d = {'status': 'GC_EQUIVALENCE_PASS', 'selected': 'smart', 'identity': identity,
             'runtime_control_issues': {}, 'evidence_sha256': evidence,
             'candidates': {'smart': {'passed': True, 'numeric_pass': True}}}
        path = root/'proof.json'
        path.write_text(json.dumps(d))
        return path, d

    def test_bound_proof(self):
        with tempfile.TemporaryDirectory() as t:
            p, _ = self.fixture(Path(t))
            verify_bindings_only(p, 'smart', 'prepared', 'initial')

    def test_changed_input(self):
        with tempfile.TemporaryDirectory() as t:
            p, _ = self.fixture(Path(t))
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                verify_bindings_only(p, 'smart', 'different', 'initial')

    def test_changed_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            p, _ = self.fixture(Path(t))
            (Path(t)/'hf/gradient0.pt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                verify_bindings_only(p, 'smart', 'prepared', 'initial')

    def test_status_fields_alone_not_enough(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'proof.json'
            p.write_text(json.dumps({'status': 'GC_EQUIVALENCE_PASS', 'selected': 'smart'}))
            with self.assertRaises(ValueError):
                verify_bindings_only(p, 'smart', 'prepared', 'initial')

    def test_even_consistent_status_fields_do_not_authorize_timing(self):
        with tempfile.TemporaryDirectory() as t:
            p, _ = self.fixture(Path(t))
            with self.assertRaisesRegex(ValueError, 'raw numerical evidence replay'):
                verify_timing_proof(p, 'smart', 'prepared', 'initial')
