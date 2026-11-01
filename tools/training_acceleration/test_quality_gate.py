"""CPU evidence tests, no GPU initialization or model loading."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import quality_gate as gate


class QualityCPU(unittest.TestCase):
    def test_normalized_utterance_preserves_meaningful_words(self):
        self.assertEqual(gate.normalize('  바로 담아줘! '), gate.normalize('바로담아줘'))
        self.assertNotEqual(gate.normalize('담지 마'), gate.normalize('담아'))

    def test_history_user_utterances_in_leak_scan(self):
        row = {'input': {'message': '수정해줘', 'recent_history': [
            {'role': 'user', 'content': '버섯 조금'}, {'role': 'assistant', 'content': '무엇을 고를까요?'}]}}
        self.assertEqual(gate.utterances(row), {'수정해줘', '버섯조금'})

    def test_agent_text_keys(self):
        for key in ('message', 'current_user_text', 'user_text'):
            self.assertEqual(gate.utterances({'input': {key: '넓은면으로 해요'}}), {'넓은면으로해요'})

    def test_group_not_family_is_explicit(self):
        self.assertIsNone(gate.group({'metadata': {'scenario_family': 'shared_skill'}}))
        self.assertEqual(gate.group({'metadata': {'dialogue_group': 'unseen17'}}), ('dialogue_group', 'unseen17'))

    def test_chunked_diff_exact_and_scaled(self):
        import torch
        x = torch.arange(1, 10001, dtype=torch.float32)
        same = gate._chunked_diff(torch, x, x)
        scaled = gate._chunked_diff(torch, x, x*2)
        self.assertEqual(same['relative_l2_error'], 0.)
        self.assertAlmostEqual(scaled['cosine'], 1.)
        self.assertAlmostEqual(scaled['relative_l2_error'], 1.)
        self.assertFalse(scaled['allclose_original_tolerance'])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
