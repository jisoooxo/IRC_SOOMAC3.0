"""Fail closed on missing runtime/dropout observation, without model/GPU imports."""
import copy
import unittest

from .compare_candidate import require_runtime


class CandidateRuntimeTest(unittest.TestCase):
    def report(self, mode):
        cls = 'UnslothCheckpointFunction' if mode == 'smart' else 'CheckpointFunction'
        repeat = {'dropout_all_zero': True, 'dropout_modules': 328, 'gradient_norm_fp64': 1.0,
                  'runtime': {'checkpoint_calls': {cls+'.forward': 48, cls+'.backward': 48},
                              'fused_dropout': {'0.0': 328, 'forward_p=0.0': 328}}}
        return {'gradient_checkpointing_flags': {'decoder': True}, 'dropout_restored': True,
                'repeats': [repeat, copy.deepcopy(repeat)]}

    def test_supported_observations(self):
        for mode in ('hf', 'native', 'smart'):
            require_runtime(self.report(mode), mode)

    def test_flags_do_not_replace_actual_checkpoint_calls(self):
        item = self.report('native')
        item['repeats'][0]['runtime']['checkpoint_calls'] = {}
        with self.assertRaisesRegex(ValueError, 'unproven'):
            require_runtime(item, 'native')

    def test_nonzero_or_missing_fused_dropout_blocks(self):
        for drop in ({}, {'0.05': 328}, {'0.0': 0}):
            item = self.report('smart')
            item['repeats'][0]['runtime']['fused_dropout'] = drop
            with self.assertRaisesRegex(ValueError, 'fused dropout'):
                require_runtime(item, 'smart')

    def test_all_expected_lora_dropout_modules(self):
        item = self.report('native')
        item['repeats'][0]['dropout_modules'] = 1
        with self.assertRaisesRegex(ValueError, 'dropout not zero'):
            require_runtime(item, 'native')

    def test_restoration_required(self):
        item = self.report('hf')
        item['dropout_restored'] = False
        with self.assertRaisesRegex(ValueError, 'evidence missing'):
            require_runtime(item, 'hf')

    def test_native_trace_not_smart_proof(self):
        with self.assertRaisesRegex(ValueError, 'unproven'):
            require_runtime(self.report('native'), 'smart')

    def test_smart_trace_not_native_proof(self):
        with self.assertRaisesRegex(ValueError, 'unproven'):
            require_runtime(self.report('smart'), 'native')
