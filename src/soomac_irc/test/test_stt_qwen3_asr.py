import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from soomac_irc import stt_qwen3_asr_node
from soomac_irc.stt_qwen3_asr_node import Qwen3AsrSttNode


class _Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[10, 11]]))
        self.to_arguments = None

    def to(self, device, dtype):
        self.to_arguments = (device, dtype)
        return self


class _Processor:
    def __init__(self):
        self.inputs = _Inputs()
        self.audio = None
        self.language = None
        self.decoded_ids = None
        self.return_format = None

    def apply_transcription_request(self, audio, language):
        self.audio = audio
        self.language = language
        return self.inputs

    def decode(self, generated_ids, return_format):
        self.decoded_ids = generated_ids
        self.return_format = return_format
        return ['페파론치노   많이']


class _Model:
    device = torch.device('cpu')
    dtype = torch.float32

    def generate(self, input_ids, max_new_tokens):
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens
        return torch.tensor([[10, 11, 12, 13]])


class TestQwen3AsrNode(unittest.TestCase):
    def test_recognition_keeps_menu_word_without_hotword_normalization(self):
        processor = _Processor()
        model = _Model()
        published = []
        logger = SimpleNamespace(info=lambda _message: None, error=self.fail)
        node = SimpleNamespace(
            stop_event=SimpleNamespace(is_set=lambda: False),
            _current_session=lambda: 7,
            _is_gate_open=lambda: True,
            processor=processor,
            model=model,
            state_lock=threading.Lock(),
            session=7,
            listening_enabled=True,
            gate_open=True,
            _gate_open_at=1.0,
            question_pub=SimpleNamespace(
                publish=lambda message: published.append(message.data)),
            get_logger=lambda: logger,
        )
        pcm = np.zeros(16000, dtype='<i2').tobytes()

        Qwen3AsrSttNode._recognize_utterance(
            node,
            pcm,
            captured_session=7,
            utterance_ended_at=time.perf_counter(),
            captured_seconds=1.0)

        self.assertEqual(published, ['페파론치노 많이'])
        self.assertEqual(processor.language, 'ko')
        self.assertEqual(processor.return_format, 'transcription_only')
        self.assertTrue(torch.equal(
            processor.decoded_ids,
            torch.tensor([[12, 13]])))
        self.assertFalse(node.gate_open)
        self.assertIsNone(node._gate_open_at)

    def test_missing_weights_fail_before_processor_load(self):
        logger = SimpleNamespace(info=lambda _message: None, error=lambda _message: None)
        node = SimpleNamespace(
            _choose_compute_device=lambda: 'cuda',
            get_logger=lambda: logger)

        with tempfile.TemporaryDirectory() as model_dir, \
                patch.object(stt_qwen3_asr_node, 'MODEL_DIR', model_dir), \
                patch.object(
                    stt_qwen3_asr_node.AutoProcessor,
                    'from_pretrained',
                    Mock()) as load_processor:
            with self.assertRaisesRegex(FileNotFoundError, '가중치가 없어요'):
                Qwen3AsrSttNode._load_model(node)

        load_processor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
