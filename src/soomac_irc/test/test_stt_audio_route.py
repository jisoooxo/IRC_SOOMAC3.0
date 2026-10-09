import os
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from soomac_irc.stt_nemotron_node import NemotronSttNode


def _completed(stdout):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr='')


def _node_stub():
    logger = SimpleNamespace(warning=lambda _message: None)
    return SimpleNamespace(get_logger=lambda: logger)


class TestSttAudioRoute(unittest.TestCase):
    def test_fails_closed_without_pactl(self):
        with patch('shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, '검증할 수 없어요'):
                NemotronSttNode._resolve_pulse_source(_node_stub())

    def test_rejects_output_monitor_as_microphone(self):
        commands = [
            _completed('1\talsa_output.card.monitor\tmodule\n'),
            _completed('alsa_output.card.monitor\n'),
        ]
        with patch.dict(os.environ, {}, clear=True), \
                patch('shutil.which', return_value='/usr/bin/pactl'), \
                patch('subprocess.run', side_effect=commands):
            with self.assertRaisesRegex(RuntimeError, '출력 모니터'):
                NemotronSttNode._resolve_pulse_source(_node_stub())

    def test_uses_requested_microphone_source(self):
        microphone = 'alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo'
        sources = _completed(f'3\t{microphone}\tmodule\n')
        with patch.dict(
                os.environ,
                {'SOOMAC_AUDIO_SOURCE': microphone},
                clear=True), \
                patch('shutil.which', return_value='/usr/bin/pactl'), \
                patch('subprocess.run', return_value=sources) as run:
            resolved = NemotronSttNode._resolve_pulse_source(_node_stub())

        self.assertEqual(resolved, microphone)
        self.assertEqual(run.call_count, 1)

    def test_rejects_disconnected_requested_source(self):
        with patch.dict(
                os.environ,
                {'SOOMAC_AUDIO_SOURCE': 'alsa_input.usb.missing'},
                clear=True), \
                patch('shutil.which', return_value='/usr/bin/pactl'), \
                patch('subprocess.run', return_value=_completed('')):
            with self.assertRaisesRegex(
                    RuntimeError,
                    '지정한 입력 장치가 없어요'):
                NemotronSttNode._resolve_pulse_source(_node_stub())


if __name__ == '__main__':
    unittest.main()
