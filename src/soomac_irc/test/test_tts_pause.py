import unittest

import numpy as np

from soomac_irc.tts_pause import SentencePauseStretcher, sentence_layout

SR = 24000
RATE = 0.14   # 글자당 말소리 초 (모듈 기본값과 같게)


def tone(seconds):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def speech(chars):
    return tone(chars * RATE)


def run(text, audio, chunk_sizes=None, target=0.60):
    stretcher = SentencePauseStretcher(text, SR, target)
    if chunk_sizes is None:
        chunk_sizes = [len(audio)]
    out, start = [], 0
    sizes = list(chunk_sizes)
    while start < len(audio):
        size = sizes.pop(0) if sizes else len(audio)
        out.append(stretcher.feed(audio[start:start + size]))
        start += size
    out.append(stretcher.flush())
    return np.concatenate(out), stretcher


class TestSentenceLayout(unittest.TestCase):
    def test_positions_ignore_spaces_and_last_end(self):
        count, ends, commas = sentence_layout('가나다, 라마. 바사?')
        self.assertEqual(count, 7)
        self.assertEqual(ends, [5])
        self.assertEqual(commas, [3])


class TestSentencePauseStretcher(unittest.TestCase):
    def test_sentence_pause_is_stretched_to_target(self):
        audio = np.concatenate([speech(5), silence(0.30), speech(5)])
        out, stretcher = run('가나다라마. 바사아자차.', audio)
        self.assertEqual(len(stretcher.inserted), 1)
        self.assertAlmostEqual(len(out) - len(audio), int(SR * 0.30), delta=SR * 0.011)

    def test_comma_pause_is_not_stretched(self):
        audio = np.concatenate([speech(5), silence(0.30), speech(5)])
        out, stretcher = run('가나다라마, 바사아자차.', audio)
        self.assertEqual(stretcher.inserted, [])
        np.testing.assert_array_equal(out, audio)

    def test_comma_closer_than_sentence_end_wins(self):
        # 쉼표(3글자 뒤)가 문장 끝(9글자 뒤)보다 무음 위치에 가깝다
        audio = np.concatenate([speech(3), silence(0.30), speech(9)])
        _, stretcher = run('가나다, 라마바사아자. 차카.', audio)
        self.assertEqual(stretcher.inserted, [])

    def test_result_does_not_depend_on_chunk_boundaries(self):
        audio = np.concatenate([speech(6), silence(0.25), speech(4), silence(0.20), speech(5)])
        text = '가나다라마바. 사아자차? 카타파하가.'
        whole, _ = run(text, audio)
        rng = np.random.default_rng(0)
        for _ in range(5):
            sizes = rng.integers(37, 9000, size=200).tolist()
            pieces, _ = run(text, audio, sizes)
            np.testing.assert_array_equal(pieces, whole)

    def test_original_samples_are_kept_in_order(self):
        audio = np.concatenate([speech(5), silence(0.30), speech(5)])
        out, stretcher = run('가나다라마. 바사아자차.', audio)
        start, original, extra = stretcher.inserted[0]
        cut = int(round((start + original) * 100)) * (SR // 100)
        restored = np.concatenate([out[:cut], out[cut + int(round(extra * 100)) * (SR // 100):]])
        np.testing.assert_array_equal(restored, audio)

    def test_long_enough_pause_is_left_alone(self):
        audio = np.concatenate([speech(5), silence(0.70), speech(5)])
        out, stretcher = run('가나다라마. 바사아자차.', audio)
        self.assertEqual(stretcher.inserted, [])
        np.testing.assert_array_equal(out, audio)

    def test_too_short_pause_is_left_alone(self):
        audio = np.concatenate([speech(5), silence(0.05), speech(5)])
        _, stretcher = run('가나다라마. 바사아자차.', audio)
        self.assertEqual(stretcher.inserted, [])

    def test_leading_and_trailing_silence_are_left_alone(self):
        audio = np.concatenate([silence(0.40), speech(5), silence(0.40)])
        out, stretcher = run('가나다라마.', audio)
        self.assertEqual(stretcher.inserted, [])
        np.testing.assert_array_equal(out, audio)

    def test_each_sentence_end_is_used_once(self):
        audio = np.concatenate([speech(5), silence(0.20), speech(0.3), silence(0.20), speech(5)])
        _, stretcher = run('가나다라마. 바사아자차.', audio)
        self.assertEqual(len(stretcher.inserted), 1)


if __name__ == '__main__':
    unittest.main()
