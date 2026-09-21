import unittest

from soomac_irc.stt_hotword import normalize_stt_text


class TestSttHotword(unittest.TestCase):
    def test_menu_words_are_normalized(self):
        self.assertEqual(
            normalize_stt_text('페파론치노랑 소세지, 넙적면으로 줘'),
            '페퍼론치노랑 소시지, 넓은면으로 줘')

    def test_wide_noodle_variants_use_order_canonical(self):
        self.assertEqual(normalize_stt_text('넙적면'), '넓은면')
        self.assertEqual(normalize_stt_text('넓적면'), '넓은면')
        self.assertEqual(normalize_stt_text('넓적 면'), '넓은면')

    def test_observed_amount_request_is_normalized(self):
        self.assertEqual(
            normalize_stt_text('페파론치노 적당히 저.'),
            '페퍼론치노 적당히 줘.')

    def test_menu_only_request_is_normalized(self):
        self.assertEqual(normalize_stt_text('양파 만조'), '양파만 줘')

    def test_unrelated_high_tide_word_is_preserved(self):
        self.assertEqual(
            normalize_stt_text('오늘 만조 시간이 언제야?'),
            '오늘 만조 시간이 언제야?')

    def test_ambiguous_negation_is_preserved(self):
        self.assertEqual(
            normalize_stt_text('느끼한 게 없고 싶어'),
            '느끼한 게 없고 싶어')

    def test_canonical_order_is_unchanged(self):
        self.assertEqual(
            normalize_stt_text('토마토 소스에 얇은면과 넓은면이 있어?'),
            '토마토 소스에 얇은면과 넓은면이 있어?')


if __name__ == '__main__':
    unittest.main()
