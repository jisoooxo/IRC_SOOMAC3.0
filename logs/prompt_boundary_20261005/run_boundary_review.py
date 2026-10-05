"""Prompt-only revision: real debug-node tests with isolated application I/O."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prompt_review_20261005'))
import run_debug_review as review

review.SCENARIOS = [
    ('01_real_execution', [
        '토마토 소스에 넓은면, 보통 양으로 주세요',
        '바로 진행해',
    ]),
    ('02_uncertain_noodle', ['널적면으로 주세요']),
    ('03_normal_noodle_is_ambiguous', ['보통면으로 주세요']),
    ('04_unsupported_direct', ['떡볶이 줘']),
    ('05_tteokbokki_reference', ['떡볶이가 뭐야', '그거 줘']),
    ('06_ramen_reference', ['라면이 뭐야', '그거 줘']),
    ('07_supported_reference_control', ['양파가 뭐야', '그거 줘']),
    ('08_ham_is_not_sausage', [
        '토마토 소스에 넓은면, 보통 양으로 하고 소시지 넣어줘',
        '햄 빼줘',
    ]),
    ('09_explanation_then_mixed_explanation', [
        '토마토 소스에 넓은면, 보통 양으로 주세요',
        '바로 진행해라는 표현이 무슨 뜻이야',
        '면 양을 적게 바꿔줘. 바로 진행해라는 표현이 무슨 뜻이야',
    ]),
    ('10_real_mixed_execution_control', [
        '토마토 소스에 넓은면, 보통 양으로 주세요',
        '면 양을 적게 바꾸고 바로 진행해',
    ]),
    ('11_unsupported_mixed', ['떡볶이 줘. 그리고 떡볶이가 뭐야']),
    ('12_multiple_reference_candidates', ['떡볶이랑 라면은 어떻게 달라', '그거 줘']),
    ('13_latest_unsupported_reference', ['양파가 뭐야', '떡볶이가 뭐야', '그거 줘']),
    ('14_original_mixed_misspelling', [
        '토마토 소스로 널적면에 보통량으로 근데 토마토소스는 역사가 어떻게 돼?',
    ]),
]
if __name__ == '__main__':
    review.main()
