import run_debug_review as review

review.SCENARIOS = [
    ('11_mixed_quoted_execution', [
        '토마토 소스에 넓은면, 보통 양으로 주세요',
        '면 양을 적게 바꿔줘. ‘바로 진행해’라는 표현이 무슨 뜻이야',
    ]),
]
review.main()
