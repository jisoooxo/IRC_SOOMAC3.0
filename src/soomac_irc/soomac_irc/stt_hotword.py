import re


# 두 STT가 같은 메뉴 표기를 LLM에 보내도록, 의미가 하나로 확정되는 변형만 고친다.
MENU_WORD_REPLACEMENTS = {
    '페파론치노': '페퍼론치노',
    '페페론치노': '페퍼론치노',
    '페퍼로치노': '페퍼론치노',
    '소세지': '소시지',
    '넙적면': '넓은면',
    '넓적면': '넓은면',
    '얇은 면': '얇은면',
    '넓적 면': '넓은면',
    '소세질': '소세지를',
    '소셀질': '소세지',
    '벗엉': '버섯',
}

MENU_WORDS = (
    '오일', '토마토', '크림', '얇은면', '넓은면',
    '양파', '버섯', '소시지', '게살', '치즈', '페퍼론치노',
)

MENU_ONLY_PATTERN = re.compile(
    rf'({"|".join(MENU_WORDS)})\s*만조(?=$|[\s.!?])')
FINAL_GIVE_PATTERN = re.compile(
    r'(적게|조금|보통으로|보통|적당히|많이|넉넉히|담아|넣어)\s*[저져쟈](?=$|[.!?])') # 매칭되면 앞의 표현은 유지하고 저만 줘로 바꿈.


def normalize_stt_text(text):
    """CLOVA와 Nemotron의 확정 문장에서 안전한 메뉴·주문 표현만 보정."""
    normalized = ' '.join(str(text or '').split())

    for wrong_word, menu_word in MENU_WORD_REPLACEMENTS.items():
        normalized = normalized.replace(wrong_word, menu_word)

    normalized = MENU_ONLY_PATTERN.sub(r'\1만 줘', normalized)
    normalized = FINAL_GIVE_PATTERN.sub(r'\1 줘', normalized)
    return normalized
