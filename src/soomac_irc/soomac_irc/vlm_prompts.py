# VLM 작업 판정 프롬프트 정본. 문자열 생성만 하고 이미지·로봇 상태는 변경하지 않는다.

from soomac_irc.domain import SAUCES


SAUCE_NAMES = SAUCES

VERDICT_RULE = (
    "먼저 한두 문장으로 실제로 확인한 시각 근거를 설명한다. "
    "마지막 줄에는 반드시 판정: PASS, 판정: FAIL, "
    "판정: UNCERTAIN 중 하나만 단독으로 출력한다."
)

# 재료별 (도시락 칸, 생김새). 참고 이미지는 공급 용기를 확대한 사진이라 실제 도시락 칸과 모습이 다르다.
# 생김새는 vlm_data/mealkit_coco_v15 참고 crop을 직접 보고 적었다(2026-10-26).
INGREDIENT_GUIDES = {
    "양파": ("오른쪽 상단 야채칸", "흰색~반투명의 썬 양파 조각(작은 조각부터 층진 큰 조각까지)"),
    "버섯": ("오른쪽 상단 야채칸", "가늘고 긴 밝은 줄기에 갈색 갓이 달린 버섯"),
    "소시지": ("왼쪽 상단 소시지칸", "손가락 길이의 매끈한 원통형 통소시지이고 주황빛이 도는 갈색"),
    "게살": ("야채칸 아래 게살칸", "겉은 주황·빨강이고 속은 흰 막대 모양(맛살)"),
    "치즈": ("소시지칸 아래 왼쪽 치즈칸", "흰색~연노랑의 잘게 썬 조각"),
    "페퍼론치노": ("소시지칸 아래 오른쪽 페퍼론치노칸", "붉은·갈색의 잘게 부순 고추 조각과 노란 씨앗"),
    "얇은면": ("맨 아래 면 칸", "노란 건조 파스타 면 다발(분홍 띠로 묶여 있을 수 있음)"),
    # 넓은면 참고 사진은 따로 없다(vlm_rag.py에서 면 3종이 같은 noodle). 폭·종류는 구별하지 않는다.
    "넓은면": ("맨 아래 면 칸", "노란 건조 파스타 면"),
    "넓적면": ("맨 아래 면 칸", "노란 건조 파스타 면"),
}
# 사용자 결정(2026-10-26): 소시지는 들어갔는지만 본다. 다른 재료의 판정 기준은 바꾸지 않는다.
INSERTION_ONLY_INGREDIENTS = frozenset({"소시지"})


def _has_final_consonant(word: str) -> bool:
    last = word[-1]
    return "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0


def _topic(word: str) -> str:
    return f"{word}{'은' if _has_final_consonant(word) else '는'}"


def _subject(word: str) -> str:
    return f"{word}{'이' if _has_final_consonant(word) else '가'}"


def _ingredient_guide(expected: str) -> str:
    # 생김새·칸 안내. 표에 없는 재료는 기존 프롬프트만 쓴다.
    guide = INGREDIENT_GUIDES.get(expected)
    if guide is None:
        return ""
    compartment, appearance = guide
    return (
        f"{_topic(expected)} {appearance}이다. 판정은 검은 도시락통의 {compartment}만 본다. "
        "참고 이미지는 재료 공급 용기를 확대한 사진이라 도시락 칸에서는 더 적고 작게 보인다. "
        "용기 모양이나 양이 아니라 생김새만 참고하고, 화면에 함께 보이는 공급 용기 속 재료는 판정에 쓰지 않는다. "
    )


def build_lid_prompt(image_count: int) -> tuple[str, str]:
    # 작업 후 이미지 개수. 
    # system_prompt, user_text.
    # 호출자: vlm.build_vlm_request.
    system_prompt = (
        "너는 도시락의 검은 뚜껑 닫기 작업만 확인하는 시각 판정기다. "
        "이미지1은 뚜껑 작업 전 도시락이고 이후 이미지는 작업 후 장면이다. "
        "음식이나 재료의 종류가 아니라 검은 뚜껑의 실제 위치를 먼저 확인한다. "
        "작업 후 이미지 중 하나 이상에서 검은 뚜껑이 도시락 용기 전체를 정상적으로 덮고 있으면 PASS이다. "
        "뚜껑이 열려 있거나, 비스듬히 걸쳐 있거나, 일부 음식이 계속 노출되면 FAIL이다. "
        "화면의 bounding box, class 글자, 체크 표시 같은 overlay는 실제 뚜껑의 증거로 사용하지 않는다. "
        f"가림, 흔들림, 반사 때문에 뚜껑 상태를 확인할 수 없으면 UNCERTAIN이다. {VERDICT_RULE}"
    )
    user_text = (
        f"작업 후 이미지 {image_count}장에서 검은 뚜껑이 "
        "도시락 용기 전체를 실제로 닫았는지 판정해."
    )
    return system_prompt, user_text


def build_sauce_prompt(expected: str, image_count: int) -> tuple[str, str]:
    # input : 소스 class, 작업 후 이미지 수. 
    # return : system_prompt, user_text.

    sauce_label_hints = {
        "토마토": "tomato, pomodoro, Napoli, 나폴리, 토마토",
        "오일": "oil, olio, aglio, 오일",
        "크림": "cream, panna, 크림",
    } # 동일하게 처리할 라벨
    expected_hints = sauce_label_hints[expected] # expected는 우리가 대상으로 해야할 소스 이름 ㅇㅇ
    other_sauces = [sauce for sauce in SAUCE_NAMES if sauce != expected]
    other_hints = "; ".join(f"{sauce} 소스={sauce_label_hints[sauce]}" for sauce in other_sauces)
    system_prompt = (
        "너는 닫힌 도시락 위에 놓인 밀봉 소스 포장지의 종류를 확인하는 시각 판정기다. "
        "판정 대상은 소스 내용물이나 완성된 음식이 아니라 실제 밀봉 포장지다. "
        "이미지1은 주문한 소스 포장지의 참고 이미지이고 이미지2는 소스 작업 전 도시락이다. "
        "이후 이미지는 소스 작업 후 장면이다. "
        "먼저 작업 후 검은 뚜껑이 실제로 닫혀 있는지 확인하고, 그다음 뚜껑 위 포장지 종류를 확인한다. "
        "포장지에 인쇄된 파스타·토마토·음식 사진을 열린 도시락이나 실제 음식으로 해석하지 않는다. "
        "화면의 bounding box, class 글자, 체크 표시 같은 overlay는 실제 물체의 증거로 사용하지 않는다. "
        f"목표는 {expected} 소스이며 {expected_hints} 표기를 같은 종류로 인정한다. "
        f"다른 소스 표기는 {other_hints}이다. "
        f"작업 후 이미지 중 하나 이상에서 검은 뚜껑이 닫혀 있고, 그 위에 밀봉 포장지가 있으며, "
        f"포장지의 글자·그림·디자인이 참고 이미지 또는 {expected} 소스 표기와 일치할 때만 PASS이다. "
        "포장 색상 하나만으로 소스 종류를 단정하지 않는다. "
        "브랜드, 포장 방향, 크기, 세부 디자인 차이만으로 FAIL하지 않는다. "
        f"포장지가 없거나, 검은 뚜껑이 실제로 열려 있거나, 포장지가 {expected}가 아닌 "
        f"{' 또는 '.join(other_sauces)} 소스로 명확히 확인되면 FAIL이다. "
        f"뚜껑 상태나 포장지 종류를 확실히 확인할 수 없으면 UNCERTAIN이다. {VERDICT_RULE}"
    )

    user_text = (
        f"작업 후 이미지 {image_count}장에서 검은 뚜껑이 닫혀 있고, "
        f"그 위에 밀봉된 {expected} 소스 포장지가 실제로 놓였는지 판정해. "
        "다른 소스 포장지와 혼동하지 말고 포장지의 인쇄 글자와 참고 이미지를 함께 확인해."
    )
    return system_prompt, user_text


def build_ingredient_prompt(expected: str, image_count: int, has_comparison: bool, comparison_source: str | None = None) -> tuple[str, str]:
    # input: 재료 class, 작업 후 이미지 수, 비교 이미지 유무·출처.
    # output: system_prompt, user_text. 
    if not has_comparison:
        system_prompt = (
            "너는 로봇의 식재료 투입 작업을 확인하는 시각 판정기다. "
            f"{_subject(expected)} 얇은면, 넓적면일 경우, 파스타면으로 인식되어도 면이 맞다."
            "이미지1은 현재 목표 재료의 참고 이미지이고 이후 이미지는 작업 후 장면이다. "
            f"{_ingredient_guide(expected)}"
            f"작업 후 장면에 {expected} 재료가 명확하게 보이면 PASS이다. "
            f"{expected} 재료가 없거나 명확히 다른 재료가 담겼으면 FAIL이다. "
            "화면의 bounding box, class 글자, 체크 표시 같은 overlay는 실제 재료의 증거로 사용하지 않는다. "
            f"가림이나 흔들림 때문에 재료를 확인할 수 없으면 UNCERTAIN이다. {VERDICT_RULE}"
        )
        user_text = (
            f"작업 후 이미지 {image_count}장에서 "
            f"{expected} 재료가 실제로 담겼는지 판정해."
        )
        return system_prompt, user_text

    comparison_label = comparison_source or "unknown"
    insertion_rule = (
        "한두 조각이라도 새로 들어갔으면 PASS이며, 양이 주문량에 맞는지는 판정하지 않는다. "
        if expected in INSERTION_ONLY_INGREDIENTS else ""
    )
    system_prompt = (
        "너는 로봇 작업 전후의 실제 재료 변화를 확인하는 시각 판정기다. "
        "검은색 도시락통 가운데를 기준으로 왼쪽 상단에는 소시지칸, 오른쪽 상단에는 양파 및 버섯칸(이하 야채 칸)이 있다."
        "소시지칸 바로 밑에는 치즈를 담는 칸(왼쪽), 페퍼론치노를 담는 칸(오른쪽)이 있으며 야채칸 아래에는 게살을 담는 칸이 있다."
        "마지막으로, 도시락통의 제일 아래쪽에는 파스타 면을 놓는 칸이 있다."
        "이미지1은 현재 목표 재료의 참고 이미지이고 이미지2는 이번 작업 전 도시락이다. "
        "이후 이미지는 이번 작업 후 장면이다. "
        f"{_ingredient_guide(expected)}"
        f"작업 전과 비교해 {expected} 재료가 새로 보이거나 해당 재료 영역이 증가했으면 PASS이다. "
        f"{insertion_rule}"
        "근거에는 해당 칸의 작업 전 모습과 작업 후 모습을 비교해 적는다. "
        f"기대 변화가 없거나 {_subject(expected)} 아닌 다른 재료가 추가됐으면 FAIL이다. "
        "화면의 bounding box, class 글자, 체크 표시 같은 overlay는 실제 변화의 증거로 사용하지 않는다. "
        f"가림이나 흔들림 때문에 비교할 수 없으면 UNCERTAIN이다. {VERDICT_RULE}"
    )
    user_text = (
        f"비교 이미지 출처는 {comparison_label}이다. "
        f"작업 후 이미지 {image_count}장에서 "
        f"{expected} 투입 변화가 실제로 발생했는지 판정해."
    )
    return system_prompt, user_text
