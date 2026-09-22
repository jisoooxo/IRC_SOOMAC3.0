# VLM 작업 판정 프롬프트 정본. 문자열 생성만 하고 이미지·로봇 상태는 변경하지 않는다.

SAUCE_NAMES = ("토마토", "오일", "크림")

VERDICT_RULE = (
    "먼저 한두 문장으로 실제로 확인한 시각 근거를 설명한다. "
    "마지막 줄에는 반드시 판정: PASS, 판정: FAIL, "
    "판정: UNCERTAIN 중 하나만 단독으로 출력한다."
)


def build_lid_prompt(image_count: int) -> tuple[str, str]:
    # 입력: 작업 후 이미지 수. 반환: system_prompt, user_text.
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
    # 입력: 소스 class, 작업 후 이미지 수. 반환: system_prompt, user_text.
    # 호출자: vlm.build_vlm_request. 소스 종류 검사는 호출자가 수행한다.
    sauce_label_hints = {
        "토마토": "tomato, pomodoro, Napoli, 나폴리, 토마토",
        "오일": "oil, olio, aglio, 오일",
        "크림": "cream, panna, 크림",
    }
    expected_hints = sauce_label_hints[expected]
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
    # 입력: 재료 class, 작업 후 이미지 수, 비교 이미지 유무·출처.
    # 반환: system_prompt, user_text. 호출자: vlm.build_vlm_request.
    if not has_comparison:
        system_prompt = (
            "너는 로봇의 식재료 투입 작업을 확인하는 시각 판정기다. "
            f"{expected}가 얇은면, 넓적면일 경우, 파스타면으로 인식되어도 면이 맞다." 
            "이미지1은 현재 목표 재료의 참고 이미지이고 이후 이미지는 작업 후 장면이다. "
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
    system_prompt = (
        "너는 로봇 작업 전후의 실제 재료 변화를 확인하는 시각 판정기다. "
        "검은색 도시락통 가운데를 기준으로 왼쪽 상단에는 소시지칸, 오른쪽 상단에는 양파 및 버섯칸(이하 야채 칸)이 있다."
        "소시지칸 바로 밑에는 치즈를 담는 칸(왼쪽), 페퍼론치노를 담는 칸(오른쪽)이 있으며 야채칸 아래에는 게살을 담는 칸이 있다."
        "마지막으로, 도시락통의 제일 아래쪽에는 파스타 면을 놓는 칸이 있다."
        "이미지1은 현재 목표 재료의 참고 이미지이고 이미지2는 이번 작업 전 도시락이다. "
        "페퍼론치노의 경우 고추 씨앗 같이 생겼으며, 잘게 갈려있는 형상이거나 이미지이다. "
        "이후 이미지는 이번 작업 후 장면이다. "
        f"작업 전과 비교해 {expected} 재료가 새로 보이거나 해당 재료 영역이 증가했으면 PASS이다. "
        f"기대 변화가 없거나 {expected}가 아닌 다른 재료가 추가됐으면 FAIL이다. "
        "화면의 bounding box, class 글자, 체크 표시 같은 overlay는 실제 변화의 증거로 사용하지 않는다. "
        f"가림이나 흔들림 때문에 비교할 수 없으면 UNCERTAIN이다. {VERDICT_RULE}"
    )
    user_text = (
        f"비교 이미지 출처는 {comparison_label}이다. "
        f"작업 후 이미지 {image_count}장에서 "
        f"{expected} 투입 변화가 실제로 발생했는지 판정해."
    )
    return system_prompt, user_text
