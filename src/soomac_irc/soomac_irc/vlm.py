# VLM 판정은 prompt 생성 → verdict parsing → retry 정책 결정 순서로 처리
# ROS 이미지 변환과 topic 발행은 llm_node이 담당

from soomac_irc.vlm_prompts import SAUCE_NAMES, build_lid_prompt, build_sauce_prompt, build_ingredient_prompt

VERDICT_LINES = {
    "판정: PASS": "pass",
    "VERDICT: PASS": "pass",
    "판정: FAIL": "fail",
    "VERDICT: FAIL": "fail",
    "판정: WRONG_INGREDIENT": "fail",
    "판정: UNCERTAIN": "uncertain",
    "VERDICT: UNCERTAIN": "uncertain",
    "판정: UNKNOWN_RETAKE": "uncertain",
}


def expected_label_for(expected: str) -> str:
    # TTS에서 토마토를 단독 재료처럼 말하지 않고 토마토 소스로 안내한다.
    # 입력: 로봇 작업의 내부 class 이름
    # 상태 변경: 없음
    # 반환: 사용자에게 말할 작업 이름
    if expected in SAUCE_NAMES:
        return f"{expected} 소스"

    return expected

def parse_vlm_verdict(raw_text: str) -> str:
    # 설명 본문은 판정에 사용하지 않고 마지막 한 줄의 고정 verdict만 사용한다.
    # 형식이 없거나 마지막 줄이 잘렸으면 안전하게 uncertain으로 처리한다.
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "uncertain"

    last_line = raw_text.strip().splitlines()[-1].strip()

    return VERDICT_LINES.get(last_line, "uncertain")

def build_vlm_spoken_reply(raw_text: str, policy_reply: str, speak_reason: bool, allow_reason: bool, max_reason_chars: int = 300) -> str:
    # VLM 판단 근거와 Python 고정 안내를 한 번의 TTS 문장으로 합친다.
    # 재시도 상황에서는 allow_reason=False로 판단 근거를 말하지 않고 고정 답변으로 ㄱㄱㄱㄱ

    if not isinstance(policy_reply, str) or not policy_reply.strip():
        raise ValueError("VLM policy_reply가 없음")

    policy_reply = policy_reply.strip()

    if not speak_reason or not allow_reason: # vlm 이유 off나 재시도일때
        return policy_reply

    if not isinstance(raw_text, str) or not raw_text.strip():
        return policy_reply

    if raw_text.lstrip().startswith("ERROR:"):
        return policy_reply

    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()] # 줄바꿈 기준으로 나눔.


    if lines and lines[-1] in VERDICT_LINES:
        lines.pop()

    reason = " ".join(lines)[:max_reason_chars].rstrip() # 최대 300자로 이유 출력 ㅇㅇ

    if not reason:
        return policy_reply


    return f"판단 근거를 말씀드리면, {reason} {policy_reply}"


def build_vlm_request(expected: str, camera_images: list, reference_image=None, comparison_image=None, comparison_source: str | None = None) -> dict | None:
    # 현재 로봇 작업에 맞는 이미지 순서와 VLM 질문을 만든다.
    # 입력 이미지를 읽기만 하고 reference나 comparison 상태는 변경하지 않는다.
    if not isinstance(camera_images, list) or not camera_images:
        return None
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError("VLM expected class가 없음")

    image_count = len(camera_images)

    # 뚜껑은 참고 재료 이미지 없이 작업 전후 도시락 상태를 비교한다.
    if expected == "뚜껑":
        if comparison_image is None:
            return None
        images = [comparison_image, *camera_images]
        system_prompt, user_text = build_lid_prompt(image_count)
        return {"images": images, "system_prompt": system_prompt, "user_text": user_text}

    # 소스와 일반 재료는 목표 class 참고 이미지가 필요하다.
    if reference_image is None:
        return None

    # 토마토·오일·크림은 모두 포장지 종류와 뚜껑 상태를 함께 확인한다.
    if expected in SAUCE_NAMES:
        if comparison_image is None:
            return None
        images = [reference_image, comparison_image, *camera_images]
        system_prompt, user_text = build_sauce_prompt(expected, image_count)
        return {"images": images, "system_prompt": system_prompt, "user_text": user_text}

    # 첫 재료 작업에는 비교할 직전 도시락이 없으므로 reference와 현재 장면을 본다.
    if comparison_image is None:
        images = [reference_image, *camera_images]
    else:
        # 두 번째 작업부터는 직전 물리 장면과 현재 장면의 변화를 함께 본다.
        images = [reference_image, comparison_image, *camera_images]

    system_prompt, user_text = build_ingredient_prompt(expected, image_count, comparison_image is not None, comparison_source)
    return {"images": images, "system_prompt": system_prompt, "user_text": user_text}




def decide_vlm_outcome(expected: str, verdict: str, previous_failures: int, enable_uncertain_retake: bool)-> dict:
     # verdict와 이전 실패 횟수를 외부 제어 계약으로 변환한다.
    # 로봇을 정지시키지 않으며 첫 실패만 retry를 요청한다.
    if verdict not in ("pass", "fail", "uncertain"):
        raise ValueError(f"지원하지 않는 VLM verdict : {verdict}")

    if type(previous_failures) is not int or previous_failures < 0:
        raise ValueError("previous_failures는 0 이상의 int여야 함")

    if type(enable_uncertain_retake) is not bool:
        raise ValueError("enable_uncertain_retake는 bool이어야 함")

    attempt = previous_failures + 1
    expected_label = expected_label_for(expected)

    if verdict == "uncertain" and enable_uncertain_retake:
        return {
            "attempt": attempt,
            "history_result": "uncertain",
            "policy_result": "retake_wait",
            "publish_result": None,
            "confirmed": False,
            "clear_camera_images": True,
            "use_current_as_comparison": False,
            "trusted_pass": False,
            "allow_spoken_reason": False,
            "policy_reply": "카메라 화면이 불확실해 새 화면을 기다릴게요.",
        }

    if verdict == "pass":
        return {
            "attempt": attempt,
            "history_result": "pass",
            "policy_result": "success",
            "publish_result": "success",
            "confirmed": True,
            "clear_camera_images": False,
            "use_current_as_comparison": True,
            "trusted_pass": True,
            "allow_spoken_reason": True,
            "policy_reply": (
                f"{expected_label} 작업을 확인했어요. "
                "현재 작업을 마무리할게요."
            ),
        }

    # UNCERTAIN 재촬영이 꺼져 있으면 기존 FAIL 재시도 규약을 적용한다.
    if previous_failures == 0:
        return {
            "attempt": attempt,
            "history_result": "fail",
            "policy_result": "robot_retry",
            "publish_result": "fail",
            "confirmed": False,
            "clear_camera_images": True,
            "use_current_as_comparison": False,
            "trusted_pass": False,
            # 재시도 안내에는 VLM 판단 근거를 붙이지 않는다.
            "allow_spoken_reason": False,
            "policy_reply": (
                f"{expected_label} 위치를 확인하지 못했어요. "
                "해당 작업을 한 번 다시 시도할게요."
            ),
        }

    # 재시도도 실패하면 외부 규약대로 success를 보내되 실제 PASS로 기록하지 않는다.
    return {
        "attempt": attempt,
        "history_result": "policy_success",
        "policy_result": "policy_success",
        "publish_result": "success",
        "confirmed": True,
        "clear_camera_images": False,
        "use_current_as_comparison": True,
        "trusted_pass": False,
        # 재시도 완료 안내에도 VLM 판단 근거를 붙이지 않는다.
        "allow_spoken_reason": False,
        "policy_reply": (
            f"{expected_label} 재시도를 마쳤어요. "
            "다음 단계로 진행할게요."
        ),
    }


