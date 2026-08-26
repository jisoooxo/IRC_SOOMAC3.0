#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int16, String


# scripts 폴더에서 실행해도 soomac_irc 안의 vlm_rag.py를 찾을 수 있게 한다.
PROJECT_CODE = Path(__file__).resolve().parents[1] / "soomac_irc"
sys.path.insert(0, str(PROJECT_CODE))

from vlm_rag import VlmRag


TEST_CAMERA_TOPIC = "/vlm_test/image_raw"
EXPECTED_CLASS = "넓적면"
MODEL_TIMEOUT = 60


class VLMReferenceScenario(Node):
    def __init__(self):
        super().__init__("soomac_vlm_reference_scenario")

        self.responses = []
        self.plans = []
        self.vlm_results = []
        self.stt_enabled = None

        self.camera_pub = self.create_publisher(Image, TEST_CAMERA_TOPIC, qos_profile_sensor_data)
        self.ui_start_pub = self.create_publisher(String, "/ui/start", 10)
        self.stt_pub = self.create_publisher(String, "/stt_question", 10)
        self.next_pub = self.create_publisher(Int16, "/llm/next", 10)

        self.create_subscription(String, "/llm_response", self.response_callback, 10)
        self.create_subscription(String, "/llm/plan", self.plan_callback, 10)
        self.create_subscription(String, "/vlm/result", self.vlm_result_callback, 10)
        self.create_subscription(Bool, "/stt/enable", self.stt_enable_callback, 10)

        self.camera_message = self.make_reference_message()
        self.camera_timer = self.create_timer(0.1, self.publish_reference_image)

    def make_reference_message(self):
        # Chroma에서 대표 이미지를 가져와 ROS Image 메시지로 바꾼다.
        reference_image = VlmRag().get_reference(EXPECTED_CLASS)

        if reference_image is None:
            raise RuntimeError(f"{EXPECTED_CLASS} 참고 이미지를 찾지 못함")

        reference_image = reference_image.convert("RGB")

        message = Image()
        message.height = reference_image.height
        message.width = reference_image.width
        message.encoding = "rgb8"
        message.is_bigendian = False
        message.step = reference_image.width * 3
        message.data = reference_image.tobytes()

        return message

    def publish_reference_image(self):
        self.camera_message.header.stamp = self.get_clock().now().to_msg()
        self.camera_pub.publish(self.camera_message)

    def response_callback(self, message: String):
        self.responses.append(message.data)
        print(f"[LLM] {message.data}")

    def plan_callback(self, message: String):
        try:
            plan = json.loads(message.data)
        except json.JSONDecodeError:
            plan = {"raw": message.data}

        self.plans.append(plan)
        print(f"[PLAN] {plan}")

    def vlm_result_callback(self, message: String):
        try:
            result = json.loads(message.data)
        except json.JSONDecodeError:
            result = {"raw": message.data}

        self.vlm_results.append(result)
        print(f"[VLM RESULT] {result}")

    def stt_enable_callback(self, message: Bool):
        self.stt_enabled = message.data
        print(f"[STT ENABLE] {message.data}")

    def wait_for(self, condition, timeout, error_message):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            if condition():
                return

        raise TimeoutError(error_message)

    def run(self):
        print(f"[TEST CAMERA] {TEST_CAMERA_TOPIC}에 {EXPECTED_CLASS} 참고 이미지 발행 중")
        print("LLM 노드 연결 대기")

        self.wait_for(
            lambda: self.camera_pub.get_subscription_count() > 0,
            15,
            "LLM이 테스트 카메라 토픽을 구독하지 않음",
        )

        self.ui_start_pub.publish(String(data="start"))
        print("[UI] 주문 시작")

        self.wait_for(
            lambda: self.stt_enabled is False,
            10,
            "UI 시작 뒤 STT 닫기 신호가 오지 않음",
        )

        self.next_pub.publish(Int16(data=3))
        print("[main] 최초 /llm/next = 3")

        self.wait_for(
            lambda: self.stt_enabled is True,
            10,
            "주문 시작 뒤 STT 열기 신호가 오지 않음",
        )

        response_count = len(self.responses)
        self.stt_pub.publish(String(data="크림소스 넓적면 보통으로 해줘"))
        print("[손님] 크림소스 넓적면 보통으로 해줘")

        self.wait_for(
            lambda: len(self.responses) > response_count,
            MODEL_TIMEOUT,
            "주문 반영 응답 시간 초과",
        )

        self.stt_pub.publish(String(data="다 골랐어"))
        print("[손님] 다 골랐어")

        self.wait_for(
            lambda: any(plan.get("class") == EXPECTED_CLASS for plan in self.plans),
            MODEL_TIMEOUT,
            "넓적면 작업 계획이 발행되지 않음",
        )

        # 테스트 이미지가 LLM의 최신 카메라 메시지로 들어갈 시간을 준다.
        end_time = time.monotonic() + 1

        while time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.next_pub.publish(Int16(data=3))
        print("[main] 작업 완료 /llm/next = 3")

        self.wait_for(
            lambda: any(result.get("class") == EXPECTED_CLASS for result in self.vlm_results),
            MODEL_TIMEOUT,
            "VLM 결과 시간 초과",
        )

        result = next(
            result for result in reversed(self.vlm_results)
            if result.get("class") == EXPECTED_CLASS
        )

        assert result.get("success") is True, result
        assert result.get("status") == "pass", result

        print()
        print("PASS: Chroma 참고 이미지 → ROS Image → Gemma VLM → /vlm/result")


def main(args=None):
    rclpy.init(args=args)
    node = VLMReferenceScenario()

    try:
        node.run()
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
