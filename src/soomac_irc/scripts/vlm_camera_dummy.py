#!/usr/bin/env python3

import json
import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int16, String


CAMERA_TOPIC = "/camera/camera/color/image_raw"


class VLMCameraDummy(Node):
    def __init__(self):
        super().__init__("soomac_vlm_camera_dummy")

        self.camera_count = 0
        self.current_plan = None

        # 더미가 UI, STT, main 역할을 대신한다.
        self.ui_start_pub = self.create_publisher(String, "/ui/start", 10)
        self.stt_pub = self.create_publisher(String, "/stt_question", 10)
        self.next_pub = self.create_publisher(Int16, "/llm/next", 10)

        # 더미는 현재 RealSense가 실제로 발행하는 color 영상을 직접 구독한다.
        self.camera_sub = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.camera_callback,
            qos_profile_sensor_data,
        )
        self.plan_sub = self.create_subscription(String, "/llm/plan", self.plan_callback, 10)
        self.vlm_result_sub = self.create_subscription(String, "/vlm/result", self.vlm_result_callback, 10)
        self.reply_sub = self.create_subscription(String, "/llm_response", self.reply_callback, 10)
        self.stt_enable_sub = self.create_subscription(Bool, "/stt/enable", self.stt_enable_callback, 10)

        self.get_logger().info("VLM 카메라 더미 시작")
        self.get_logger().info(f"카메라 대기 : {CAMERA_TOPIC}")

        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

    def camera_callback(self, message: Image):
        self.camera_count += 1

        # 카메라 프레임마다 로그를 찍으면 너무 많아서 첫 장과 60장마다 출력한다.
        if self.camera_count == 1 or self.camera_count % 60 == 0:
            self.get_logger().info(
                f"[CAMERA] count={self.camera_count}, "
                f"size={message.width}x{message.height}, "
                f"encoding={message.encoding}"
            )

    def plan_callback(self, message: String):
        try:
            self.current_plan = json.loads(message.data)
        except json.JSONDecodeError:
            self.current_plan = message.data

        self.get_logger().info(f"[PLAN] {self.current_plan}")
        self.get_logger().info("재료 작업과 홈 복귀가 끝나면 터미널에 next 입력")

    def vlm_result_callback(self, message: String):
        try:
            result = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"[VLM RESULT] JSON 오류 : {message.data}")
            return

        self.get_logger().info(f"[VLM RESULT] {result}")

        if result.get("success") is True:
            self.get_logger().info(
                f"[MAIN] {result.get('class')} 성공, class·repeat_count 초기화"
            )
            self.current_plan = None

        else:
            self.get_logger().warning(
                f"[MAIN] {result.get('class')} 실패, 재발행되는 /llm/plan 대기"
            )

    def reply_callback(self, message: String):
        self.get_logger().info(f"[LLM] {message.data}")

    def stt_enable_callback(self, message: Bool):
        self.get_logger().info(f"[STT ENABLE] {message.data}")

    def input_loop(self):
        self.print_help()

        while rclpy.ok():
            try:
                command = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                return

            if command == "ui":
                self.ui_start_pub.publish(String(data="start"))
                print("[발행] /ui/start = start")

            elif command == "next":
                if self.current_plan is not None and self.camera_count == 0:
                    print("[경고] 카메라 이미지가 아직 한 장도 들어오지 않았습니다.")
                    continue

                self.next_pub.publish(Int16(data=3))
                print("[발행] /llm/next = 3")

            elif command.startswith("say "):
                text = command[4:].strip()

                if not text:
                    print("[오류] say 뒤에 문장을 입력하세요.")
                    continue

                self.stt_pub.publish(String(data=text))
                print(f"[발행] /stt_question = {text}")

            elif command == "camera":
                print(f"[카메라] 받은 프레임 수 = {self.camera_count}")

            elif command == "plan":
                print(f"[현재 계획] {self.current_plan}")

            elif command == "help":
                self.print_help()

            elif command in ("quit", "exit"):
                rclpy.shutdown()
                return

            elif command:
                print("[오류] 모르는 명령입니다. help를 입력하세요.")

    def print_help(self):
        print()
        print("사용 명령")
        print("  ui              : /ui/start 발행")
        print("  next            : /llm/next=3 발행")
        print("  say 문장        : /stt_question 발행")
        print("  camera          : 받은 카메라 프레임 수 확인")
        print("  plan            : 현재 /llm/plan 확인")
        print("  help            : 명령 다시 보기")
        print("  quit            : 종료")
        print()


def main(args=None):
    rclpy.init(args=args)
    node = VLMCameraDummy()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
