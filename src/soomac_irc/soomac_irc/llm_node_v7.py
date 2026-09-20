import copy
import json
import os
import queue
import re
import threading
from collections import deque
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Int16, String

from soomac_irc.agent_v7 import build_graph, build_selection_next_step, classify_confirmation_intent, commit_section_skip
from soomac_irc.call_model_v7 import load_model, make_call_model, make_call_vlm, make_generate_reply
from soomac_irc.order_v7 import SECTION_LABELS, SECTION_ORDER, build_section_plan, new_order, next_section, strongest_restriction_for
from soomac_irc.vlm_rag import VlmRag

ENABLE_VLM = True
ENABLE_VLM_UI_IMAGES = True  # 실제 VLM 입력 3분할 UI 발행
# 실물 없는 전체 사이클에서만 VLM OFF 우회를 UI 표시용 PASS로 기록
ENABLE_DUMMY_VLM_PASS = os.environ.get("SOOMAC_DUMMY_VLM_PASS") == "1"
ENABLE_TOOL_LORA = True
TOOL_ADAPTER_PATH = "/home/roma/ros2_ws/src/soomac_irc/finetune/v5_1/runs/gemma4_tool_lora_int8_v5_1_deterministic"

ENABLE_RUNTIME_LOG = True
RUNTIME_LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "soomac_runtime_logs"

BY_VLM_NUM = 3  # /llm/next에서 사용하는 기존 완료 코드

IMAGE_TOPIC_TYPE = CompressedImage  # raw 이미지면 Image로 변경
WORLD_CAM_TOPIC = "/vision/overlay_image"
VLM_UI_IMAGE_TOPIC = "/agent/vlm_snapshot" # ui에 보낼것
NUMBER_IMAGE_FOR_CONFIRM = 5

# False에서는 UNCERTAIN도 기존 robot retry 규약으로 처리
# True는 제어의 inspection pose 계약이 생긴 뒤 사용
ENABLE_UNCERTAIN_RETAKE = False

BY_VLM_NUM = 3

VLM_JUDGE_REASON_SPEAK = False

# Python 내부는 한글, /llm/plan 발행 직전에만 영문 class로 변환
TOPIC_CLASS_NAMES = {
    "얇은면": "noodle_thin",
    "넓은면": "noodle_thick",
    "양파": "onion",
    "버섯": "mushroom",
    "소시지": "sausage",
    "게살": "crab",
    "치즈": "cheese",
    "페퍼론치노": "pepperoncino",
    "뚜껑": "cover",
    "오일": "sauce_oil",
    "토마토": "sauce_tomato",
    "크림": "sauce_cream",
}


class LLMNode(Node):
    def __init__(self):
        super().__init__("soomac_llm_node_v7")

        # 주문 상태: 사용자가 선택한 메뉴와 restriction
        self.order = new_order()

        # 실행 상태: 실제 값은 order에만 저장하고 여기에는 진행 위치만 저장
        self.section = "noodle"
        self.selected = []           # 아직 plan으로 보내지 않은 선택 key
        self.task_queue = []         # plan 발행 전 작업
        self.active_task = None      # 현재 plan으로 발행한 작업
        self.completed_tasks = []    # 로봇 공정이 완료된 작업
        self.skipped_sections = {}   # section: 건너뛴 이유
        self.robot_started = False   # 첫 /llm/plan 발행 여부

        # 대화 정책 상태
        self.recommendation_state = {"phase": "idle", "confirmed_sections": []}
        self.last_recommendation = {}
        self.awaiting_confirm = None
        self.already_notified_safety_facts = set()

        # 시작·종료 상태
        self.ui_started = False
        self.pending_initial_next = False
        self.conversation_started = False
        self.order_finished = False

        # 모델에 전달할 최근 대화와 실제 처리 결과
        self.history = []
        self.action_history = []

        # 노드를 실행할 때마다 별도 runtime log 파일을 사용
        self.runtime_session_id = ""
        self.runtime_turn_index = 0
        self.runtime_event_index = 0

        # base model은 한 번만 올리고 Tool 호출에서도 같은 객체를 계속 사용함
        # False면 adapter 없이 nominal INT8 base만 사용함
        self.get_logger().info("모델 로딩중")
        self.model, self.processor = load_model(
            TOOL_ADAPTER_PATH if ENABLE_TOOL_LORA else None
        )

        # call_model: 사용자 발화 → Tool JSON
        # graph: Tool JSON → Python 검증 → transaction과 policy reply
        call_model = make_call_model(
            self.model,
            self.processor,
            self.get_logger(),
        )
        self.graph = build_graph(call_model)
        # respond일 때 Tool adapter를 끄고 일반 답변을 생성함
        self.generate_reply = make_generate_reply(self.model, self.processor)

        # VLM 상태는 주문 상태와 분리하며, 카메라 callback은 lock 안에서만 수정

        self.call_vlm = None
        self.vlm_rag = None
        self.vlm_reference = {}

        self.latest_camera_message = None
        self.vlm_camera_messages = deque(maxlen=NUMBER_IMAGE_FOR_CONFIRM)

        # 실제 PASS에서만 갱신하는 신뢰 가능한 성공 이미지
        self.previous_success_image = None

        # 다음 작업이 비교할 직전 물리 장면
        # source는 실제 성공이면 pass, 재시도 뒤 정책 통과면 retried
        self.comparison_image = None
        self.comparison_source = None
        self.comparison_task_class = None

        self.verification_history = []
        self.vlm_confirmed = False
        self.camera_lock = threading.Lock()
        self.camera_sub = None

        if ENABLE_VLM:
            self.call_vlm = make_call_vlm(
                self.model,
                self.processor,
                self.get_logger(),
            )
            self.vlm_rag = VlmRag()
        self.get_logger().info("모델 로딩 완료")

        # Publish: 처리 결과를 기존 ROS 계약 그대로 외부에 보냄
        runtime_file_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.runtime_log_path = RUNTIME_LOG_DIRECTORY / f"tool_turns_{runtime_file_id}.jsonl"
        self.runtime_event_log_path = RUNTIME_LOG_DIRECTORY / f"runtime_events_{runtime_file_id}.jsonl"

        if ENABLE_RUNTIME_LOG:
                    RUNTIME_LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
                    self.runtime_log_path.touch(exist_ok=False)
                    self.runtime_event_log_path.touch(exist_ok=False)
                    self.get_logger().info(f"Tool 로그 파일 : {self.runtime_log_path}")
                    self.get_logger().info(f"ROS 이벤트 로그 파일 : {self.runtime_event_log_path}")


        self.reply_pub = self.create_publisher(String, "/llm_response", 10)  # UI와 TTS가 사용할 답변
        self.plan_pub = self.create_publisher(String, "/llm/plan", 10)       # main이 받을 로봇 작업
        self.done_pub = self.create_publisher(Bool, "/llm/done", 10)         # 전체 주문 완료
        self.vlm_ui_image_pub = None

        if ENABLE_VLM and ENABLE_VLM_UI_IMAGES:
            self.vlm_ui_image_pub = self.create_publisher(CompressedImage, VLM_UI_IMAGE_TOPIC, 1)
        self.status_pub = self.create_publisher(String, "/agent/status", 10) # UI가 표시할 현재 상태
        self.stt_enable_pub = self.create_publisher(Bool, "/stt/enable", 10) # STT 입력 허용 여부
        self.vlm_result_pub = self.create_publisher(String, "/vlm/confirm", 10)  # VLM 판정 결과

        # callback에서는 모델을 돌리지 않고 작업 종류와 값만 저장함
        # worker 하나가 순서대로 꺼내므로 주문 상태가 동시에 수정되지 않음
        self._jobs = queue.Queue(maxsize=32)

        # Subscribe: main·UI·STT에서 사용하는 기존 토픽 계약
        self.stt_sub = self.create_subscription(String, "/stt_question", self.stt_callback, 10)
        self.ui_start_sub = self.create_subscription(String, "/ui/start", self.ui_start_callback, 10)
        self.ui_reset_sub = self.create_subscription(String, "/ui/reset", self.ui_reset_callback, 10)
        self.next_trigger_sub = self.create_subscription(Int16, "/llm/next", self.trigger_callback, 10)
        self.confirm_start_sub = self.create_subscription(Bool, "/llm/confirm_start", self.confirm_start_callback, 10)
        self.llm_reset_sub = self.create_subscription(String, "/llm/reset", self.llm_reset_callback, 10)


        if ENABLE_VLM:
            self.camera_sub = self.create_subscription(IMAGE_TOPIC_TYPE,WORLD_CAM_TOPIC, self.camera_callback, qos_profile_sensor_data)

        # callback이 넣은 작업은 이 worker 하나만 순서대로 처리한다.
        self._worker = threading.Thread(
            target=self.worker_loop,
            name="llm_worker_v4",
            daemon=True,
        )
        self._worker.start()


    def _enqueue_job(self, mode: str, data):
        # 큐가 가득 찼을 때 callback을 기다리게 하지 않는다.
        # 정상 토픽 빈도에서는 32개가 쌓이지 않으며, 넘치면 원인을 로그로 남긴다.
        try:
            self._jobs.put_nowait((mode, data))
        except queue.Full:
            self.get_logger().error(f"LLM 작업 큐가 가득 차서 {mode} 메시지를 받지 못함")

    def ui_start_callback(self, msg: String):
        # UI 시작 신호만 저장한다. greeting 시작 조건은 worker에서 판단한다.
        self._enqueue_job("start", msg.data)

    def stt_callback(self, msg: String):
        # 여기서 모델을 직접 호출하지 않는다.
        self._enqueue_job("turn", msg.data)

    def ui_reset_callback(self, msg: String):
        self._enqueue_job("reset", msg.data)

    def llm_reset_callback(self, msg: String):
        # main이 마지막 sauce 작업을 완료한 뒤 보내는 기존 종료 신호
        self._enqueue_job("finish", msg.data)

    def confirm_start_callback(self, msg: Bool):
        # 현재 active_task의 VLM 판단을 시작하라는 기존 신호
        self._enqueue_job("confirm", msg.data)

    def trigger_callback(self, msg: Int16):
        # main이 다음 작업 진행 시점에 보내는 /llm/next
        self._enqueue_job("next", msg.data)

    def camera_callback(self, msg: Image | CompressedImage):
        # 카메라 프레임은 worker queue에 계속 쌓지 않고 최근 5장만 보관한다.
        with self.camera_lock:
            self.latest_camera_message = msg
            self.vlm_camera_messages.append(msg)


    def _image_message_to_pil(self, message: Image | CompressedImage):
        # ROS 이미지 메시지를 Gemma processor가 받을 PIL RGB 이미지로 변환한다.
        if message is None:
            return None

        if isinstance(message, CompressedImage):
            if not message.data:
                return None

            try:
                with PILImage.open(BytesIO(message.data)) as image:
                    return image.convert("RGB").copy()
            except (OSError, ValueError) as error:
                self.get_logger().warning(
                    f"압축 이미지 변환 실패 : format={message.format}, error={error}"
                )
                return None

        if not isinstance(message, Image):
            self.get_logger().warning(
                f"지원하지 않는 이미지 타입 : {type(message).__name__}"
            )
            return None

        if message.encoding in ("rgb8", "bgr8"):
            channels = 3
        elif message.encoding in ("rgba8", "bgra8"):
            channels = 4
        else:
            self.get_logger().warning(
                f"지원하지 않는 raw image encoding : {message.encoding}"
            )
            return None

        pixel_row_size = message.width * channels

        if message.step < pixel_row_size:
            return None

        image_bytes = np.frombuffer(message.data, dtype=np.uint8)
        required_size = message.height * message.step

        if image_bytes.size < required_size:
            return None

        image_rows = image_bytes[:required_size].reshape(
            message.height,
            message.step,
        )
        pixels = image_rows[:, :pixel_row_size].reshape(
            message.height,
            message.width,
            channels,
        )

        if message.encoding == "bgr8":
            pixels = pixels[:, :, ::-1]
        elif message.encoding == "rgba8":
            pixels = pixels[:, :, :3]
        elif message.encoding == "bgra8":
            pixels = pixels[:, :, [2, 1, 0]]

        return PILImage.fromarray(pixels.copy())

    def _snapshot_camera_images(self) -> list:
        # VLM 판정 도중 새 frame이 들어와도 입력이 바뀌지 않도록 ROS 메시지를 먼저 고정한다.
        with self.camera_lock:
            camera_messages = list(self.vlm_camera_messages)

            if not camera_messages and self.latest_camera_message is not None:
                camera_messages = [self.latest_camera_message]

        camera_images = []

        # 이미지 변환은 camera lock 밖에서 수행해 callback이 오래 막히지 않게 한다.
        for camera_message in camera_messages:
            camera_image = self._image_message_to_pil(camera_message)

            if camera_image is not None:
                camera_images.append(camera_image)

        return camera_images

    def _parse_vlm_verdict(self, raw_text: str) -> str:
        # 설명 본문은 무시하고 마지막 한 줄의 고정 판정만 사용한다.
        if not isinstance(raw_text, str) or not raw_text.strip():
            return "uncertain"
        
        last_line = raw_text.strip().splitlines()[-1].strip()

        judge_reason_text = raw_text - last_line

        if last_line in ("판정: PASS", "VERDICT: PASS"):

            if VLM

            return judge_reason_text, "pass"

        if last_line in (
            "판정: FAIL",
            "VERDICT: FAIL",
            "판정: WRONG_INGREDIENT",
        ):
            return judge_reason_text, "fail"

        if last_line in (
            "판정: UNCERTAIN",
            "VERDICT: UNCERTAIN",
            "판정: UNKNOWN_RETAKE",
        ):
            return judge_reason_text, "uncertain"

        return judge_reason_text, "uncertain"

    def _build_vlm_request(self, expected: str, camera_images: list) -> dict | None: # expected : 현재 로봇이 실제로 작업한 재료 이름.
        # 현재 observation이 없으면 모델을 호출하지 않고 UNCERTAIN으로 처리한다.
        if not camera_images:
            return None

        verdict_rule = (
            "마지막 줄에는 반드시 판정: PASS, 판정: FAIL, "
            "판정: UNCERTAIN 중 하나만 출력한다."
        )

        # lid는 DB reference 없이 작업 전후 도시락 장면만 비교한다.
        if expected == "뚜껑":
            if self.comparison_image is None:
                return None

            return {
                "images": [self.comparison_image, *camera_images],
                "system_prompt": (
                    "너는 도시락 뚜껑 작업을 확인하는 시각 판정기다. "
                    "이미지1은 뚜껑 작업 전 도시락이고 이후 이미지는 작업 후 장면이다. "
                    "작업 후 음식이 검은 뚜껑으로 정상적으로 덮였으면 PASS이다. "
                    "뚜껑이 열렸거나 일부만 덮였거나 음식이 계속 노출되면 FAIL이다. "
                    f"가림, 흔들림, 반사 때문에 확신할 수 없으면 UNCERTAIN이다. {verdict_rule}"
                ),
                "user_text": (
                    f"작업 후 이미지 {len(camera_images)}장에서 "
                    "검은 뚜껑이 도시락을 정상적으로 덮었는지 판정해."
                ),
            }

        reference_image = self.vlm_reference.get(expected)

        if reference_image is None and self.vlm_rag is not None:
            reference_image = self.vlm_rag.get_reference(expected)

            if reference_image is not None:
                self.vlm_reference[expected] = reference_image

        if reference_image is None:
            return None

        # sauce는 닫힌 뚜껑 상태가 유지되고 선택한 소스가 올라갔는지 함께 본다.
        if expected in ("크림", "토마토", "오일"):
            if self.comparison_image is None:
                return None

            sauce_label_hints = {
                "토마토": "tomato, pomodoro, Napoli, 나폴리, 토마토",
                "크림": "cream, 크림",
                "오일": "oil, olio, 오일",
            }
            expected_labels = sauce_label_hints[expected]

            return {
                "images": [reference_image, self.comparison_image, *camera_images],
                "system_prompt": (
                    "너는 닫힌 도시락 위에 밀봉된 소스 포장지가 놓였는지 확인하는 시각 판정기다. "
                    "판정 대상은 소스 내용물이나 완성된 음식이 아니라 밀봉된 소스 포장지다. "
                    "이미지1은 선택한 소스 포장지의 참고 이미지이고 이미지2는 소스 작업 전 닫힌 도시락이다. "
                    "이후 이미지는 소스 작업 후 장면이다. "
                    "포장지에 인쇄된 파스타, 토마토, 음식 사진을 실제 음식이나 열린 도시락으로 해석하지 않는다. "
                    "화면의 bounding box, class 글자, 체크 표시 같은 overlay도 실제 물체의 증거로 사용하지 않는다. "
                    f"작업 후 이미지 중 하나 이상에서 검은 뚜껑이 닫혀 있고, 그 위에 밀봉된 포장지가 있으며, "
                    f"포장지의 글자·색상·디자인이 {expected} 소스와 일치하면 PASS이다. "
                    f"{expected} 소스 표기에는 {expected_labels}를 같은 종류로 인정한다. "
                    "브랜드, 포장 방향, 크기, 세부 디자인이 참고 이미지와 다르다는 이유만으로 FAIL하지 않는다. "
                    "선명한 모든 작업 후 이미지에 포장지가 없거나, 명확히 다른 종류의 소스이거나, "
                    "검은 뚜껑이 실제로 열려 있으면 FAIL이다. "
                    f"가림, 흔들림, 반사 때문에 포장지 종류를 확인할 수 없으면 UNCERTAIN이다. {verdict_rule}"
                ),
                "user_text": (
                    f"작업 후 이미지 {len(camera_images)}장에서 "
                    f"닫힌 검은 뚜껑 위에 밀봉된 {expected} 소스 포장지가 놓였는지 판정해. "
                    "포장지에 인쇄된 음식 사진과 화면 overlay는 판정 근거에서 제외해."
                ),
            }

        # 첫 작업에는 이전 장면이 없으므로 reference와 current만 사용한다.
        if self.comparison_image is None:
            return {
                "images": [reference_image, *camera_images],
                "system_prompt": (
                    "너는 로봇의 식재료 투입 작업을 확인하는 시각 판정기다. "
                    "이미지1은 현재 목표 재료의 참고 이미지이고 이후 이미지는 작업 후 장면이다. "
                    f"작업 후 장면에 {expected} 재료가 명확하게 보이면 PASS이다. "
                    "재료가 없거나 다른 재료이면 FAIL이고 확신할 수 없으면 UNCERTAIN이다. "
                    f"{verdict_rule}"
                ),
                "user_text": (
                    f"작업 후 이미지 {len(camera_images)}장에서 "
                    f"{expected} 재료가 실제로 담겼는지 판정해."
                ),
            }

        # 두 번째 작업부터는 직전 물리 장면과 현재 장면의 변화를 함께 본다.
        return {
            "images": [reference_image, self.comparison_image, *camera_images],
            "system_prompt": (
                "너는 로봇 작업 전후의 변화를 확인하는 시각 판정기다. "
                "이미지1은 현재 목표 재료의 참고 이미지이고 이미지2는 이번 작업 전 도시락이다. "
                "이후 이미지는 이번 작업 후 장면이다. "
                f"작업 전과 비교해 {expected} 재료가 새로 보이거나 해당 재료 영역이 증가했으면 PASS이다. "
                "기대 변화가 없거나 다른 재료가 추가됐으면 FAIL이다. "
                f"가림이나 흔들림 때문에 비교할 수 없으면 UNCERTAIN이다. {verdict_rule}"
            ),
            "user_text": (
                f"비교 이미지 출처는 {self.comparison_source}이다. "
                f"작업 후 이미지 {len(camera_images)}장에서 "
                f"{expected} 투입 변화가 실제로 발생했는지 판정해."
            ),
        }

    def _publish_vlm_ui_snapshot(self, expected: str, request: dict | None):
        # UI 기능이 꺼져 있으면 이미지 합성·JPEG 인코딩을 전부 건너뛴다.
        if not ENABLE_VLM_UI_IMAGES or self.vlm_ui_image_pub is None or request is None:
            return

        try:
            request_images = request.get("images")
            if not isinstance(request_images, list) or not request_images:
                return

            reference_image = None
            previous_image = None
            current_image = request_images[-1]

            # lid는 reference가 없고 작업 전·후 이미지만 사용한다.
            if expected == "뚜껑":
                if len(request_images) >= 2:
                    previous_image = request_images[0]
            else:
                reference_image = request_images[0]

                if self.comparison_image is not None and len(request_images) >= 3:
                    previous_image = request_images[1]

            panel_width = 480
            panel_height = 360
            snapshot = PILImage.new(
                "RGB", (panel_width * 3, panel_height), (28, 32, 38)
            )

            # REFERENCE | PREVIOUS | CURRENT 순서로 한 장에 합친다.
            for panel_index, source_image in enumerate(
                [reference_image, previous_image, current_image]
            ):
                if source_image is None:
                    continue

                panel_image = source_image.convert("RGB").copy()
                panel_image.thumbnail(
                    (panel_width, panel_height),
                    PILImage.Resampling.LANCZOS,
                )
                panel_left = panel_index * panel_width + (
                    panel_width - panel_image.width
                ) // 2
                panel_top = (panel_height - panel_image.height) // 2
                snapshot.paste(panel_image, (panel_left, panel_top))

            jpeg_buffer = BytesIO()
            snapshot.save(
                jpeg_buffer,
                format="JPEG",
                quality=80,
                optimize=True,
            )

            snapshot_message = CompressedImage()
            snapshot_message.header.stamp = self.get_clock().now().to_msg()
            snapshot_message.header.frame_id = "vlm_ui_snapshot"
            snapshot_message.format = "jpeg"
            snapshot_message.data = jpeg_buffer.getvalue()
            self.vlm_ui_image_pub.publish(snapshot_message)

        except Exception as error:
            # UI 표시 실패가 실제 VLM 판정을 중단하면 안 된다.
            self.get_logger().warning(f"VLM UI 이미지 발행 실패 : {error}")


    def worker_loop(self):
        # callback은 queue 적재만 하고, 주문 상태 변경은 worker 하나만 담당한다.
        while True:
            mode, data = self._jobs.get()

            try:
                if mode == "start":
                    self._prepare_order()

                elif mode == "reset":
                    self._process_reset(data)

                elif mode == "finish":
                    self._process_finish()

                elif mode == "confirm":
                    self._process_vlm_confirm(data)

                elif mode == "next":
                    self._process_next(data)

                elif mode == "turn":
                    self._process_turn(data)

                else:
                    self.get_logger().warning(f"알 수 없는 LLM 작업 종류 : {mode}")

            except Exception as error:
                self.get_logger().error(f"LLM worker 에러 ({mode}) : {error}")

                if self.conversation_started and self.active_task is None and not self.task_queue and not self.order_finished:
                    self._set_stt_enabled(True)

                self.reply_pub.publish(
                    String(data="처리 중 문제가 생겼어요. 다시 말씀해 주세요.")
                )

            finally:
                self._record_runtime_event(mode, {"data": data})

                try:
                    self._publish_status()
                except Exception as error:
                    self.get_logger().warning(f"UI status 발행 실패 : {error}")

                self._jobs.task_done()


    def _build_agent_state(self, user_text: str) -> dict:
        # LangGraph에는 실제 node 상태를 직접 주지 않고 복사본만 전달함
        # 모델이나 검증 도중 문제가 생겨도 원본 주문이 반쯤 바뀌지 않게 하기 위함
        return {
            "user_text": user_text,
            "order": copy.deepcopy(self.order),
            "execution": {
                "section": self.section,
                "selected": copy.deepcopy(self.selected),
                "task_queue": copy.deepcopy(self.task_queue),
                "active_task": copy.deepcopy(self.active_task),
                "completed_tasks": copy.deepcopy(self.completed_tasks),
                "skipped_sections": copy.deepcopy(self.skipped_sections),
                "robot_started": self.robot_started,
            },
            "recommendation": copy.deepcopy(self.recommendation_state),
            "history": copy.deepcopy(self.history),
            "action_history": copy.deepcopy(self.action_history),
        }

    def _validate_agent_result(self, result: dict):
        # 응답 생성 전에 graph 결과 전체를 검사하되 실제 node 상태는 바꾸지 않는다.
        order_after = result.get("order")
        execution_after = result.get("execution")
        recommendation_after = result.get("recommendation")
        transaction = result.get("transaction")
        policy_reply = result.get("policy_reply")

        if not isinstance(order_after, dict):
            raise ValueError("agent 결과 order가 dict가 아님")

        if not isinstance(execution_after, dict):
            raise ValueError("agent 결과 execution이 dict가 아님")

        if not isinstance(recommendation_after, dict):
            raise ValueError("agent 결과 recommendation이 dict가 아님")

        if not isinstance(transaction, dict):
            raise ValueError("agent 결과 transaction이 dict가 아님")

        if policy_reply is not None and not isinstance(policy_reply, str):
            raise ValueError("agent 결과 policy_reply가 문자열 또는 None이 아님")

        if execution_after.get("section") not in SECTION_ORDER:
            raise ValueError(f"agent 결과 section 이상함 : {execution_after.get('section')}")

        if not isinstance(execution_after.get("selected"), list):
            raise ValueError("agent 결과 selected가 list가 아님")

        if not isinstance(execution_after.get("task_queue"), list):
            raise ValueError("agent 결과 task_queue가 list가 아님")

        if execution_after.get("active_task") is not None and not isinstance(execution_after["active_task"], dict):
            raise ValueError("agent 결과 active_task가 dict 또는 None이 아님")

        if not isinstance(execution_after.get("completed_tasks"), list):
            raise ValueError("agent 결과 completed_tasks가 list가 아님")

        if not isinstance(execution_after.get("skipped_sections"), dict):
            raise ValueError("agent 결과 skipped_sections가 dict가 아님")

        if type(execution_after.get("robot_started")) is not bool:
            raise ValueError("agent 결과 robot_started가 bool이 아님")

    def _commit_agent_result(self, result: dict):
        # 결과 검증과 응답 생성이 모두 성공한 뒤 최종 상태만 원자적으로 반영한다.
        self._validate_agent_result(result)
        order_after = result["order"]
        execution_after = result["execution"]
        recommendation_after = result["recommendation"]

        # 여기부터 실제 상태 commit. 이 함수는 worker 하나만 호출.
        self.order = copy.deepcopy(order_after)
        self.section = execution_after["section"]
        self.selected = copy.deepcopy(execution_after["selected"])
        self.task_queue = copy.deepcopy(execution_after["task_queue"])
        self.active_task = copy.deepcopy(execution_after["active_task"])
        self.completed_tasks = copy.deepcopy(execution_after["completed_tasks"])
        self.skipped_sections = copy.deepcopy(execution_after["skipped_sections"])
        self.robot_started = execution_after["robot_started"]
        self.recommendation_state = copy.deepcopy(recommendation_after)

    def _run_agent_turn(self, user_text: str) -> dict:
        # 한 사용자 발화를 Tool 해석부터 Python 정책 검증까지 한 번만 실행함
        # ROS Reply·plan 발행은 아직 하지 않고 다음 worker 단계로 결과를 반환함
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("빈 사용자 발화는 처리할 수 없음")

        result = self.graph.invoke(self._build_agent_state(user_text))
        self._validate_agent_result(result)

        return result

    def _build_turn_reply(self, user_text: str, state_before: dict, result: dict) -> str | None:
        # 정책 사실과 다음 행동은 Python이 고정하고, 자유 Reply는 설명·공감만 담당한다.
        transaction = result.get("transaction")
        policy_reply = result.get("policy_reply")

        if not isinstance(transaction, dict):
            raise ValueError("Reply를 만들 transaction이 없음")

        reply_state = copy.deepcopy(state_before)
        reply_state["order"] = copy.deepcopy(result["order"])
        reply_state["execution"] = copy.deepcopy(result["execution"])
        reply_state["recommendation"] = copy.deepcopy(result["recommendation"])
        next_step = build_selection_next_step(
            result["order"],
            result["execution"],
        )

        def append_next_step(reply: str) -> str:
            return f"{reply} {next_step}" if next_step else reply

        if policy_reply is not None:
            if not isinstance(policy_reply, str) or not policy_reply.strip():
                raise ValueError("policy_reply 형식이 잘못됨")

            policy_reply = policy_reply.strip()
            action = transaction.get("action")
            confirm = transaction.get("confirm_validation", {})
            recommendation = transaction.get("recommendation_validation", {})
            section_skip = transaction.get("section_skip")
            policy_already_asks_user = (
                action == "recommend_order"
                and (
                    recommendation.get("needs_scope")
                    or recommendation.get("proposal_created")
                )
            ) or (
                action == "refuse_section"
                and isinstance(section_skip, dict)
                and section_skip.get("needs_confirmation")
            )
            work_will_start = (
                confirm.get("requested")
                and confirm.get("allowed")
            ) or recommendation.get("proposal_confirmed")
            section_will_advance = (
                isinstance(section_skip, dict)
                and section_skip.get("applied")
            )

            if policy_already_asks_user or work_will_start or section_will_advance:
                return policy_reply

            policy_already_guides_user = any(marker in policy_reply for marker in (
                "?",
                "말씀해 주세요",
                "다시 말씀해 주세요",
            ))

            if policy_already_guides_user:
                return policy_reply

            return append_next_step(policy_reply)

        action = transaction.get("action")

        # 질문·설명·공감은 자유롭게 답하되 주문 흐름 안내는 Python 문장만 사용한다.
        if action == "respond":
            free_reply = self.generate_reply(reply_state, user_text).strip()
            operational_claims = (
                "반영했",
                "확정했",
                "제외했",
                "제외할게",
                "제외하고",
                "넣어드릴게",
                "담기를 시작",
                "담기가 끝",
                "담기가 완료",
                "작업을 시작",
                "작업을 마쳤",
                "다음은",
                "다음 단계",
                "차례입니다",
                "진행할게",
                "진행하겠",
                "선택해 주",
                "말씀해 주",
                "골라 주",
            )
            reply_sentences = re.split(r"(?<=[.!?])\s+", free_reply)
            safe_sentences = [
                sentence for sentence in reply_sentences
                if "?" not in sentence
                and not any(claim in sentence for claim in operational_claims)
            ]

            if len(safe_sentences) != len(reply_sentences):
                self.get_logger().warning(
                    f"자유응답의 주문 흐름 문장을 제거함 : {free_reply}"
                )

            safe_reply = " ".join(safe_sentences).strip()

            if not safe_reply:
                return next_step or "현재 주문에서 원하시는 내용을 말씀해 주세요."

            return append_next_step(safe_reply)

        # VLM은 LLM MVP 완료 뒤 연결한다.
        if action == "describe_scene":
            if not ENABLE_VLM:
                return append_next_step("현재 카메라 화면 확인 기능을 준비하고 있어요.")

            camera_images = self._snapshot_camera_images()

            if self.call_vlm is None or not camera_images:
                return append_next_step("아직 확인할 카메라 화면이 없어요. 화면이 들어오면 다시 확인할게요.")

            scene_reply = self.call_vlm(
                [camera_images[-1]],
                (
                    "너는 로봇 카메라의 현재 장면을 손님에게 설명한다. "
                    "보이는 내용만 한국어 한두 문장으로 답하고 추측하지 않는다. "
                    "PASS, FAIL 같은 판정 문장은 출력하지 않는다."
                ),
                user_text,
            )

            scene_reply = scene_reply or "카메라 화면을 정확히 설명하지 못했어요."
            return append_next_step(scene_reply)

        # 정상 확정은 다음 단계에서 plan 발행 문장과 합친다.
        confirm = transaction.get("confirm_validation", {})

        if confirm.get("requested") and confirm.get("allowed"):
            return f"{SECTION_LABELS[self.section]} 선택을 확정했어요."

        return None


    def _process_pending_confirmation(self, user_text: str) -> bool:
        # 현재 section 전체 제외의 재확인은 Tool 모델을 다시 호출하지 않고 처리한다.
        if self.awaiting_confirm is None:
            return False

        self._set_stt_enabled(False)
        confirmation_intent = classify_confirmation_intent(
            user_text,
            "section_skip",
        )

        if confirmation_intent is None:
            self._set_stt_enabled(True)
            self.history.append({"role": "user", "content": user_text})
            self._publish_reply("네 또는 아니요로 말씀해 주세요.")
            return True

        pending = copy.deepcopy(self.awaiting_confirm)
        self.awaiting_confirm = None
        self.runtime_turn_index += 1
        self.history.append({"role": "user", "content": user_text})

        action = f"{pending['type']}_{'accept' if confirmation_intent else 'reject'}"
        self.action_history.append({"turn": self.runtime_turn_index, "action": action})
        self.action_history = self.action_history[-12:]

        if pending["type"] == "refuse_section":
            section = pending["section"]

            if section != self.section:
                raise RuntimeError(
                    f"재확인 section이 현재 section과 다름 : {section} != {self.section}"
                )

            if confirmation_intent is False:
                self._set_stt_enabled(True)
                if pending.get("source") == "all_options_restricted":
                    dislike_items = [
                        item
                        for item in pending["items"]
                        if (
                            strongest_restriction_for(self.order, item) or {}
                        ).get("reason") == "dislike"
                    ]
                    item_text = "나 ".join(dislike_items)
                    reply = (
                        f"{SECTION_LABELS[section]} 단계를 제외하지 않을게요. "
                        f"{item_text}은 취향 제한이라 원하면 재료와 양을 다시 말씀해 주세요."
                    )
                else:
                    reply = (
                        f"{SECTION_LABELS[section]} 단계에서 계속 고를게요. "
                        f"{self._section_prompt()}"
                    )
            else:
                current_state = self._build_agent_state(user_text)
                committed = commit_section_skip(
                    current_state["order"],
                    current_state["execution"],
                    current_state["recommendation"],
                    pending,
                )
                self._commit_agent_result({
                    "order": committed["order"],
                    "execution": committed["execution"],
                    "recommendation": committed["recommendation"],
                    "transaction": {},
                    "policy_reply": None,
                })
                reply = self._advance_after_section(section)

        else:
            raise ValueError(f"알 수 없는 재확인 종류 : {pending['type']}")

        self._publish_reply(reply)
        return True

    def _process_turn(self, user_text: str):
        # 주문 대화가 열린 동안의 STT 한 문장만 Agent에 전달함
        if not self.conversation_started:
            self.get_logger().warning("주문 대화 시작 전 STT 결과를 무시함")
            return

        if self.order_finished:
            self.get_logger().warning("완료된 주문의 STT 결과를 무시함")
            return

        # 단계 제외 질문에 대한 네/아니요를 Tool 호출 전에 처리
        if self._process_pending_confirmation(user_text):
            return

        # 로봇 작업 중 들어온 늦은 STT 결과는 주문에 반영하지 않음
        if self.active_task is not None or self.task_queue:
            self._set_stt_enabled(False)
            self.get_logger().warning("로봇 작업 중 STT 결과를 무시함")
            return

        # 모델 추론과 TTS 중 추가 발화를 받지 않도록 먼저 닫음
        self._set_stt_enabled(False)

        state_before = self._build_agent_state(user_text)
        result = self._run_agent_turn(user_text)
        transaction = result["transaction"]
        action = transaction["action"]

        # 모델이 말한 내용이 아니라 Python이 확정한 결과만 다음 턴에 전달함
        action_event = {
            "turn": self.runtime_turn_index + 1,
            "action": action,
        }

        if transaction["accepted"]:
            action_event["accepted"] = copy.deepcopy(transaction["accepted"])

        if transaction["restriction_applied"]:
            action_event["restriction_applied"] = copy.deepcopy(
                transaction["restriction_applied"]
            )

        if action == "recommend_order":
            tool_call = result.get("tool_call") or {}
            action_event["scope"] = tool_call.get("changes", {}).get("scope")

        if action == "confirm_section":
            action_event["section"] = self.section

        # dislike가 포함된 section 제외만 다음 사용자 발화를 기다림
        section_skip = transaction["section_skip"]
        reply = self._build_turn_reply(user_text, state_before, result)

        # 결과 검증과 응답 생성이 모두 성공한 뒤에만 실제 주문 상태를 commit한다.
        self._commit_agent_result(result)
        self.runtime_turn_index += 1
        self.history.append({"role": "user", "content": user_text})
        self.action_history.append(action_event)
        self.action_history = self.action_history[-12:]

        if section_skip is not None and section_skip["needs_confirmation"]:
            self.awaiting_confirm = copy.deepcopy(section_skip)
            self.awaiting_confirm["type"] = "refuse_section"

        # allergy·cannot_eat 전체 제외는 Python에서 바로 적용되므로 재확인 없이 다음 단계로 간다.
        if section_skip is not None and section_skip["applied"]:
            reply = self._advance_after_section(section_skip["section"])

        # 일반 확정 또는 변경 후 즉시 시작 요청
        confirm = transaction["confirm_validation"]
        recommendation = transaction["recommendation_validation"]
        recommendation_confirmed = (
            recommendation["proposal_confirmed"]
            and self.section in self.recommendation_state.get(
                "confirmed_sections", []
            )
        )

        if (
            confirm["requested"]
            and confirm["allowed"]
        ) or recommendation_confirmed:
            if self._queue_current_section_tasks():
                start_reply = (
                    f"{SECTION_LABELS[self.section]} 담기를 시작할게요."
                )
                reply = f"{reply} {start_reply}" if reply else start_reply

            elif self.section == "extra":
                # 추가 재료를 고르지 않고 확정하면 바로 뚜껑 단계로 이동
                self.skipped_sections["extra"] = "empty_confirm"
                reply = self._advance_after_section("extra")

            else:
                raise RuntimeError(
                    f"{self.section} 확정 후 실행할 작업을 만들지 못함"
                )

        else:
            # TTS가 끝난 뒤 다음 사용자 발화를 받을 수 있게 예약
            self._set_stt_enabled(True)

        if reply:
            self._publish_reply(reply)

        self._record_tool_turn(user_text, state_before, result, reply)

    def _record_tool_turn(self, user_text: str, state_before: dict, result: dict, reply: str | None):
        # 모델 출력과 Python 검증 결과를 같은 행에 기록한다.
        if not ENABLE_RUNTIME_LOG:
            return

        try:
            log_row = {
                "schema_version": 4,
                "record_type": "tool_turn",
                "timestamp": datetime.now().astimezone().isoformat(),
                "session_id": self.runtime_session_id,
                "turn_index": self.runtime_turn_index,
                "record_id": f"{self.runtime_session_id}_{self.runtime_turn_index:04d}",
                "adapter_version": Path(TOOL_ADAPTER_PATH).name if ENABLE_TOOL_LORA else "base",
                "parser_status": result.get("parser_status"),
                "user_text": user_text,
                "history_before": copy.deepcopy(state_before.get("history", [])[-12:]),
                "action_history_before": copy.deepcopy(state_before.get("action_history", [])[-12:]),
                "tool_call": copy.deepcopy(result.get("tool_call")),
                "transaction": copy.deepcopy(result.get("transaction")),
                "policy_reply": result.get("policy_reply"),
                "final_reply": reply,
                "state_after": {
                    "order": copy.deepcopy(self.order),
                    "section": self.section,
                    "selected": copy.deepcopy(self.selected),
                    "active_task": copy.deepcopy(self.active_task),
                    "task_queue": copy.deepcopy(self.task_queue),
                    "completed_tasks": copy.deepcopy(self.completed_tasks),
                    "skipped_sections": copy.deepcopy(self.skipped_sections),
                    "recommendation": copy.deepcopy(self.recommendation_state),
                },
            }

            with self.runtime_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_row, ensure_ascii=False) + "\n")

        except Exception as error:
            self.get_logger().warning(f"Tool runtime 로그 저장 실패 : {error}")


    def _record_runtime_event(self, event: str, payload=None):
        # ROS 작업 하나를 처리한 뒤 당시 실행 상태를 기록
        if not ENABLE_RUNTIME_LOG:
            return

        try:
            self.runtime_event_index += 1

            event_row = {
                "schema_version": 1,
                "record_type": "runtime_event",
                "timestamp": datetime.now().astimezone().isoformat(),
                "session_id": self.runtime_session_id,
                "event_index": self.runtime_event_index,
                "record_id": f"{self.runtime_session_id}_event_{self.runtime_event_index:04d}",
                "event": event,
                "turn_index": self.runtime_turn_index,
                "payload": copy.deepcopy(payload or {}),
                "state": {
                    "section": self.section,
                    "active_task": copy.deepcopy(self.active_task),
                    "task_queue": copy.deepcopy(self.task_queue),
                    "completed_tasks": copy.deepcopy(self.completed_tasks),
                    "awaiting_confirm": copy.deepcopy(self.awaiting_confirm),
                    "order_finished": self.order_finished,
                },
            }

            with self.runtime_event_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(event_row, ensure_ascii=False) + "\n")

        except Exception as error:
            self.get_logger().warning(f"ROS runtime 로그 저장 실패 : {error}")

    def _publish_status(self, work_state_override=None):
        # 현재 주문·로봇 진행 상태를 기존 UI JSON 계약으로 발행한다.
        amount_word = {"low": "적게", "normal": "보통", "high": "많이"}
        reason_word = {
            "allergy": "알레르기",
            "cannot_eat": "섭취 불가",
            "dislike": "비선호",
            "dietary_rule": "식단 조건",
        }
        selected = []

        if self.order["sauce"]:
            selected.append(f"{self.order['sauce']} 소스")

        if self.order["noodle_type"]:
            portion = amount_word.get(self.order["noodle_portion"], "")
            selected.append(f"{self.order['noodle_type']} {portion}".strip())

        for topping, amount in self.order["toppings"].items():
            selected.append(f"{topping} {amount_word.get(amount, amount)}")

        for restriction in self.order["restrictions"]:
            target = restriction["target"]
            reason = reason_word.get(restriction["reason"], restriction["reason"])
            selected.append(f"제외 조건 : {target} ({reason})")

        section_index = SECTION_ORDER.index(self.section)
        section_label = SECTION_LABELS[self.section]

        if self.order_finished:
            status_text = "포장 완료"
            completed_sections = SECTION_ORDER

        elif isinstance(self.awaiting_confirm, dict):
            status_text = "현재 단계 제외 확인 중"
            completed_sections = SECTION_ORDER[:section_index]

        elif self.active_task is not None:
            status_text = f"{self.active_task['class']} 담는 중"
            completed_sections = SECTION_ORDER[:section_index]

        elif self.ui_started:
            status_text = "주문 시작 대기 중"
            completed_sections = []

        else:
            status_text = f"{section_label} 선택 중"
            completed_sections = SECTION_ORDER[:section_index]

        current_task = (
            self.active_task["class"]
            if self.active_task is not None
            else section_label
        )

        confirmed_sections = self.recommendation_state.get("confirmed_sections", [])
        auto_running = bool(
            self.section in confirmed_sections
            and (self.active_task is not None or self.task_queue)
        )

        # 새 작업 화면은 주문 선택값과 실제 VLM PASS 결과를 따로 표시한다.
        ui_group_names = {
            "noodle": "noodle", "veggie": "vegetable", "meat": "meat",
            "extra": "extra", "lid": "cover", "sauce": "sauce",
        }
        current_class = self.active_task["class"] if self.active_task is not None else ""
        selected_classes = []

        if self.order["noodle_type"]:
            selected_classes.append(self.order["noodle_type"])

        selected_classes.extend(self.order["toppings"].keys())

        # policy_success는 공정 진행용 success이므로 실제 확인 재료에 넣지 않는다.
        filled_classes = []
        for result in self.verification_history:
            class_name = result.get("class")
            if result.get("result") == "pass" and isinstance(class_name, str) and class_name and class_name not in filled_classes:
                filled_classes.append(class_name)

        current_vlm_result = None
        if self.active_task is not None:
            for result in reversed(self.verification_history):
                if result.get("class") == self.active_task["class"]:
                    current_vlm_result = result.get("result")
                    break

        if self.order_finished:
            work_state = "completed"
        elif self.active_task is None:
            work_state = "idle"
        elif self.vlm_confirmed and current_vlm_result == "pass":
            work_state = "vlm_done"
        elif current_vlm_result == "fail":
            work_state = "vlm_fail"
        elif current_vlm_result == "uncertain":
            work_state = "uncertain_wait"
        elif self.vlm_confirmed and current_vlm_result == "policy_success":
            work_state = "retried"
        elif not ENABLE_VLM and self.vlm_confirmed:
            work_state = "work_done"
        else:
            work_state = "working"

        if work_state_override is not None:
            work_state = work_state_override

        completed_groups = [ui_group_names[name] for name in completed_sections if name not in self.skipped_sections]
        skipped_groups = [ui_group_names[name] for name in SECTION_ORDER if name in self.skipped_sections]
        has_order_value = bool(self.order["sauce"] or self.order["noodle_type"] or self.order["noodle_portion"] or self.order["toppings"] or self.order["restrictions"])
        reset_allowed = not has_order_value and not self.robot_started and self.active_task is None and not self.task_queue and not self.completed_tasks and not self.order_finished

        ui_json = {
            "phase": "완료" if self.order_finished else "주문중",
            "section": section_label,
            "section_status": "완료" if self.order_finished else "진행 중",
            "current_task": current_task,
            "target_list": [SECTION_LABELS[name] for name in SECTION_ORDER],
            "completed": [],
            "selected": selected,
            "auto_running": auto_running,
            "skipped": [
                SECTION_LABELS[name]
                for name in SECTION_ORDER
                if name in self.skipped_sections
            ],
            "status_text": status_text,
            "current_class": current_class,
            "current_group": ui_group_names[self.section],
            "work_state": work_state,
            "selected_classes": selected_classes,
            "selected_sauce": self.order["sauce"],
            "filled_classes": filled_classes,
            "completed_groups": completed_groups,
            "skipped_groups": skipped_groups,
            "reset_allowed": reset_allowed,
        }

        for task in self.completed_tasks:
            ui_json["completed"].append(
                f"{task['class']} × {task['repeat_count']}"
            )

        for section_name in completed_sections:
            if section_name not in self.skipped_sections:
                ui_json["completed"].append(SECTION_LABELS[section_name])

        self.status_pub.publish(
            String(data=json.dumps(ui_json, ensure_ascii=False))
        )

    def _set_stt_enabled(self, enabled: bool):
        # True는 TTS 종료 뒤 STT를 열라는 예약 신호
        self.stt_enable_pub.publish(Bool(data=enabled))
        self.get_logger().info(f"STT 상태 : {'열기 대기' if enabled else '닫기'}")

    def _publish_reply(self, reply: str):
        # UI·TTS로 실제 발행한 문장만 다음 멀티턴 history에 저장한다.
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("발행할 reply가 빈 문자열임")

        reply = reply.strip()
        self.reply_pub.publish(String(data=reply))
        self.history.append({"role": "assistant", "content": reply})

    def _clear_state(self):
        # 주문 세션 상태만 초기화하고 model·ROS publisher·queue는 유지한다.
        self.order = new_order()
        self.section = "noodle"
        self.selected = []
        self.task_queue = []
        self.active_task = None
        self.completed_tasks = []
        self.skipped_sections = {}
        self.robot_started = False

        self.recommendation_state = {"phase": "idle", "confirmed_sections": []}
        self.last_recommendation = {}
        self.awaiting_confirm = None
        self.already_notified_safety_facts = set()

        self.ui_started = False
        self.pending_initial_next = False
        self.conversation_started = False
        self.order_finished = False
        self.history = []
        self.action_history = []

        self.previous_success_image = None
        self.comparison_image = None
        self.comparison_source = None
        self.comparison_task_class = None
        self.verification_history = []
        self.vlm_confirmed = False

        with self.camera_lock:
            self.latest_camera_message = None
            self.vlm_camera_messages.clear()

    def _process_reset(self, data):
        # 완료 화면 reset과 시작 전 수동 reset을 구분한다.
        reset_action = None

        try:
            reset_payload = json.loads(data)

            if isinstance(reset_payload, dict):
                reset_action = reset_payload.get("action")

        except (json.JSONDecodeError, TypeError):
            pass

        if reset_action == "complete":
            if not self.order_finished:
                self.get_logger().warning("주문 완료 전이라 complete UI reset을 무시함")
                return

            self._clear_state()
            self._set_stt_enabled(False)
            self.get_logger().info("정상 주문 완료 후 LLM 상태 초기화")
            return

        if reset_action not in ("back", "home"):
            self.get_logger().warning(f"지원하지 않는 /ui/reset action을 무시함 : {reset_action}")
            return

        has_order_value = bool(self.order["sauce"] or self.order["noodle_type"] or self.order["noodle_portion"] or self.order["toppings"] or self.order["restrictions"])

        if has_order_value or self.robot_started or self.active_task is not None or self.task_queue or self.completed_tasks:
            self.get_logger().warning("선택된 주문 또는 로봇 작업이 있어서 /ui/reset을 무시함")
            return

        self._clear_state()
        self._set_stt_enabled(False)
        self.get_logger().info(f"시작 전 LLM 상태 초기화 : {data}")

    def _process_finish(self):
        if self.section != "sauce" or self.active_task is None:
            self.get_logger().warning("마지막 소스 작업 중이 아니어서 /llm/reset을 무시함")
            return

        # 바로 여기에 추가
        if ENABLE_VLM and not self.vlm_confirmed:
            self.get_logger().warning("소스 VLM 판정 전이라 /llm/reset을 보류함")
            return

        completed_task = copy.deepcopy(self.active_task)
        self.active_task = None

        if completed_task not in self.completed_tasks:
            self.completed_tasks.append(completed_task)

        self.awaiting_confirm = None
        self.conversation_started = False
        self.order_finished = True
        self._set_stt_enabled(False)

        self._publish_reply("소스까지 모두 담았어요. 이용해 주셔서 감사합니다.")
        self.done_pub.publish(Bool(data=True))
        self.get_logger().info("총 주문 완료")


    def _process_vlm_confirm(self, should_confirm: bool):
        # /llm/confirm_start는 로봇 작업 완료 후 VLM 판정을 시작하는 신호이다.
        self.get_logger().info(f"/llm/confirm_start 수신 : {should_confirm}")

        if type(should_confirm) is not bool or not should_confirm:
            self.get_logger().warning("confirm_start가 True가 아니어서 무시함")
            return

        if self.active_task is None:
            self.get_logger().warning("판정할 active_task가 없어서 confirm_start를 무시함")
            return

        if self.vlm_confirmed:
            self.get_logger().warning("현재 active_task는 이미 VLM 확인을 마침")
            return

        expected = self.active_task["class"]
        self.vlm_confirmed = False

        # VLM OFF에서는 기존 제어 사이클을 그대로 통과시킨다.
        if not ENABLE_VLM:
            self.vlm_confirmed = True

            if ENABLE_DUMMY_VLM_PASS:
                self.verification_history.append({
                    "class": expected,
                    "attempt": 1,
                    "verdict": "dummy_pass",
                    "result": "pass",
                })

            self.vlm_result_pub.publish(String(data="success"))
            mode = "dummy PASS 기록" if ENABLE_DUMMY_VLM_PASS else "실제 PASS 기록 없음"
            self.get_logger().info(f"VLM OFF 우회 결과 발행 : success, {mode}")
            return


        self._publish_status("vlm_checking")

        camera_images = self._snapshot_camera_images()
        request = self._build_vlm_request(expected, camera_images)
        self._publish_vlm_ui_snapshot(expected, request)
        raw_text = ""

        try:
            if self.call_vlm is None or request is None:
                verdict = "uncertain"
            else:
                raw_text = self.call_vlm(
                    request["images"],
                    request["system_prompt"],
                    request["user_text"],
                )
                verdict = self._parse_vlm_verdict(raw_text)

        except Exception as error:
            raw_text = f"ERROR: {error}"
            verdict = "uncertain"
            self.get_logger().error(f"VLM 판정 실패 : {error}")

        previous_failures = sum(
            1
            for result in self.verification_history
            if result["class"] == expected and result["result"] == "fail"
        )
        attempt = previous_failures + 1

        # 향후 inspection pose 계약이 생기면 새 frame만 받아 재판정한다.
        if verdict == "uncertain" and ENABLE_UNCERTAIN_RETAKE:
            self.verification_history.append({
                "class": expected,
                "attempt": attempt,
                "verdict": verdict,
                "result": "uncertain",
            })

            with self.camera_lock:
                self.latest_camera_message = None
                self.vlm_camera_messages.clear()

            self._record_runtime_event("vlm_result", {
                "class": expected,
                "attempt": attempt,
                "verdict": verdict,
                "policy_result": "retake_wait",
                "comparison_source": self.comparison_source,
                "current_frame_count": len(camera_images),
                "raw_text": raw_text[-1000:],
            })
            self._publish_reply("카메라 화면이 불확실해 새 화면을 기다릴게요.")
            return

        # PASS에서만 신뢰 가능한 previous_success_image를 갱신한다.
        if verdict == "pass":
            current_image = camera_images[-1].copy()

            self.previous_success_image = current_image.copy()
            self.comparison_image = current_image
            self.comparison_source = "pass"
            self.comparison_task_class = expected
            self.vlm_confirmed = True

            self.verification_history.append({
                "class": expected,
                "attempt": attempt,
                "verdict": verdict,
                "result": "pass",
            })

            self.vlm_result_pub.publish(String(data="success"))
            self._record_runtime_event("vlm_result", {
                "class": expected,
                "attempt": attempt,
                "verdict": verdict,
                "policy_result": "success",
                "comparison_source": self.comparison_source,
                "current_frame_count": len(camera_images),
                "raw_text": raw_text[-1000:],
            })
            self._publish_reply(
                f"{expected} 작업을 확인했어요. 로봇 작업을 마무리할게요."
            )
            return

        # UNCERTAIN 재촬영이 꺼져 있으면 기존 FAIL 재시도 규약을 적용한다.
        if previous_failures == 0:
            self.verification_history.append({
                "class": expected,
                "attempt": attempt,
                "verdict": verdict,
                "result": "fail",
            })

            fail_for_publish = {
                "result": "fail",
                "class": TOPIC_CLASS_NAMES[expected],
            }
            self.vlm_result_pub.publish(
                String(data=json.dumps(fail_for_publish, ensure_ascii=False))
            )

            # 재시도 판정에는 첫 시도 이후 들어온 새 frame만 사용한다.
            with self.camera_lock:
                self.latest_camera_message = None
                self.vlm_camera_messages.clear()

            self._record_runtime_event("vlm_result", {
                "class": expected,
                "attempt": attempt,
                "verdict": verdict,
                "policy_result": "robot_retry",
                "comparison_source": self.comparison_source,
                "current_frame_count": len(camera_images),
                "raw_text": raw_text[-1000:],
            })
            self._publish_reply(
                f"{expected} 위치를 확인하지 못했어요. 해당 작업을 한 번 다시 시도할게요."
            )
            return

        # 재시도도 실패하면 외부 규약대로 success를 보내되 실제 PASS로 기록하지 않는다.
        self.verification_history.append({
            "class": expected,
            "attempt": attempt,
            "verdict": verdict,
            "result": "policy_success",
        })

        if camera_images:
            self.comparison_image = camera_images[-1].copy()
            self.comparison_source = "retried"
            self.comparison_task_class = expected
        else:
            self.comparison_image = None
            self.comparison_source = None
            self.comparison_task_class = None

        self.vlm_confirmed = True
        self.vlm_result_pub.publish(String(data="success"))

        self._record_runtime_event("vlm_result", {
            "class": expected,
            "attempt": attempt,
            "verdict": verdict,
            "policy_result": "policy_success",
            "comparison_source": self.comparison_source,
            "current_frame_count": len(camera_images),
            "raw_text": raw_text[-1000:],
        })
        self._publish_reply(
            f"{expected} 재시도를 마쳤어요. 다음 단계로 진행할게요."
        )



    def _prepare_order(self):
        # /ui/start와 첫 /llm/next=3이 모두 도착할 때만 greeting을 시작한다.
        if self.ui_started or self.conversation_started or self.active_task is not None:
            self.get_logger().warning("이미 주문 세션이 진행 중이라 /ui/start를 무시함")
            return

        pending_initial_next = self.pending_initial_next
        self._clear_state()

        self.runtime_session_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
        self.runtime_turn_index = 0
        self.runtime_event_index = 0

        self._set_stt_enabled(False)
        self.ui_started = True

        if pending_initial_next:
            self.get_logger().info("먼저 도착한 /llm/next로 주문 대화를 시작함")
            self._start_order()

    def _start_order(self):
        # 기존 계약: greeting TTS가 끝난 뒤 실제 STT가 열린다.
        self.ui_started = False
        self.pending_initial_next = False
        self.conversation_started = True
        self._set_stt_enabled(True)

        reply = (
            "안녕하세요! 저는 스파게티 밀키트 주문을 도와드리는 로봇이에요. "
            "먼저 면 종류, 소스, 어느 정도 양을 드시고 싶은지 말씀해 주세요."
        )
        self._publish_reply(reply)
        self.get_logger().info("새 주문 시작")


    def _publish_next_task(self) -> bool:
        # active 작업이 끝나기 전에는 다음 plan으로 덮어쓰지 않는다.
        if self.active_task is not None:
            self.get_logger().warning("진행 중인 active_task가 있어서 다음 plan을 발행하지 않음")
            return False

        if not self.task_queue:
            return False

        task = self.task_queue[0]
        task_class = task.get("class")
        repeat_count = task.get("repeat_count")

        if task_class not in TOPIC_CLASS_NAMES:
            raise ValueError(f"plan class를 ROS 이름으로 변환할 수 없음 : {task_class}")

        if type(repeat_count) is not int or repeat_count <= 0:
            raise ValueError(f"plan repeat_count가 양의 정수가 아님 : {repeat_count}")

        # 검증을 끝낸 뒤에만 queue에서 꺼내 active 상태로 이동한다.
        self.task_queue.pop(0)
        self.active_task = copy.deepcopy(task)
        self.robot_started = True
        self.vlm_confirmed = False

        # plan 발행 이전 프레임은 이번 작업 판정에 사용하지 않는다.
        if ENABLE_VLM:
            with self.camera_lock:
                self.latest_camera_message = None
                self.vlm_camera_messages.clear()

        plan_for_publish = {
            "class": TOPIC_CLASS_NAMES[task_class],
            "repeat_count": repeat_count,
        }

        self.plan_pub.publish(
            String(data=json.dumps(plan_for_publish, ensure_ascii=False))
        )

        self.get_logger().info(
            f"plan 발행 : {plan_for_publish}, "
            f"내부 작업 : {self.active_task}, "
            f"남은 작업 : {len(self.task_queue)}"
        )
        return True

    def _queue_current_section_tasks(self) -> bool:
        # 현재 주문을 해당 section의 로봇 작업 목록으로 한 번만 변환한다.
        if self.active_task is not None or self.task_queue:
            self.get_logger().warning("이미 실행할 작업이 있어서 section task를 다시 만들지 않음")
            return False

        tasks = build_section_plan(self.order, self.section)

        if not tasks:
            return False

        # 선택 상태를 queued 상태로 이동한다. 실제 메뉴값은 order에 그대로 남는다.
        queued_keys = []

        for task in tasks:
            task_class = task["class"]

            if task_class in ("얇은면", "넓은면"):
                queued_keys.extend(["noodle_type", "noodle_portion"])
            elif task_class in ("양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노"):
                queued_keys.append(f"toppings.{task_class}")
            elif task_class in ("오일", "토마토", "크림"):
                queued_keys.append("sauce")

        self.selected = [
            key for key in self.selected
            if key not in queued_keys
        ]
        self.task_queue.extend(copy.deepcopy(tasks))

        # 첫 plan부터 마지막 task 완료까지 사용자 수음을 Defense
        self._set_stt_enabled(False)
        return self._publish_next_task()

    def _section_prompt(self) -> str:
        # 사용자 입력이 필요한 section에서 재료와 양을 함께 안내
        if self.section == "veggie":
            return "다음은 야채를 고르실 차례입니다. 양파와 버섯 중 어떤 재료를 어느 정도 양으로 드시고 싶은지 말씀해 주세요."

        if self.section == "meat":
            return "다음은 육류를 고르실 차례입니다. 소시지와 게살 중 어떤 재료를 어느 정도 양으로 드시고 싶은지 말씀해 주세요."

        if self.section == "extra":
            return "다음은 추가 재료를 고르실 차례입니다. 치즈와 페퍼론치노 중 어떤 재료를 어느 정도 양으로 드시고 싶은지 말씀해 주세요."

        raise ValueError(f"사용자 선택 안내가 없는 section임 : {self.section}")

    def _advance_after_section(self, completed_section: str) -> str:
        # 완료·제외 문장을 여기서 만들고 호출자가 한 번만 TTS로 발행한다.
        section_label = SECTION_LABELS[completed_section]

        if completed_section in self.skipped_sections:
            completed_reply = f"{section_label}는 제외했어요."
        elif completed_section == "lid":
            completed_reply = "뚜껑을 닫았어요."
        else:
            completed_reply = f"{section_label} 담기가 끝났어요."

        # 명시적으로 제외된 section은 다시 질문하지 않는다.
        next_step = next_section(completed_section)

        while next_step in self.skipped_sections:
            self.get_logger().info(
                f"건너뛴 section 통과 : {next_step}, "
                f"이유 : {self.skipped_sections[next_step]}"
            )
            next_step = next_section(next_step)

        if next_step is None:
            raise ValueError(f"{completed_section} 다음 section이 없음")

        self.section = next_step

        # 추가 재료 뒤에는 뚜껑 plan을 바로 발행한다.
        if self.section == "lid":
            if not self._queue_current_section_tasks():
                raise RuntimeError("lid 작업을 만들지 못함")

            return f"{completed_reply} 이제 뚜껑을 닫을게요."

        # 뚜껑 뒤에는 처음 선택한 소스 plan을 바로 발행한다.
        if self.section == "sauce":
            if not self._queue_current_section_tasks():
                raise RuntimeError("선택된 소스 작업을 만들지 못함")

            return f"{completed_reply} 마지막으로 고르신 소스를 올릴게요."

        # 추천으로 확정된 다음 section은 다시 묻지 않고 실행한다.
        confirmed_sections = self.recommendation_state.get("confirmed_sections", [])

        if self.section in confirmed_sections and self._queue_current_section_tasks():
            return f"{completed_reply} 추천한 {SECTION_LABELS[self.section]} 재료 담기를 시작할게요."

        self._set_stt_enabled(True)
        return f"{completed_reply} {self._section_prompt()}"

    def _process_next(self, result_code: int):
        # 첫 next는 주문 시작, 이후 next는 active task 완료 신호이다.
        self.get_logger().info(f"/llm/next 수신 : {result_code}")

        if result_code != BY_VLM_NUM:
            self.get_logger().warning(f"규약과 다른 /llm/next 값을 무시함 : {result_code}")
            return

        # /ui/start보다 먼저 도착한 최초 next는 버리지 않고 저장한다.
        if not self.ui_started and not self.conversation_started and self.active_task is None and not self.order_finished:
            self.pending_initial_next = True
            self.get_logger().info("/ui/start보다 먼저 도착한 /llm/next를 보관함")
            return

        if self.ui_started:
            self._start_order()
            return

        if self.active_task is None:
            self.get_logger().warning("진행 중인 active_task가 없어서 /llm/next를 무시함")
            return

        # sauce 완료는 기존 계약대로 /llm/reset에서 처리
        if self.section == "sauce":
            self.get_logger().warning("소스 작업 후 /llm/reset을 기다리는 중")
            return

        if ENABLE_VLM and not self.vlm_confirmed:
            self.get_logger().warning("VLM 판정 전이라 /llm/next를 보류함")
            return

        completed_task = copy.deepcopy(self.active_task)
        self.active_task = None

        if completed_task not in self.completed_tasks:
            self.completed_tasks.append(completed_task)

        self.get_logger().info(f"active task 완료 : {completed_task}")

        # 같은 section에 남은 작업이 있으면 다음 plan만 발행
        if self.task_queue:
            self._publish_next_task()
            return

        reply = self._advance_after_section(self.section)
        self._publish_reply(reply)


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
