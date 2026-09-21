import copy
import json
import os
import queue
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

from soomac_irc.agent import build_graph, classify_confirmation_intent
from soomac_irc.call_model import load_model, make_call_model, make_call_vlm, make_generate_reply
from soomac_irc.order_policy import commit_section_skip
from soomac_irc.order import SECTION_LABELS, SECTION_ORDER, build_section_plan, new_order, next_section
from soomac_irc.reply import build_selection_next_step, build_turn_reply
from soomac_irc.restriction import strongest_restriction_for
from soomac_irc.vlm import build_vlm_request, build_vlm_spoken_reply, decide_vlm_outcome, parse_vlm_verdict
from soomac_irc.validation import validate_agent_result


# 운영 기능 토글
ENABLE_VLM = True
ENABLE_VLM_UI_IMAGES = True
ENABLE_DUMMY_VLM_PASS = os.environ.get("SOOMAC_DUMMY_VLM_PASS") == "1"
ENABLE_UNCERTAIN_RETAKE = False
ENABLE_TOOL_LORA = True
ENABLE_RUNTIME_LOG = True

# VLM 판단 근거를 TTS에 포함할지 결정한다.
VLM_JUDGE_REASON_SPEAK = False
VLM_JUDGE_REASON_MAX_CHARS = 300

# UI가 마지막 TTS 종료를 대조할 문장. 실제 발행문과 상태 JSON에 같은 값을 쓴다.
ORDER_COMPLETE_REPLY = "소스까지 모두 담았어요. 이용해 주셔서 감사합니다."

# 모델과 로그 경로
TOOL_ADAPTER_PATH = "/home/roma/ros2_ws/src/soomac_irc/finetune/v5_1/runs/gemma4_tool_lora_int8_v5_1_deterministic"
RUNTIME_LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "soomac_runtime_logs"

# 기존 ROS 계약
BY_VLM_NUM = 3
IMAGE_TOPIC_TYPE = CompressedImage
WORLD_CAM_TOPIC = "/vision/overlay_image"
VLM_UI_IMAGE_TOPIC = "/agent/vlm_snapshot" # UI에 보낼 이미지 토픽
NUMBER_IMAGE_FOR_CONFIRM = 5

# Python 내부 주문 이름 -> 토픽 클래스로 바꿈
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
        super().__init__("soomac_llm_node_v8")

        # 확정된 주문 상태
        self.order = new_order()

        # 로봇 실행 상태
        self.section = "noodle"
        self.selected = []
        self.task_queue = []
        self.active_task = None
        self.completed_tasks = []
        self.skipped_sections = {}
        self.robot_started = False

        # 추천과 멀티턴 확인 상태
        self.recommendation_state = {"phase": "idle", "confirmed_sections": []}
        self.last_recommendation = {}
        self.awaiting_confirm = None
        self.already_notified_safety_facts = set()

        # 주문 세션 상태
        self.ui_started = False
        self.pending_initial_next = False
        self.conversation_started = False
        self.order_finished = False

        # 모델에 전달할 최근 대화와 실제 처리 기록
        self.history = []
        self.action_history = []

        # runtime 로그 상태
        self.runtime_session_id = ""
        self.runtime_turn_index = 0
        self.runtime_event_index = 0

        runtime_file_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.runtime_log_path = RUNTIME_LOG_DIRECTORY / f"tool_turns_{runtime_file_id}.jsonl"
        self.runtime_event_log_path = RUNTIME_LOG_DIRECTORY / f"runtime_events_{runtime_file_id}.jsonl"

        if ENABLE_RUNTIME_LOG:
            RUNTIME_LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
            self.runtime_log_path.touch(exist_ok=False)
            self.runtime_event_log_path.touch(exist_ok=False)

        # base model은 한 번만 올리고 Tool·Reply·VLM에서 공유한다.
        self.get_logger().info("모델 로딩중")
        self.model, self.processor = load_model(TOOL_ADAPTER_PATH if ENABLE_TOOL_LORA else None)

        call_model = make_call_model(self.model, self.processor, self.get_logger())
        self.graph = build_graph(call_model)
        self.generate_reply = make_generate_reply(self.model, self.processor)

        # VLM 상태는 주문 상태와 분리한다.
        self.call_vlm = None
        self.vlm_rag = None
        self.vlm_reference = {}
        self.latest_camera_message = None
        self.vlm_camera_messages = deque(maxlen=NUMBER_IMAGE_FOR_CONFIRM)
        self.previous_success_image = None
        self.comparison_image = None
        self.comparison_source = None
        self.comparison_task_class = None
        self.verification_history = []
        self.vlm_confirmed = False
        self.camera_lock = threading.Lock()
        self.camera_sub = None

        if ENABLE_VLM:
            from soomac_irc.vlm_rag import VlmRag

            self.call_vlm = make_call_vlm(self.model, self.processor, self.get_logger())
            self.vlm_rag = VlmRag()

        # UI·TTS·제어가 사용하는 기존 publisher 계약
        self.reply_pub = self.create_publisher(String, "/llm_response", 10)
        self.plan_pub = self.create_publisher(String, "/llm/plan", 10)
        self.done_pub = self.create_publisher(Bool, "/llm/done", 10)
        self.status_pub = self.create_publisher(String, "/agent/status", 10)
        self.stt_enable_pub = self.create_publisher(Bool, "/stt/enable", 10)
        self.vlm_result_pub = self.create_publisher(String, "/vlm/confirm", 10)
        self.vlm_ui_image_pub = None

        if ENABLE_VLM and ENABLE_VLM_UI_IMAGES:
            self.vlm_ui_image_pub = self.create_publisher(CompressedImage, VLM_UI_IMAGE_TOPIC, 1)

        # callback은 작업만 queue에 넣고 worker 하나가 실제 상태를 변경한다.
        self._jobs = queue.Queue(maxsize=32)

        self.stt_sub = self.create_subscription(String, "/stt_question", self.stt_callback, 10)
        self.ui_start_sub = self.create_subscription(String, "/ui/start", self.ui_start_callback, 10)
        self.ui_reset_sub = self.create_subscription(String, "/ui/reset", self.ui_reset_callback, 10)
        self.next_trigger_sub = self.create_subscription(Int16, "/llm/next", self.trigger_callback, 10)
        self.confirm_start_sub = self.create_subscription(Bool, "/llm/confirm_start", self.confirm_start_callback, 10)
        self.llm_reset_sub = self.create_subscription(String, "/llm/reset", self.llm_reset_callback, 10)

        if ENABLE_VLM:
            self.camera_sub = self.create_subscription(IMAGE_TOPIC_TYPE, WORLD_CAM_TOPIC, self.camera_callback, qos_profile_sensor_data)

        self._worker = threading.Thread(target=self.worker_loop, name="llm_worker_v8", daemon=True)
        self._worker.start()
        self.get_logger().info("모델 로딩 완료")


    def _enqueue_job(self, mode: str, data):
        # callback을 기다리게 하지 않고 worker queue에 작업만 넣는다.
        try:
            self._jobs.put_nowait((mode, data))
        except queue.Full:
            self.get_logger().error(f"LLM 작업 큐가 가득 차서 {mode} 메시지를 받지 못함")


    def ui_start_callback(self, msg: String):
        # UI 시작 신호를 worker에 전달
        self._enqueue_job("start", msg.data)


    def stt_callback(self, msg: String):
        # 사용자 발화를 callback에서 추론하지 않고 worker에 전달
        self._enqueue_job("turn", msg.data)


    def ui_reset_callback(self, msg: String):
        # UI 초기화 요청을 worker에 전달
        self._enqueue_job("reset", msg.data)

    def llm_reset_callback(self, msg: String):
    # 마지막 소스 작업 완료 뒤 main이 보내는 종료 신호를 worker에 전달
        self._enqueue_job("finish", msg.data)


    def confirm_start_callback(self, msg: Bool):
        # 현재 active_task의 VLM 판정을 시작하라는 신호를 worker에 전달
        self._enqueue_job("confirm", msg.data)


    def trigger_callback(self, msg: Int16):
        # 첫 주문 시작 또는 active_task 완료 신호를 worker에 전달
        self._enqueue_job("next", msg.data)


    def camera_callback(self, msg: Image | CompressedImage):
        # worker queue에 모든 frame을 넣지 않고 최근 frame만 별도로 보관
        with self.camera_lock:
            self.latest_camera_message = msg
            self.vlm_camera_messages.append(msg)


    def _image_message_to_pil(self, message: Image | CompressedImage):
        # ROS 이미지 메시지를 Gemma processor가 받을 PIL RGB 이미지로 변환
        if message is None:
            return None

        if isinstance(message, CompressedImage):
            if not message.data:
                return None

            try:
                with PILImage.open(BytesIO(message.data)) as image:
                    return image.convert("RGB").copy()
            except (OSError, ValueError) as error:
                self.get_logger().warning(f"압축 이미지 변환 실패 : format={message.format}, error={error}")
                return None

        if not isinstance(message, Image):
            self.get_logger().warning(f"지원하지 않는 이미지 타입 : {type(message).__name__}")
            return None

        if message.encoding in ("rgb8", "bgr8"):
            channels = 3
        elif message.encoding in ("rgba8", "bgra8"):
            channels = 4
        else:
            self.get_logger().warning(f"지원하지 않는 raw image encoding : {message.encoding}")
            return None

        pixel_row_size = message.width * channels

        if message.step < pixel_row_size:
            return None

        image_bytes = np.frombuffer(message.data, dtype=np.uint8)
        required_size = message.height * message.step

        if image_bytes.size < required_size:
            return None

        image_rows = image_bytes[:required_size].reshape(message.height, message.step)
        pixels = image_rows[:, :pixel_row_size].reshape(message.height, message.width, channels)

        if message.encoding == "bgr8":
            pixels = pixels[:, :, ::-1]
        elif message.encoding == "rgba8":
            pixels = pixels[:, :, :3]
        elif message.encoding == "bgra8":
            pixels = pixels[:, :, [2, 1, 0]]

        return PILImage.fromarray(pixels.copy())

    def _snapshot_camera_images(self) -> list:
        # VLM 판정 도중 새 frame이 들어와도 입력이 바뀌지 않도록 ROS 메시지 목록을 먼저 고정한다.
        with self.camera_lock:
            camera_messages = list(self.vlm_camera_messages)

            if not camera_messages and self.latest_camera_message is not None:
                camera_messages = [self.latest_camera_message]

        camera_images = []

        # 이미지 변환은 camera lock 밖에서 수행해 callback을 오래 막지 않는다.
        for camera_message in camera_messages:
            camera_image = self._image_message_to_pil(camera_message)

            if camera_image is not None:
                camera_images.append(camera_image)

        return camera_images


    def worker_loop(self):
        # callback이 queue에 넣은 작업을 worker 하나가 순서대로 처리
        # 주문, 추천, 작업 상태는 이 worker만 변경
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

                self.reply_pub.publish(String(data="처리 중 문제가 생겼어요. 다시 말씀해 주세요."))

            finally:
                self._record_runtime_event(mode, {"data": data})

                try:
                    self._publish_status()
                except Exception as error:
                    self.get_logger().warning(f"UI status 발행 실패 : {error}")

                self._jobs.task_done()


    def _clear_state(self):
        # 주문 세션 상태만 초기화하고 model·ROS publisher·worker queue는 유지한다.
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

    def _build_agent_state(self, user_text: str) -> dict:
        # 실제 Node 상태를 직접 넘기지 않고 복사본으로 한 턴을 계산한다.
        # 검증 또는 응답 생성 실패 시 기존 주문 상태는 그대로 유지된다.
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
        # 상태 commit 전에 Agent 결과의 필수 구조를 검사한다.
        validate_agent_result(result)


    def _commit_agent_result(self, result: dict):
        # 결과 검증과 최종 응답 생성이 성공한 뒤 실제 Node 상태를 한 번에 반영한다.
        # 호출자는 _validate_agent_result를 통과한 결과만 전달한다.
        order_after = result["order"]
        execution_after = result["execution"]
        recommendation_after = result["recommendation"]

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
        # Tool 해석과 Python 주문 정책을 실행하지만 실제 Node 상태에는 아직 반영하지 않는다.
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("빈 사용자 발화는 처리할 수 없음")

        state_before = self._build_agent_state(user_text)
        result = self.graph.invoke(state_before)
        self._validate_agent_result(result)
        return result




    def _build_turn_reply(self, user_text: str, state_before: dict, result: dict) -> str | None:
        # 검증된 정책 응답과 자유응답, Python 다음 행동을 길이 절단 없이 조립한다.
        return build_turn_reply(user_text, state_before, result, self.generate_reply, warning=self.get_logger().warning, scene_reply=lambda: self._describe_scene_reply(user_text))

    def _describe_scene_reply(self, user_text: str) -> str:
        # 자유응답 Tool이 현재 카메라 화면 설명을 요청했을 때만 사용한다.
        if not ENABLE_VLM:
            return "현재 카메라 화면 확인 기능을 준비하고 있어요."

        camera_images = self._snapshot_camera_images()

        if self.call_vlm is None or not camera_images:
            return "아직 확인할 카메라 화면이 없어요. 화면이 들어오면 다시 확인할게요."

        system_prompt = (
            "너는 로봇 카메라의 현재 장면을 손님에게 설명한다. "
            "화면에 실제로 보이는 내용만 한국어 한두 문장으로 답한다. "
            "보이지 않는 물체나 작업 결과를 추측하지 않는다. "
            "PASS, FAIL 같은 판정 문장은 출력하지 않는다."
        )
        scene_reply = self.call_vlm([camera_images[-1]], system_prompt, user_text)

        return scene_reply or "카메라 화면을 정확히 설명하지 못했어요."


    def _process_pending_confirmation(self, user_text: str) -> bool:
        # 현재 section 전체 제외 질문의 긍정·부정을 Tool 모델보다 먼저 판정한다.
        if self.awaiting_confirm is None:
            return False

        self._set_stt_enabled(False)
        confirmation_intent = classify_confirmation_intent(user_text, "section_skip")

        if confirmation_intent is None:
            self.history.append({"role": "user", "content": user_text})
            self._publish_reply("제외할지 유지할지 네 또는 아니요로 말씀해 주세요.")
            self._set_stt_enabled(True)
            return True

        pending = copy.deepcopy(self.awaiting_confirm)
        self.awaiting_confirm = None
        self.runtime_turn_index += 1
        self.history.append({"role": "user", "content": user_text})

        action = f"{pending['type']}_{'accept' if confirmation_intent else 'reject'}"
        self.action_history.append({"turn": self.runtime_turn_index, "action": action})
        self.action_history = self.action_history[-12:]

        if pending["type"] != "refuse_section":
            raise ValueError(f"알 수 없는 재확인 종류 : {pending['type']}")

        section = pending["section"]

        if section != self.section:
            raise RuntimeError(f"재확인 section이 현재 section과 다름 : {section} != {self.section}")

        if confirmation_intent is False:
            if pending.get("source") == "all_options_restricted":
                dislike_items = [item for item in pending["items"] if (strongest_restriction_for(self.order, item) or {}).get("reason") == "dislike"]
                item_text = "나 ".join(dislike_items)
                reply = f"{SECTION_LABELS[section]} 단계를 제외하지 않을게요. {item_text}은 취향 제한이라 원하면 재료와 양을 다시 말씀해 주세요."
            else:
                reply = f"{SECTION_LABELS[section]} 단계에서 계속 고를게요. {self._section_prompt()}"

            self._publish_reply(reply)
            self._set_stt_enabled(True)
            return True

        current_state = self._build_agent_state(user_text)
        committed = commit_section_skip(current_state["order"], current_state["execution"], current_state["recommendation"], pending)
        commit_result = {
            "order": committed["order"],
            "execution": committed["execution"],
            "recommendation": committed["recommendation"],
            "turn_result": {},
            "policy_reply": None,
        }
        self._validate_agent_result(commit_result)
        self._commit_agent_result(commit_result)
        reply = self._advance_after_section(section)
        self._publish_reply(reply)

        return True


    def _process_turn(self, user_text: str):
        # 한 사용자 발화를 검증하고 응답까지 만든 뒤 실제 상태에 반영한다.
        if not self.conversation_started:
            self.get_logger().warning("주문 대화 시작 전 STT 결과를 무시함")
            return

        if self.order_finished:
            self.get_logger().warning("완료된 주문의 STT 결과를 무시함")
            return

        if self._process_pending_confirmation(user_text):
            return

        if self.active_task is not None or self.task_queue:
            self._set_stt_enabled(False)
            self.get_logger().warning("로봇 작업 중 STT 결과를 무시함")
            return

        self._set_stt_enabled(False)

        state_before = self._build_agent_state(user_text)
        result = self._run_agent_turn(user_text)
        turn_result = result["turn_result"]
        action = turn_result["action"]

        action_event = {"turn": self.runtime_turn_index + 1, "action": action}

        if turn_result["accepted"]:
            action_event["accepted"] = copy.deepcopy(turn_result["accepted"])

        if turn_result["restriction_applied"]:
            action_event["restriction_applied"] = copy.deepcopy(turn_result["restriction_applied"])

        if action == "recommend_order":
            tool_call = result.get("tool_call") or {}
            action_event["scope"] = tool_call.get("changes", {}).get("scope")

        if action == "confirm_section":
            action_event["section"] = self.section

        section_skip = turn_result["section_skip"]

        # 응답 생성까지 성공해야 실제 주문 상태를 반영한다.
        reply = self._build_turn_reply(user_text, state_before, result)
        self._commit_agent_result(result)

        self.runtime_turn_index += 1
        self.history.append({"role": "user", "content": user_text})
        self.action_history.append(action_event)
        self.action_history = self.action_history[-12:]

        if section_skip is not None and section_skip["needs_confirmation"]:
            self.awaiting_confirm = copy.deepcopy(section_skip)
            self.awaiting_confirm["type"] = "refuse_section"

        if section_skip is not None and section_skip["applied"]:
            reply = self._advance_after_section(section_skip["section"])

        confirm = turn_result["confirm_validation"]
        recommendation = turn_result["recommendation_validation"]
        recommendation_confirmed = recommendation["proposal_confirmed"] and self.section in self.recommendation_state.get("confirmed_sections", [])

        if (confirm["requested"] and confirm["allowed"]) or recommendation_confirmed:
            if self._queue_current_section_tasks():
                start_reply = f"{SECTION_LABELS[self.section]} 담기를 시작할게요."
                reply = f"{reply} {start_reply}" if reply else start_reply
            elif self.section == "extra":
                self.skipped_sections["extra"] = "empty_confirm"
                reply = self._advance_after_section("extra")
            else:
                raise RuntimeError(f"{self.section} 확정 후 실행할 작업을 만들지 못함")
        else:
            self._set_stt_enabled(True)

        if reply:
            self._publish_reply(reply)

        self._record_tool_turn(user_text, state_before, result, reply)

    def _record_tool_turn(self, user_text: str, state_before: dict, result: dict, reply: str | None):
        # 모델 Tool 출력, Python 검증 결과, 실제 발행 응답을 같은 행에 기록한다.
        if not ENABLE_RUNTIME_LOG:
            return

        try:
            log_row = {
                "schema_version": 5,
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
                "turn_result": copy.deepcopy(result.get("turn_result")),
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
        # ROS 작업 처리 뒤 당시 로봇 실행 상태를 별도 이벤트 로그에 기록한다.
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


    def _set_stt_enabled(self, enabled: bool):
        # False는 즉시 입력 차단, True는 TTS 종료 뒤 다음 입력 허용 예약이다.
        self.stt_enable_pub.publish(Bool(data=enabled))
        self.get_logger().info(f"STT 상태 : {'열기 대기' if enabled else '닫기'}")


    def _publish_reply(self, reply: str):
        # 실제 UI·TTS로 발행한 문장만 assistant 멀티턴 history에 저장한다.
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("발행할 reply가 빈 문자열임")

        reply = reply.strip()
        self.reply_pub.publish(String(data=reply))
        self.history.append({"role": "assistant", "content": reply})

    def _build_selected_status_items(self) -> list:
        # 내부 주문값을 손님 화면에서 읽을 한국어 목록으로 변환한다.
        amount_word = {"low": "적게", "normal": "보통", "high": "많이"}
        reason_word = {"allergy": "알레르기", "cannot_eat": "섭취 불가", "dislike": "비선호", "dietary_rule": "식단 조건"}
        selected_items = []

        if self.order["sauce"]:
            selected_items.append(f"{self.order['sauce']} 소스")

        if self.order["noodle_type"]:
            portion = amount_word.get(self.order["noodle_portion"], "")
            selected_items.append(f"{self.order['noodle_type']} {portion}".strip())

        for topping, amount in self.order["toppings"].items():
            selected_items.append(f"{topping} {amount_word.get(amount, amount)}")

        for restriction in self.order["restrictions"]:
            target = restriction["target"]
            reason = reason_word.get(restriction["reason"], restriction["reason"])
            selected_items.append(f"제외 조건 : {target} ({reason})")

        return selected_items


    def _build_recommendation_ui_state(self, section_index: int, section_label: str) -> tuple:
        # 추천 대기·확정·자동 실행 상태를 UI 표시값으로 변환한다.
        confirmed_sections = self.recommendation_state.get("confirmed_sections", [])

        if not isinstance(confirmed_sections, list):
            confirmed_sections = []

        auto_running = bool(self.section in confirmed_sections and (self.active_task is not None or self.task_queue))
        recommendation_phase = self.recommendation_state.get("phase", "idle")
        covered_sections = self.recommendation_state.get("covered_sections", [])

        if not isinstance(covered_sections, list):
            covered_sections = []

        visible_confirmed_sections = [name for name in confirmed_sections if name in SECTION_ORDER and SECTION_ORDER.index(name) >= section_index and name not in self.skipped_sections]

        if self.order_finished:
            return auto_running, "none", []

        if recommendation_phase == "await_scope":
            return auto_running, "await_scope", []

        if recommendation_phase == "confirming":
            recommendation_sections = [SECTION_LABELS[name] for name in covered_sections if name in SECTION_LABELS]
            return auto_running, "confirming", recommendation_sections

        if auto_running:
            return auto_running, "running", [section_label]

        if visible_confirmed_sections:
            recommendation_sections = [SECTION_LABELS[name] for name in visible_confirmed_sections]
            return auto_running, "confirmed", recommendation_sections

        return auto_running, "none", []


    def _build_work_ui_state(self, completed_sections: list, work_state_override=None) -> dict:
        # 실제 로봇 작업과 VLM 판정 결과를 주문 선택값과 분리해 표시한다.
        ui_group_names = {"noodle": "noodle", "veggie": "vegetable", "meat": "meat", "extra": "extra", "lid": "cover", "sauce": "sauce"}
        current_class = self.active_task["class"] if self.active_task is not None else ""
        selected_classes = []

        if self.order["noodle_type"]:
            selected_classes.append(self.order["noodle_type"])

        selected_classes.extend(self.order["toppings"].keys())

        # policy_success는 재시도 종료 정책이며 실제 VLM PASS가 아니다.
        filled_classes = []

        for verification in self.verification_history:
            class_name = verification.get("class")

            if verification.get("result") == "pass" and isinstance(class_name, str) and class_name and class_name not in filled_classes:
                filled_classes.append(class_name)

        current_vlm_result = None

        if self.active_task is not None:
            for verification in reversed(self.verification_history):
                if verification.get("class") == self.active_task["class"]:
                    current_vlm_result = verification.get("result")
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

        return {
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


    def _publish_status(self, work_state_override=None):
        # 현재 주문·추천·로봇 상태를 기존 UI JSON 계약으로 발행한다.
        section_index = SECTION_ORDER.index(self.section)
        section_label = SECTION_LABELS[self.section]

        if self.order_finished:
            status_text = "포장 완료"
            completed_sections = list(SECTION_ORDER)
        elif isinstance(self.awaiting_confirm, dict):
            status_text = "현재 단계 제외 확인 중"
            completed_sections = list(SECTION_ORDER[:section_index])
        elif self.active_task is not None:
            status_text = f"{self.active_task['class']} 담는 중"
            completed_sections = list(SECTION_ORDER[:section_index])
        elif self.ui_started:
            status_text = "주문 시작 대기 중"
            completed_sections = []
        else:
            status_text = f"{section_label} 선택 중"
            completed_sections = list(SECTION_ORDER[:section_index])

        current_task = self.active_task["class"] if self.active_task is not None else section_label
        auto_running, recommendation_status, recommendation_sections = self._build_recommendation_ui_state(section_index, section_label)
        work_ui_state = self._build_work_ui_state(completed_sections, work_state_override)

        ui_json = {
            "phase": "완료" if self.order_finished else "주문중",
            "section": section_label,
            "section_status": "완료" if self.order_finished else "진행 중",
            "current_task": current_task,
            "target_list": [SECTION_LABELS[name] for name in SECTION_ORDER],
            "completed": [],
            "selected": self._build_selected_status_items(),
            "auto_running": auto_running,
            "recommendation_status": recommendation_status,
            "recommendation_sections": recommendation_sections,
            "skipped": [SECTION_LABELS[name] for name in SECTION_ORDER if name in self.skipped_sections],
            "status_text": status_text,
            "completion_reply": ORDER_COMPLETE_REPLY if self.order_finished else None,
            **work_ui_state,
        }

        for task in self.completed_tasks:
            ui_json["completed"].append(f"{task['class']} × {task['repeat_count']}")

        for section_name in completed_sections:
            if section_name not in self.skipped_sections:
                ui_json["completed"].append(SECTION_LABELS[section_name])

        self.status_pub.publish(String(data=json.dumps(ui_json, ensure_ascii=False)))

    def _get_vlm_reference(self, expected: str):
        # 재료 참고 이미지를 RAG에서 한 번만 읽고 이후 판정에서 재사용한다.
        if expected == "뚜껑":
            return None

        reference_image = self.vlm_reference.get(expected)

        if reference_image is None and self.vlm_rag is not None:
            reference_image = self.vlm_rag.get_reference(expected)

            if reference_image is not None:
                self.vlm_reference[expected] = reference_image

        return reference_image


    def _publish_vlm_ui_snapshot(self, expected: str, request: dict | None):
        # REFERENCE | PREVIOUS | CURRENT 이미지를 한 장으로 합쳐 UI에 발행한다.
        if not ENABLE_VLM_UI_IMAGES or self.vlm_ui_image_pub is None or request is None:
            return

        try:
            request_images = request.get("images")

            if not isinstance(request_images, list) or not request_images:
                return

            reference_image = None
            previous_image = None
            current_image = request_images[-1]

            if expected == "뚜껑":
                if len(request_images) >= 2:
                    previous_image = request_images[0]
            else:
                reference_image = request_images[0]

                if self.comparison_image is not None and len(request_images) >= 3:
                    previous_image = request_images[1]

            panel_width = 480
            panel_height = 360
            snapshot = PILImage.new("RGB", (panel_width * 3, panel_height), (28, 32, 38))

            for panel_index, source_image in enumerate([reference_image, previous_image, current_image]):
                if source_image is None:
                    continue

                panel_image = source_image.convert("RGB").copy()
                panel_image.thumbnail((panel_width, panel_height), PILImage.Resampling.LANCZOS)
                panel_left = panel_index * panel_width + (panel_width - panel_image.width) // 2
                panel_top = (panel_height - panel_image.height) // 2
                snapshot.paste(panel_image, (panel_left, panel_top))

            jpeg_buffer = BytesIO()
            snapshot.save(jpeg_buffer, format="JPEG", quality=80, optimize=True)

            snapshot_message = CompressedImage()
            snapshot_message.header.stamp = self.get_clock().now().to_msg()
            snapshot_message.header.frame_id = "vlm_ui_snapshot"
            snapshot_message.format = "jpeg"
            snapshot_message.data = jpeg_buffer.getvalue()
            self.vlm_ui_image_pub.publish(snapshot_message)
        except Exception as error:
            # UI 이미지 실패는 실제 로봇 판정을 중단시키지 않는다.
            self.get_logger().warning(f"VLM UI 이미지 발행 실패 : {error}")


    def _process_vlm_confirm(self, should_confirm: bool):
        # 로봇 작업 완료 신호를 VLM 판정, 재시도 정책, 외부 결과 발행으로 연결한다.
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

        # VLM OFF에서는 실제 PASS 이력을 만들지 않고 제어 사이클만 통과시킨다.
        if not ENABLE_VLM:
            self.vlm_confirmed = True

            if ENABLE_DUMMY_VLM_PASS:
                self.verification_history.append({"class": expected, "attempt": 1, "verdict": "dummy_pass", "result": "pass"})

            self.vlm_result_pub.publish(String(data="success"))
            mode = "dummy PASS 기록" if ENABLE_DUMMY_VLM_PASS else "실제 PASS 기록 없음"
            self.get_logger().info(f"VLM OFF 우회 결과 발행 : success, {mode}")
            return

        self._publish_status("vlm_checking")
        camera_images = self._snapshot_camera_images()
        request = None
        raw_text = ""

        try:
            reference_image = self._get_vlm_reference(expected)
            request = build_vlm_request(expected, camera_images, reference_image=reference_image, comparison_image=self.comparison_image, comparison_source=self.comparison_source)
            self._publish_vlm_ui_snapshot(expected, request)

            if self.call_vlm is None or request is None:
                verdict = "uncertain"
            else:
                raw_text = self.call_vlm(request["images"], request["system_prompt"], request["user_text"])
                verdict = parse_vlm_verdict(raw_text)
        except Exception as error:
            raw_text = f"ERROR: {error}"
            verdict = "uncertain"
            self.get_logger().error(f"VLM 판정 실패 : {error}")

        previous_failures = sum(1 for verification in self.verification_history if verification.get("class") == expected and verification.get("result") == "fail")
        outcome = decide_vlm_outcome(expected, verdict, previous_failures, ENABLE_UNCERTAIN_RETAKE)

        self.verification_history.append({
            "class": expected,
            "attempt": outcome["attempt"],
            "verdict": verdict,
            "result": outcome["history_result"],
        })

        if outcome["clear_camera_images"]:
            with self.camera_lock:
                self.latest_camera_message = None
                self.vlm_camera_messages.clear()

        if outcome["use_current_as_comparison"]:
            if camera_images:
                current_image = camera_images[-1].copy()
                self.comparison_image = current_image
                self.comparison_source = "pass" if outcome["trusted_pass"] else "retried"
                self.comparison_task_class = expected

                if outcome["trusted_pass"]:
                    self.previous_success_image = current_image.copy()
            else:
                self.comparison_image = None
                self.comparison_source = None
                self.comparison_task_class = None

        self.vlm_confirmed = outcome["confirmed"]

        if outcome["publish_result"] == "fail":
            fail_result = {"result": "fail", "class": TOPIC_CLASS_NAMES[expected]}
            self.vlm_result_pub.publish(String(data=json.dumps(fail_result, ensure_ascii=False)))
        elif outcome["publish_result"] == "success":
            self.vlm_result_pub.publish(String(data="success"))

        self._record_runtime_event("vlm_result", {
            "class": expected,
            "attempt": outcome["attempt"],
            "verdict": verdict,
            "policy_result": outcome["policy_result"],
            "comparison_source": self.comparison_source,
            "current_frame_count": len(camera_images),
            "raw_text": raw_text[-1000:],
        })

        spoken_reply = build_vlm_spoken_reply(raw_text, outcome["policy_reply"], VLM_JUDGE_REASON_SPEAK, outcome["allow_spoken_reason"], VLM_JUDGE_REASON_MAX_CHARS)
        self._publish_reply(spoken_reply)

    def _process_reset(self, data):
        # 완료 화면 reset과 주문 시작 전 back/home reset을 구분한다.
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
        # 마지막 소스 작업과 VLM 확인이 끝난 경우에만 주문 전체를 완료한다.
        if self.section != "sauce" or self.active_task is None:
            self.get_logger().warning("마지막 소스 작업 중이 아니어서 /llm/reset을 무시함")
            return

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
        self._publish_reply(ORDER_COMPLETE_REPLY)
        self.done_pub.publish(Bool(data=True))
        self.get_logger().info("총 주문 완료")


    def _prepare_order(self):
        # UI 시작과 첫 /llm/next 신호가 모두 도착하면 실제 주문 대화를 시작한다.
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
        # 인사 TTS가 끝난 뒤 첫 손님 발화를 받을 수 있도록 STT를 연다.
        self.ui_started = False
        self.pending_initial_next = False
        self.conversation_started = True
        self._set_stt_enabled(True)

        reply = (
            "안녕하세요! 저는 스파게티 밀키트 주문을 도와드리는 로봇이에요. "
            "먼저 면 종류와 소스, 원하시는 면 양을 말씀해 주세요."
        )
        self._publish_reply(reply)
        self.get_logger().info("새 주문 시작")

    def _publish_next_task(self) -> bool:
        # active 작업을 덮어쓰지 않고 queue의 첫 작업만 ROS plan으로 발행한다.
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

        self.task_queue.pop(0)
        self.active_task = copy.deepcopy(task)
        self.robot_started = True
        self.vlm_confirmed = False

        # plan 발행 전 카메라 frame이 이번 작업 판정에 섞이지 않도록 비운다.
        if ENABLE_VLM:
            with self.camera_lock:
                self.latest_camera_message = None
                self.vlm_camera_messages.clear()

        plan_for_publish = {"class": TOPIC_CLASS_NAMES[task_class], "repeat_count": repeat_count}
        self.plan_pub.publish(String(data=json.dumps(plan_for_publish, ensure_ascii=False)))
        self.get_logger().info(f"plan 발행 : {plan_for_publish}, 내부 작업 : {self.active_task}, 남은 작업 : {len(self.task_queue)}")

        return True


    def _queue_current_section_tasks(self) -> bool:
        # 현재 주문을 해당 section의 로봇 작업 목록으로 한 번만 변환한다.
        if self.active_task is not None or self.task_queue:
            self.get_logger().warning("이미 실행할 작업이 있어서 section task를 다시 만들지 않음")
            return False

        tasks = build_section_plan(self.order, self.section)

        if not tasks:
            return False

        if self.section == "noodle":
            queued_keys = ["noodle_type", "noodle_portion"]
        elif self.section in ("veggie", "meat", "extra"):
            queued_keys = [f"toppings.{task['class']}" for task in tasks]
        elif self.section == "sauce":
            queued_keys = ["sauce"]
        else:
            queued_keys = []

        self.selected = [key for key in self.selected if key not in queued_keys]
        self.task_queue.extend(copy.deepcopy(tasks))
        self._set_stt_enabled(False)

        return self._publish_next_task()


    def _section_prompt(self) -> str:
        # 사용자 선택이 필요한 section의 재료와 양을 안내한다.
        if self.section == "veggie":
            return "다음은 야채를 고르실 차례입니다. 양파와 버섯 중 어떤 재료를 어느 정도 양으로 드시고 싶은지 말씀해 주세요."

        if self.section == "meat":
            return "다음은 육류를 고르실 차례입니다. 소시지와 게살 중 어떤 재료를 어느 정도 양으로 드시고 싶은지 말씀해 주세요."

        if self.section == "extra":
            return "다음은 추가 재료를 고르실 차례입니다. 치즈와 페퍼론치노 중 어떤 재료를 어느 정도 양으로 드시고 싶은지 말씀해 주세요."

        raise ValueError(f"사용자 선택 안내가 없는 section임 : {self.section}")


    def _advance_after_section(self, completed_section: str) -> str:
        # 완료 문장 생성과 다음 section 이동을 처리하되 TTS 발행은 호출자가 한 번만 한다.
        section_label = SECTION_LABELS[completed_section]

        if completed_section in self.skipped_sections:
            completed_reply = f"{section_label}는 제외했어요."
        elif completed_section == "lid":
            completed_reply = "뚜껑을 닫았어요."
        else:
            completed_reply = f"{section_label} 담기가 끝났어요."

        next_step = next_section(completed_section)

        while next_step in self.skipped_sections:
            self.get_logger().info(f"건너뛴 section 통과 : {next_step}, 이유 : {self.skipped_sections[next_step]}")
            next_step = next_section(next_step)

        if next_step is None:
            raise ValueError(f"{completed_section} 다음 section이 없음")

        self.section = next_step

        if self.section == "lid":
            if not self._queue_current_section_tasks():
                raise RuntimeError("lid 작업을 만들지 못함")

            return f"{completed_reply} 이제 뚜껑을 닫을게요."

        if self.section == "sauce":
            if not self._queue_current_section_tasks():
                raise RuntimeError("선택된 소스 작업을 만들지 못함")

            return f"{completed_reply} 마지막으로 고르신 소스를 올릴게요."

        confirmed_sections = self.recommendation_state.get("confirmed_sections", [])

        if self.section in confirmed_sections and self._queue_current_section_tasks():
            return f"{completed_reply} 추천한 {SECTION_LABELS[self.section]} 재료 담기를 시작할게요."

        self._set_stt_enabled(True)

        return f"{completed_reply} {self._section_prompt()}"


    def _process_next(self, result_code: int):
        # 첫 next는 주문 시작, 이후 next는 현재 active task의 완료 신호로 처리한다.
        self.get_logger().info(f"/llm/next 수신 : {result_code}")

        if result_code != BY_VLM_NUM:
            self.get_logger().warning(f"규약과 다른 /llm/next 값을 무시함 : {result_code}")
            return

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

        # 마지막 소스 완료는 기존 ROS 계약대로 /llm/reset에서 처리한다.
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
