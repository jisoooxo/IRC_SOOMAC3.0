import copy
import json
import queue
import threading
import time
from collections import deque
from io import BytesIO

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Int16, String

from soomac_irc.model_runtime import load_model, make_call_vlm
from soomac_irc.decision_model import make_call_decision
from soomac_irc.llm_langgraph import build_graph, build_preselected_confirmation_reply, build_preselected_section_confirmation, new_session_state, new_turn_state
from soomac_irc.llm_runtime_logger import LLMSessionJsonlLogger
from soomac_irc.recommendation_model import make_call_recommendation
from soomac_irc.response_model import make_call_response
from soomac_irc.vlm import build_vlm_request, build_vlm_spoken_reply, decide_vlm_outcome, parse_vlm_verdict
# main_vlm.py가 소스 완료 뒤 cover를 직접 실행하고 /llm/reset을 보낸다.
from soomac_irc.domain import SECTION_ORDER


ENABLE_VLM = True
ENABLE_VLM_UI_IMAGES = True
ENABLE_UNCERTAIN_RETAKE = False
VLM_JUDGE_REASON_SPEAK = False

BY_VLM_NUM = 3
IMAGE_TOPIC_TYPE = CompressedImage # Image가 될수도 일단 CompressedImage임
WORLD_CAM_TOPIC = "/vision/overlay_image"
VLM_UI_IMAGE_TOPIC = "/agent/vlm_snapshot"
NUMBER_IMAGE_FOR_CONFIRM = 5
JOB_STOP = "stop"
JOB_MODES = ("start", "turn", "reset", "finish", "confirm", "next") 

AMOUNT_TO_COUNT = {
    "low": 1,
    "normal": 2,
    "high": 3,
}

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


class LLMLangGraphNode(Node):
    def __init__(self, call_decision=None):
        super().__init__("soomac_llm_langgraph_node_0922")

        self.model = None
        self.processor = None
        self.call_vlm = None
        self.vlm_rag = None
        self.vlm_reference = {}

        if call_decision is None:
            # Decision, Recommendation, Response, VLM은 같은 base model을 순차적으로 사용
            self.get_logger().info("Agentic LLM base model 로딩중")
            self.model, self.processor = load_model(None)
            call_decision = make_call_decision(self.model, self.processor, self.get_logger())
            call_recommendation = make_call_recommendation(self.model, self.processor, self.get_logger())
            call_response = make_call_response(self.model, self.processor, self.get_logger())
            self.get_logger().info("Agentic LLM base model 로딩 완료")
        else:
            call_recommendation = getattr(call_decision, "call_recommendation", None)
            call_response = getattr(call_decision, "call_response", None)

        if call_recommendation is None or call_response is None:
            raise ValueError("Node 생성에는 Decision, Recommendation, Response 호출 함수가 모두 필요함")

        if ENABLE_VLM and self.model is not None:
            from soomac_irc.vlm_rag import VlmRag

            self.call_vlm = make_call_vlm(self.model, self.processor, self.get_logger())
            self.vlm_rag = VlmRag()

        ## 이하 llm agents
        self.call_decision = call_decision
        self.call_recommendation = call_recommendation
        self.call_response = call_response
        self.graph = build_graph(call_decision, call_recommendation, call_response)


        # 메인 스레드 -> callback 오면 큐에 ㄱㄱ
        self._jobs : queue.Queue[tuple[str, object]] = queue.Queue(maxsize=32)
        self._worker_stop = threading.Event()

        """
        threading.Event 객체는 Python에서 멀티쓰레딩 환경에서 스레드 간의 신호를 주고받기 위한 동기화 프리미티브
        Event 객체는 내부적으로 불리언 플래그를 유지하며, 플래그가 설정(set)되었는지 여부를 스레드들이 확인하고 대기할 수 있다.
        이를 통해 여러 스레드 간의 통신 및 동기화를 간편하게 할 수 있다.
        """
                # 최근 프레임만 보관하도록

        self.camera_lock = threading.Lock()
        self.latest_camera_message = None
        self.vlm_camera_messages = deque(maxlen=NUMBER_IMAGE_FOR_CONFIRM)
        self._clear_state()

        """ clear state시 생성
        self.graph_state = new_session_state()

        self.section = "noodle"
        self.task_queue = []
        self.active_task = None
        self.completed_tasks = []
        self.robot_started = False

        self.ui_started = False
        self.pending_initial_next = False
        self.conversation_started = False
        self.order_finished = False

        self.vlm_confirmed = False
        self.verification_history = []

        # 실제 PASS 또는 재시도 종료 장면을 다음 작업의 비교 이미지로 사용한다.
        self.comparison_image = None
        self.comparison_source = None
        self.comparison_task_class = None

        with self.camera_lock:
            self.latest_camera_message = None
            self.vlm_camera_messages.clear()

        """

        self.runtime_log = LLMSessionJsonlLogger(logger=self.get_logger())

        ########## publisher 모음집
        self.reply_pub = self.create_publisher(String, "/llm_response", 10)
        self.plan_pub = self.create_publisher(String, "/llm/plan", 10)
        self.done_pub = self.create_publisher(Bool, "/llm/done", 10)
        self.status_pub = self.create_publisher(String, "/agent/status", 10)
        self.stt_enable_pub = self.create_publisher(Bool, "/stt/enable", 10)
        self.vlm_result_pub = self.create_publisher(String, "/vlm/confirm", 10)
        self.vlm_ui_image_pub = None
        self.rail_move_ahead_pub = self.create_publisher(String, "/rail/move_ahead", 10) # 미리 이동하는 레일 용도의 토픽

        if ENABLE_VLM and ENABLE_VLM_UI_IMAGES:
            self.vlm_ui_image_pub = self.create_publisher(CompressedImage, VLM_UI_IMAGE_TOPIC, 1)

        ############ subscriber 모음집

        self.stt_sub = self.create_subscription(String, "/stt_question", self.stt_callback, 10)
        self.ui_start_sub = self.create_subscription(String, "/ui/start", self.ui_start_callback, 10)
        self.ui_reset_sub = self.create_subscription(String, "/ui/reset", self.ui_reset_callback, 10)
        self.next_trigger_sub = self.create_subscription(Int16, "/llm/next", self.trigger_callback, 10)
        self.confirm_start_sub = self.create_subscription(Bool, "/llm/confirm_start", self.confirm_start_callback, 10)
        self.llm_reset_sub = self.create_subscription(String, "/llm/reset", self.llm_reset_callback, 10)
        self.camera_sub = None

        if ENABLE_VLM:
            self.camera_sub = self.create_subscription(IMAGE_TOPIC_TYPE, WORLD_CAM_TOPIC, self.camera_callback, qos_profile_sensor_data)

        # 실질적으로 돌아가는 내부 로직용 워커 스레드(콜백은 메인에서, 일은 내부 스레드에서 ㅇㅇ)
        self._worker = threading.Thread(target=self.worker_loop, name="llm_langgraph_worker", daemon=True)
        self._worker.start()
        self.get_logger().info("김도현 박지수 이준미 주재영 수맥 3.0 파이팅")


    # 큐(Queue) : FIFO.(First In, First Out) - 옛날 기억으론 양쪽 뻥 뚫린 원통 생각하면 될 듯
    # Enqueue : 데이터를 입력하는 함수
    # Dequeue : 데이터를 출력하는 함수

    def _enqueue_job(self, mode: str, data)-> bool:
        # queue는 필요하고 이 함수는 여섯 callback의 put_nowait·queue.Full 처리를 한 곳에서 처리하기 위함
        try:
            self._jobs.put_nowait((mode, data)) # 큐 내부에 

            """
            put_nowait : (Queue)가 가득 차 있을 때 대기(blocking)하지 않고 즉시 queue — 동기화된 큐 클래스 예외를 발생시키는 비동기/동기 메서드
            논블로킹(Non-blocking): 큐에 빈 자리가 없어도 코드가 멈추지 않고 즉시 실행된다.
            """
            return True
        except queue.Full:
            self.get_logger().error(f"LLM 작업 queue가 가득 차서 {mode} 입력을 받지 못함")
            return False

############################### 콜백을 메인스레드에서 돌리고 큐 안에 해야할 것들 넣어둠 #########################

    def ui_start_callback(self, msg: String):
        self._enqueue_job("start", msg.data)

    def stt_callback(self, msg: String):
        self._enqueue_job("turn", msg.data)

    def ui_reset_callback(self, msg: String):
        self._enqueue_job("reset", msg.data)

    def llm_reset_callback(self, msg: String):
        self._enqueue_job("finish", msg.data)

    def confirm_start_callback(self, msg: Bool):
        self._enqueue_job("confirm", msg.data)

    def trigger_callback(self, msg: Int16):
        self._enqueue_job("next", msg.data)

    def camera_callback(self, msg: CompressedImage | Image):
        if not msg.data:
            self.get_logger().warning("빈 카메라 frame을 무시함")
            return

        with self.camera_lock:
            self.latest_camera_message = msg
            self.vlm_camera_messages.append(msg)


################################ VLM에 넣기 위해 compressed image를 PIL 형식으로 변환 ####################

    def _image_message_to_pil(self, message: Image | CompressedImage):
        # ROS 이미지를 Gemma processor가 받을 PIL RGB 이미지로 변환한다.
        if message is None:
            return None

        if isinstance(message, CompressedImage):
            if not message.data:
                return None

            try:
                with PILImage.open(BytesIO(message.data)) as image:
                    return image.convert("RGB").copy()
            except (OSError, ValueError) as error:
                self.get_logger().warning(f"압축 이미지 변환 실패 : {error}")
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

        rows = image_bytes[:required_size].reshape(message.height, message.step)
        pixels = rows[:, :pixel_row_size].reshape(message.height, message.width, channels)

        if message.encoding == "bgr8":
            pixels = pixels[:, :, ::-1]
        elif message.encoding == "rgba8":
            pixels = pixels[:, :, :3]
        elif message.encoding == "bgra8":
            pixels = pixels[:, :, [2, 1, 0]]

        return PILImage.fromarray(pixels.copy())


#################### VLM 판정중에 다른 프레임이 들어오지 않도록 메시지 고정 ###################

    def _snapshot_camera_images(self) -> list:
        with self.camera_lock: # 스레딩 락 ㅇㅇ
            camera_messages = list(self.vlm_camera_messages) 

            if not camera_messages and self.latest_camera_message is not None:
                camera_messages = [self.latest_camera_message]

        camera_images = []

        for message in camera_messages:
            image = self._image_message_to_pil(message)

            if image is not None:
                camera_images.append(image)

        return camera_images


#################### 래퍼런스 사진 가져오기 #######################

# 뚜껑은 래퍼런스 X

    def _get_vlm_reference(self, expected: str):
        # 뚜껑은 reference가 필요 없고, 나머지는 RAG 결과를 재사용한다.
        if expected == "뚜껑" or self.vlm_rag is None:
            return None

        reference = self.vlm_reference.get(expected)

        if reference is None:
            reference = self.vlm_rag.get_reference(expected)

            if reference is not None:
                self.vlm_reference[expected] = reference

        return reference


################ 래퍼런스 + 이전 이미지 + 현재 이미지 ui에 쏴버림 #################

    def _publish_vlm_ui_snapshot(self, request: dict | None):
        # UI에는 이번 판정에 사용한 가장 최신 이미지만 JPEG로 보낸다.
        if not ENABLE_VLM_UI_IMAGES or self.vlm_ui_image_pub is None or request is None:
            return

        images = request.get("images", [])

        if not images:
            return

        jpeg_buffer = BytesIO()
        images[-1].convert("RGB").save(jpeg_buffer, format="JPEG", quality=80)

        message = CompressedImage()
        message.format = "jpeg"
        message.data = jpeg_buffer.getvalue()
        self.vlm_ui_image_pub.publish(message)


    def _process_vlm_confirm(self, should_confirm: bool):
        # /llm/confirm_start는 현재 active_task의 로봇 작업이 끝났다는 신호
        # 그니까 검증해야겠지 ㅇㅇ
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

        expected = self.active_task["class"] # 방금 실행한 task 반환

        if not ENABLE_VLM:
            self.vlm_confirmed = True
            self.vlm_result_pub.publish(String(data="success"))
            self.get_logger().info("VLM OFF 우회 결과 발행 : success")
            return

        camera_images = self._snapshot_camera_images() # 이미지 콜백 잠궈버림
        reference_image = self._get_vlm_reference(expected) # task 래퍼런스 이미지 bring up
        request = build_vlm_request(
            expected,
            camera_images,
            reference_image=reference_image,
            comparison_image=self.comparison_image,
            comparison_source=self.comparison_source,
        ) # 리턴값 : reference, 이전꺼, 최신꺼(요청)
        self._publish_vlm_ui_snapshot(request) 

        raw_text = ""

        try:
            if self.call_vlm is None or request is None:
                verdict = "uncertain"
            else:
                raw_text = self.call_vlm(request["images"], request["system_prompt"], request["user_text"]) # 판정 이유
                verdict = parse_vlm_verdict(raw_text) # 판정 결과
        except Exception as error:
            raw_text = f"ERROR: {error}"
            verdict = "uncertain"
            self.get_logger().error(f"VLM 판정 실패 : {error}")

        previous_failures = 0

        for result in self.verification_history:
            if result["class"] == expected and result["result"] == "fail":
                previous_failures += 1

        outcome = decide_vlm_outcome(expected, verdict, previous_failures, ENABLE_UNCERTAIN_RETAKE) 
        history_result = outcome["history_result"]

        self.verification_history.append({
            "class": expected,
            "attempt": outcome["attempt"],
            "verdict": verdict,
            "result": history_result,
        })

        self.graph_state["action_history"].append({
            "type": "vlm_result",
            "class": expected,
            "attempt": outcome["attempt"],
            "verdict": verdict,
            "policy_result": outcome["policy_result"],
        }) # action history 필드 업데이트

        if outcome["use_current_as_comparison"] and camera_images:
            self.comparison_image = camera_images[-1].copy()
            self.comparison_source = "pass" if outcome["trusted_pass"] else "retried"
            self.comparison_task_class = expected

        if outcome["clear_camera_images"]:
            with self.camera_lock:
                self.latest_camera_message = None
                self.vlm_camera_messages.clear()

        self.vlm_confirmed = outcome["confirmed"]
        publish_result = outcome["publish_result"]

        if publish_result == "fail":
            payload = {"result": "fail", "class": TOPIC_CLASS_NAMES[expected]}
            self.vlm_result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        elif publish_result == "success":
            self.vlm_result_pub.publish(String(data="success"))

        reply = build_vlm_spoken_reply(
            raw_text,
            outcome["policy_reply"],
            speak_reason=VLM_JUDGE_REASON_SPEAK,
            allow_reason=outcome["allow_spoken_reason"],
        ) 
        self._publish_reply(reply) 
        self.get_logger().info(
            f"VLM 결과 : class={expected}, verdict={verdict}, "
            f"policy={outcome['policy_result']}, attempt={outcome['attempt']}"
        )

        # 로그 기록

        self.runtime_log.log_event(
            "vlm_result",
            {
                "class": expected,
                "request": {
                    "system_prompt": request.get("system_prompt") if request else None,
                    "user_text": request.get("user_text") if request else None,
                    "image_count": len(request.get("images", [])) if request else 0,
                },
                "raw": raw_text,
                "verdict": verdict,
                "outcome": copy.deepcopy(outcome),
                "session": copy.deepcopy(self.graph_state),
                "robot_state": self._build_robot_state(),
            },
        )

##################### 시작 초기에 변수 반환, 그리고 끝날때 초기화 ###################

    def _clear_state(self):
        # 초기화
        self.graph_state = new_session_state()

        self.section = "noodle"
        self.task_queue = []
        self.active_task = None
        self.completed_tasks = []
        self.robot_started = False

        self.ui_started = False
        self.pending_initial_next = False
        self.conversation_started = False
        self.order_finished = False

        self.vlm_confirmed = False
        self.verification_history = []

        # 실제 PASS 또는 재시도 종료 장면을 다음 작업의 비교 이미지로 사용
        self.comparison_image = None
        self.comparison_source = None
        self.comparison_task_class = None

        with self.camera_lock:
            self.latest_camera_message = None
            self.vlm_camera_messages.clear()


################### 턴 관련 #######################

    def _reset_model_trace_buffers(self):
        # 한 사용자 turn에 실제 호출된 모델 trace만 모으기 위해 이전 trace를 비운다.
        for call in (self.call_decision, self.call_recommendation, self.call_response):
            trace_events = getattr(call, "trace_events", None)
            if isinstance(trace_events, list):
                trace_events.clear()

    def _collect_model_traces(self) -> dict:
        # 모델 wrapper가 남긴 exact input/raw/output trace를 복사해 JSONL record로 넘긴다.
        return {
            "decision": copy.deepcopy(getattr(self.call_decision, "trace_events", [])),
            "recommendation": copy.deepcopy(getattr(self.call_recommendation, "trace_events", [])),
            "response": copy.deepcopy(getattr(self.call_response, "trace_events", [])),
        }

    def _log_ignored_turn(self, user_text: str, reason: str):
        self.runtime_log.log_event(
            "turn_ignored",
            {
                "reason": reason,
                "user_text": user_text if isinstance(user_text, str) else str(user_text),
                "session": copy.deepcopy(self.graph_state),
                "robot_state": self._build_robot_state(),
            },
        )


######################### 처음 ui start 오면 세션, state, 로그 초기화 ##################### 
# pending initial next는 초기 llm next임



    def _process_start(self):
        # /ui/start를 받아 새 주문 세션을 준비
        if self.ui_started or self.conversation_started or self.active_task is not None:
            self.get_logger().warning("이미 주문 세션이 진행 중이라 /ui/start를 무시함")
            return

        pending_initial_next = self.pending_initial_next
        self._clear_state()
        self.ui_started = True
        log_path = self.runtime_log.start_session(
            copy.deepcopy(self.graph_state),
            self._build_robot_state(),
            reason="ui_start",
        )
        self.get_logger().info(f"LLM session JSONL 시작 : {log_path}")
        self._set_stt_enabled(False)

        if pending_initial_next:
            self.get_logger().info("먼저 도착한 /llm/next로 주문 대화를 시작함")
            self._start_order()

    def _section_prompt(self) -> str:
        # 사용자 입력이 필요한 section 안내
        if self.section == "veggie":
            return "다음은 야채를 고르실 차례입니다. 양파와 버섯 중 원하는 재료와 양을 말씀해 주세요. 원하지 않으면 다음 단계라고 말씀하셔도 돼요."

        if self.section == "meat":
            return "다음은 육류를 고르실 차례입니다. 소시지와 게살 중 원하는 재료와 양을 말씀해 주세요. 원하지 않으면 다음 단계라고 말씀하셔도 돼요."

        if self.section == "extra":
            return "다음은 추가 재료 차례입니다. 치즈와 페퍼론치노 중 원하는 재료와 양을 말씀해 주세요. 원하지 않으면 다음 단계라고 말씀하셔도 돼요."

        raise ValueError(f"사용자 선택 안내가 없는 section : {self.section}")
    
    def _advance_after_section(self, completed_section: str) -> str:
        section_replies = {
            "noodle": "면을 다 담았어요.",
            "veggie": "고르신 야채를 전부 담았어요.",
            "meat": "고르신 육류를 다 담았어요.",
            "extra": "추가 재료를 다 담았어요",
        }

        if completed_section not in SECTION_ORDER:
            raise ValueError(f"알 수 없는 section : {completed_section}")

        next_index = SECTION_ORDER.index(completed_section) + 1

        # 아마 여기에 publish rail_move_ahead_pub을 하면 되지 않을까?
        # 야채면 양파, 육류면 소세지, 추가재료면 치즈(String, json.dumps ㄱㄱ)

        if next_index >= len(SECTION_ORDER):
            raise ValueError(f"{completed_section} 다음 section이 없음")

        self.section = SECTION_ORDER[next_index] # 다음거 ㅇㅇ

###################### 레일 먼저 선수 이동 치도록 일단 추가 #######################

        if self.section == "veggie":
            self.rail_move_ahead_pub.publish(String(data=json.dumps({"class" : "onion"}, ensure_ascii=False)))

        elif self.section == "meat":
            self.rail_move_ahead_pub.publish(String(data=json.dumps({"class" : "sausage"}, ensure_ascii=False)))

        elif self.section == "extra":
            self.rail_move_ahead_pub.publish(String(data=json.dumps({"class" : "cheese"}, ensure_ascii=False)))

##############################################################################

        completed_reply = section_replies[completed_section]

        if self.section == "sauce":
            if not self._start_current_section():
                raise RuntimeError("선택된 소스 작업을 만들지 못함")

            return f"{completed_reply} 마지막으로 고르신 소스를 올릴게요."

        pending = build_preselected_section_confirmation(self.graph_state, self.section)

        if pending is not None:
            self.graph_state["pending_confirmation"] = pending
            self._set_stt_enabled(True)
            return f"{completed_reply} {build_preselected_confirmation_reply(pending)}"

        self._set_stt_enabled(True)
        return f"{completed_reply} {self._section_prompt()}"


#################### llm next 받으면 지금 task 끝내고 다음거 넘김 ####################

    def _process_next(self, result_code: int):
        # 첫 next는 주문 시작, 이후 next는 active task 완료 신호이다.
        self.get_logger().info(f"/llm/next 수신 : {result_code}")

        if type(result_code) is not int or result_code != BY_VLM_NUM:
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

        # 마지막 소스 작업은 기존 규약대로 /llm/reset에서 완료 처리한다.
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

        self.graph_state["action_history"].append({
            "type": "task_completed",
            "section": self.section,
            "task": copy.deepcopy(completed_task),
        })

        self.runtime_log.log_event(
            "task_completed",
            {
                "section": self.section,
                "task": copy.deepcopy(completed_task),
                "session": copy.deepcopy(self.graph_state),
                "robot_state": self._build_robot_state(),
            },
        )
        self.get_logger().info(f"active task 완료 : {completed_task}")

        if self.task_queue:
            self._publish_next_task()
            return

        reply = self._advance_after_section(self.section)
        self._publish_reply(reply)


#################### ui reset 들어오면 초기화 가능한 상태인지 확인 ####################

    def _process_reset(self, data: str):
        # 완료 화면 reset과 시작 전 back/home reset만 처리한다.
        reset_action = None

        try:
            payload = json.loads(data)

            if isinstance(payload, dict):
                reset_action = payload.get("action")
        except (json.JSONDecodeError, TypeError):
            pass

        if reset_action == "complete":
            if not self.order_finished:
                self.get_logger().warning("주문 완료 전이라 complete reset을 무시함")
                return

            self.runtime_log.end_session(
                reason="ui_reset_complete",
                final_session=copy.deepcopy(self.graph_state),
                robot_state=self._build_robot_state(),
            )
            self._clear_state()
            self._set_stt_enabled(False)
            self.get_logger().info("완료된 주문 상태 초기화")
            return

        if reset_action not in ("back", "home"):
            self.get_logger().warning(f"지원하지 않는 /ui/reset action : {reset_action}")
            return

        order = self.graph_state["order"]
        has_order_value = bool(
            order["sauce"]
            or order["noodle_type"]
            or order["noodle_portion"]
            or order["toppings"]
            or order["restrictions"]
        )

        if has_order_value or self.robot_started or self.active_task is not None or self.task_queue or self.completed_tasks:
            self.get_logger().warning("선택된 주문 또는 로봇 작업이 있어서 reset을 무시함")
            return

        self.runtime_log.end_session(
            reason=f"ui_reset_{reset_action}",
            final_session=copy.deepcopy(self.graph_state),
            robot_state=self._build_robot_state(),
        )
        self._clear_state()
        self._set_stt_enabled(False)
        self.get_logger().info(f"시작 전 상태 초기화 : {data}")

#################### 마지막 소스까지 끝나면 주문 전체 완료 처리 ####################

    def _process_finish(self):
        # main이 마지막 소스 작업 완료 후 보내는 /llm/reset 처리
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

        self.graph_state["action_history"].append({
            "type": "task_completed",
            "section": "sauce",
            "task": copy.deepcopy(completed_task),
        })

        self.conversation_started = False
        self.order_finished = True
        self._set_stt_enabled(False)
        self._publish_reply("소스까지 모두 담았어요. 이용해 주셔서 감사합니다.")
        self.done_pub.publish(Bool(data=True))
        self.runtime_log.log_event(
            "task_completed",
            {"section": "sauce", "task": copy.deepcopy(completed_task), "robot_state": self._build_robot_state()},
        )
        self.runtime_log.end_session(
            reason="order_finished",
            final_session=copy.deepcopy(self.graph_state),
            robot_state=self._build_robot_state(),
        )
        self.get_logger().info("총 주문 완료")

#################### ui start랑 첫 next 둘 다 오면 주문 대화 시작 ####################

    def _start_order(self):
        # UI 시작과 첫 next가 모두 확인되면 대화를 시작
        self.ui_started = False
        self.pending_initial_next = False
        self.conversation_started = True
        self._set_stt_enabled(True)

        reply = (
            "안녕하세요! 저는 스파게티 밀키트 주문을 도와드리는 로봇이에요. "
            "먼저 면 종류와 소스, 원하시는 면 양을 말씀해 주세요."
        )
        self._publish_reply(reply)
        self.runtime_log.log_event(
            "conversation_start",
            {"reply": reply, "session": copy.deepcopy(self.graph_state), "robot_state": self._build_robot_state()},
        )
        self.get_logger().info("새 주문 시작")

#################### 주문 가능 상태에 맞춰 stt 열고 닫기 ####################

    def _set_stt_enabled(self, enabled: bool):
        # stt open or not. 열기 대기 True or 즉시 닫기 False. 
        self.stt_enable_pub.publish(Bool(data=enabled))
        self.get_logger().info(f"STT 상태 : {'열기 대기' if enabled else '닫기'}")

#################### 최종 답변 문자열 llm response로 발행 ####################

    def _publish_reply(self, reply: str):
        # 사용자한테 publish할 문자열
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("발행할 reply가 빈 문자열임")

        self.reply_pub.publish(String(data=reply.strip()))

#################### 지금까지 고른 주문을 ui 표시용 한글 목록으로 변환 ####################

    def _build_selected_status_items(self) -> list[str]:
        # 역할: 누적 주문을 UI에서 읽을 수 있는 한국어 목록으로 바꾼다.
        # 입력/상태 변경/반환/호출자: self.graph_state / 없음 / 문자열 목록 / _build_status_payload.
        amount_word = {"low": "적게", "normal": "보통", "high": "많이"}
        reason_word = {
            "allergy": "알레르기",
            "cannot_eat": "섭취 불가",
            "dietary_rule": "식단 조건",
            "dislike": "비선호",
        }
        order = self.graph_state["order"]
        selected = []

        if order["sauce"] is not None:
            selected.append(f"{order['sauce']} 소스")

        if order["noodle_type"] is not None:
            portion = amount_word.get(order["noodle_portion"], "")
            selected.append(f"{order['noodle_type']} {portion}".strip())

        for topping, amount in order["toppings"].items():
            selected.append(f"{topping} {amount_word.get(amount, amount)}")

        for restriction in order["restrictions"]:
            reason = reason_word.get(restriction["reason"], restriction["reason"])
            selected.append(f"제외 조건: {restriction['target']} ({reason})")

        return selected


#################### 주문 로봇 vlm 상태를 ui status json 하나로 조립 ####################

    def _build_status_payload(self, work_state_override=None) -> dict:
        # 역할: Node의 주문·로봇·VLM 상태를 기존 UI JSON 계약으로 만든다.
        # 입력/상태 변경/반환/호출자: 선택적인 work_state / 없음 / status dict / _publish_status.
        section_labels = {
            "noodle": "면",
            "veggie": "야채",
            "meat": "육류",
            "extra": "추가 재료",
            "sauce": "소스",
        }
        group_names = {
            "noodle": "noodle",
            "veggie": "vegetable",
            "meat": "meat",
            "extra": "extra",
            "sauce": "sauce",
        }
        section_items = {
            "veggie": ("양파", "버섯"),
            "meat": ("소시지", "게살"),
            "extra": ("치즈", "페퍼론치노"),
        }

        order = self.graph_state["order"]
        section_index = SECTION_ORDER.index(self.section)
        passed_sections = SECTION_ORDER if self.order_finished else SECTION_ORDER[:section_index]

        completed_groups = []
        skipped_groups = []

        if self.ui_started or self.conversation_started or self.robot_started or self.order_finished:
            completed_groups.append("container")

        for section_name in passed_sections:
            if section_name in section_items:
                has_selected_item = False

                for item in section_items[section_name]:
                    if item in order["toppings"]:
                        has_selected_item = True
                        break

                if not has_selected_item:
                    skipped_groups.append(group_names[section_name])
                    continue

            completed_groups.append(group_names[section_name])

        if self.order_finished and "cover" not in completed_groups:
            completed_groups.append("cover")

        selected_classes = []

        if order["noodle_type"] is not None:
            selected_classes.append(TOPIC_CLASS_NAMES[order["noodle_type"]])

        for topping in order["toppings"]:
            selected_classes.append(TOPIC_CLASS_NAMES[topping])

        filled_classes = []

        for verification in self.verification_history:
            class_name = verification.get("class")

            if verification.get("result") == "pass" and class_name in TOPIC_CLASS_NAMES:
                topic_class = TOPIC_CLASS_NAMES[class_name]

                if topic_class not in filled_classes:
                    filled_classes.append(topic_class)

        current_class = ""
        current_task = section_labels[self.section]

        if self.active_task is not None:
            task_class = self.active_task["class"]
            current_class = TOPIC_CLASS_NAMES[task_class]
            current_task = {
                "item": task_class,
                "repeat_count": self.active_task["repeat_count"],
            }

        current_vlm_result = None

        if self.active_task is not None:
            for verification in reversed(self.verification_history):
                if verification.get("class") == self.active_task["class"]:
                    current_vlm_result = verification.get("result")
                    break

        if work_state_override is not None:
            work_state = work_state_override
        elif self.order_finished:
            work_state = "completed"
        elif self.active_task is None:
            work_state = "idle"
        elif self.vlm_confirmed and current_vlm_result == "pass":
            work_state = "vlm_done"
        elif self.vlm_confirmed:
            work_state = "retried"
        elif current_vlm_result == "fail":
            work_state = "vlm_fail"
        elif current_vlm_result == "uncertain":
            work_state = "uncertain_wait"
        else:
            work_state = "working"

        if self.order_finished:
            phase = "completed"
            section_status = "completed"
            status_text = "포장 완료"
        elif self.active_task is not None:
            phase = "running"
            section_status = "executing"
            status_text = f"{self.active_task['class']} 담는 중"
        elif self.ui_started and not self.conversation_started:
            phase = "ordering"
            section_status = "waiting"
            status_text = "주문 시작 대기 중"
        elif self.conversation_started:
            phase = "ordering"
            section_status = "waiting_input"
            status_text = f"{section_labels[self.section]} 선택 중"
        else:
            phase = "idle"
            section_status = "waiting"
            status_text = "대기 중"

        completed = []

        for task in self.completed_tasks:
            completed.append(f"{task['class']} × {task['repeat_count']}")

        done_sections = []

        for section_name in passed_sections:
            if group_names[section_name] not in skipped_groups:
                done_sections.append(section_labels[section_name])

        skipped = []

        for section_name in SECTION_ORDER:
            if group_names[section_name] in skipped_groups:
                skipped.append(section_labels[section_name])

        has_order_value = bool(
            order["sauce"]
            or order["noodle_type"]
            or order["noodle_portion"]
            or order["toppings"]
            or order["restrictions"]
        )
        reset_allowed = not has_order_value and not self.robot_started and self.active_task is None and not self.task_queue and not self.completed_tasks and not self.order_finished

        return {
            "phase": phase,
            "section": section_labels[self.section],
            "section_status": section_status,
            "current_task": current_task,
            "target_list": [section_labels[name] for name in SECTION_ORDER],
            "completed": completed,
            "done_sections": done_sections,
            "selected": self._build_selected_status_items(),
            "skipped": skipped,
            "status_text": status_text,
            "auto_running": False,
            "current_class": current_class,
            "current_group": group_names[self.section],
            "work_state": work_state,
            "selected_classes": selected_classes,
            "selected_sauce": order["sauce"],
            "filled_classes": filled_classes,
            "completed_groups": completed_groups,
            "skipped_groups": skipped_groups,
            "reset_allowed": reset_allowed,
        }


#################### 완성한 status를 agent status 토픽으로 발행 ####################

    def _publish_status(self, work_state_override=None):
        # 역할: 현재 상태 스냅샷을 기존 /agent/status 계약으로 발행한다.
        # 입력/상태 변경/반환/호출자: 선택적인 work_state / ROS publish / 없음 / worker_loop.
        payload = self._build_status_payload(work_state_override)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

#################### graph에 넣을 로봇 상태는 deepcopy해서 넘김 ####################

    def _build_robot_state(self) -> dict:
        # graph에는 Node 상태의 읽기용 복사본만 전달한다.
        return {
            "section": self.section,
            "task_queue": copy.deepcopy(self.task_queue),
            "active_task": copy.deepcopy(self.active_task),
            "completed_tasks": copy.deepcopy(self.completed_tasks),
            "robot_started": self.robot_started,
        }

#################### 현재 section 주문값을 실제 로봇 task 목록으로 바꿈 ####################

    def _build_current_section_tasks(self) -> list[dict]:
        # 현재 graph 주문을 기존 ROS 작업 단위로 변환한다.
        # cover는 main_vlm.py가 소스 완료 뒤 자동 실행하므로 여기서 만들지 않는다.
        order = self.graph_state["order"]

        if self.section == "noodle":
            noodle_type = order["noodle_type"]
            amount = order["noodle_portion"]

            if noodle_type is None or amount is None:
                return []

            return [{
                "class": noodle_type,
                "repeat_count": AMOUNT_TO_COUNT[amount],
            }]

        if self.section == "veggie":
            section_items = ("양파", "버섯")
        elif self.section == "meat":
            section_items = ("소시지", "게살")
        elif self.section == "extra":
            section_items = ("치즈", "페퍼론치노")
        elif self.section == "sauce":
            sauce = order["sauce"]

            if sauce is None:
                return []

            return [{
                "class": sauce,
                "repeat_count": 1,
            }]
        else:
            return []

        tasks = []

        for item in section_items:
            amount = order["toppings"].get(item)

            if amount in AMOUNT_TO_COUNT:
                tasks.append({
                    "class": item,
                    "repeat_count": AMOUNT_TO_COUNT[amount],
                })

        return tasks

#################### queue 맨 앞 task를 llm plan으로 하나씩 발행 ####################

    def _publish_next_task(self) -> bool:
        # active task를 덮어쓰지 않고 queue 첫 작업만 발행한다.
        if self.active_task is not None or not self.task_queue:
            return False

        task = self.task_queue.pop(0)
        task_class = task["class"]

        self.active_task = copy.deepcopy(task)
        self.robot_started = True
        self.vlm_confirmed = False

        if ENABLE_VLM:
            with self.camera_lock:
                self.latest_camera_message = None
                self.vlm_camera_messages.clear()

        plan = {
            "class": TOPIC_CLASS_NAMES[task_class],
            "repeat_count": task["repeat_count"],
        }

        self.plan_pub.publish(
            String(data=json.dumps(plan, ensure_ascii=False))
        )
        self.runtime_log.log_event(
            "robot_task_started",
            {
                "section": self.section,
                "task": copy.deepcopy(task),
                "plan": copy.deepcopy(plan),
                "robot_state": self._build_robot_state(),
            },
        )
        self.get_logger().info(f"/llm/plan 발행 : {plan}")
        return True

#################### 현재 section task 만들고 첫 번째 작업 바로 시작 ####################

    def _start_current_section(self) -> bool:
        # 현재 section 작업을 한 번 queue에 넣고 첫 작업을 발행한다.
        if self.active_task is not None or self.task_queue:
            self.get_logger().warning("이미 실행 중인 작업이 있음")
            return False

        tasks = self._build_current_section_tasks()

        if not tasks:
            self.get_logger().warning(f"{self.section}에서 실행할 작업이 없음")
            return False

        self.task_queue.extend(copy.deepcopy(tasks)) # 여기서 current section의 task를 반환 받음
        self._set_stt_enabled(False)
        return self._publish_next_task()

#################### 사용자 한 턴을 graph에 넣고 결과를 실제 session에 반영 ####################

    def _process_turn(self, user_text: str):
        # 역할: 한 사용자 발화를 graph에 전달하고 검증된 세션만 실제 상태에 반영한다.
        # 각 turn의 exact LLM input/raw/final, state 전후, policy, reply를 세션 JSONL에 남긴다.
        if not self.conversation_started:
            self.get_logger().warning("주문 대화 시작 전 STT 결과를 무시함")
            self._log_ignored_turn(user_text, "conversation_not_started")
            return

        if self.order_finished:
            self.get_logger().warning("완료된 주문의 STT 결과를 무시함")
            self._log_ignored_turn(user_text, "order_finished")
            return

        if self.active_task is not None or self.task_queue:
            self.get_logger().warning("로봇 작업 중 STT 결과를 무시함")
            self._log_ignored_turn(user_text, "robot_busy")
            self._set_stt_enabled(False)
            return

        if not isinstance(user_text, str) or not user_text.strip():
            self.get_logger().warning("빈 사용자 발화를 무시함")
            self._log_ignored_turn(user_text, "empty_user_text")
            self._set_stt_enabled(True)
            return

        self._set_stt_enabled(False)
        self._reset_model_trace_buffers()
        turn_id = self.runtime_log.next_turn_id()
        turn_started = time.perf_counter()
        session_before = copy.deepcopy(self.graph_state)
        robot_state_before = self._build_robot_state()
        turn_state = new_turn_state(session_before, user_text, robot_state_before)
        result = None
        published_reply = None

        def write_turn(status: str, error: Exception | None = None):
            error_payload = None
            if error is not None:
                error_payload = {
                    "type": type(error).__name__,
                    "message": str(error),
                }

            self.runtime_log.log_turn({
                "turn_id": turn_id,
                "status": status,
                "user_text": user_text.strip(),
                "session_before": session_before,
                "robot_state_before": robot_state_before,
                "llm_trace": self._collect_model_traces(),
                "decision_final": copy.deepcopy(result.get("decision")) if isinstance(result, dict) else None,
                "recommendation_result": copy.deepcopy(result.get("recommendation_result")) if isinstance(result, dict) else None,
                "policy": copy.deepcopy(result.get("policy")) if isinstance(result, dict) else None,
                "graph_reply": result.get("reply") if isinstance(result, dict) else None,
                "published_reply": published_reply,
                "session_after": copy.deepcopy(self.graph_state),
                "robot_state_after": self._build_robot_state(),
                "latency_ms": {
                    "total": round((time.perf_counter() - turn_started) * 1000.0, 3),
                },
                "error": error_payload,
            })

        try:
            result = self.graph.invoke(turn_state)
        except Exception as error:
            self.get_logger().error(f"LLM turn 실패 ({type(error).__name__}): {error}")

            # graph는 graph_state 복사본에서 동작하므로 실패한 결과는 반영하지 않는다.
            if self.conversation_started and not self.order_finished and self.active_task is None and not self.task_queue:
                published_reply = "주문을 처리하지 못했어요. 다시 말씀해 주세요."
                self._publish_reply(published_reply)
                self._set_stt_enabled(True)

            write_turn("graph_error", error)
            return

        self.graph_state = copy.deepcopy(result["session"])

        try:
            if not result["policy"]["execute"]:
                if result["reply"]:
                    published_reply = result["reply"]
                    self._publish_reply(published_reply)

                self._set_stt_enabled(True)
                write_turn("completed")
                return

            if self._start_current_section():
                if result["reply"]:
                    published_reply = result["reply"]
                    self._publish_reply(published_reply)

                write_turn("execution_started")
                return

            # 선택 재료가 없는 optional section은 로봇 작업 없이 바로 다음 section으로 이동한다.
            if self.section in ("veggie", "meat", "extra"):
                reply = self._advance_after_section(self.section)

                if self.graph_state["history"] and self.graph_state["history"][-1]["role"] == "assistant":
                    self.graph_state["history"][-1]["content"] = reply

                published_reply = reply
                self._publish_reply(reply)
                write_turn("section_advanced")
                return

            if result["reply"]:
                published_reply = result["reply"]
                self._publish_reply(published_reply)

            self._set_stt_enabled(True)
            write_turn("completed")
        except Exception as error:
            write_turn("post_graph_error", error)
            raise

#################### callback에서 쌓은 job을 종류별 처리 함수로 분배 ####################

    def _dispatch_job(self, mode: str, data):
        # queue 작업을 한 worker에서 명시적인 상태 처리 함수로 보낸다.
        if mode == "start":
            self._process_start()
        elif mode == "turn":
            self._process_turn(data)
        elif mode == "reset":
            self._process_reset(data)
        elif mode == "finish":
            self._process_finish()
        elif mode == "confirm":
            self._publish_status("vlm_checking")
            self._process_vlm_confirm(data)
        elif mode == "next":
            self._process_next(data)
        else:
            self.get_logger().warning(f"알 수 없는 LLM 작업 종류 : {mode}")

#################### ros callback이 안 막히도록 내부 worker에서 job 순서대로 처리 ####################

    def worker_loop(self):
        while True:
            try:
                mode, data = self._jobs.get(timeout=0.5)
            except queue.Empty:
                if self._worker_stop.is_set():
                    return
                continue

            try:
                if mode == JOB_STOP:
                    return

                self._dispatch_job(mode, data)

            except Exception as error:
                self.get_logger().error(f"LLM worker 오류 ({mode}) : {error}")

            finally:
                if mode != JOB_STOP:
                    try:
                        self._publish_status()
                    except Exception as error:
                        self.get_logger().warning(f"UI status 발행 실패: {error}")

                self._jobs.task_done()

#################### 노드 종료 전에 worker랑 runtime log 안전하게 정리 ####################

    def destroy_node(self):
        if not self._worker_stop.is_set():
            self._worker_stop.set()

            try:
                self._jobs.put_nowait((JOB_STOP, None))
            except queue.Full:
                self.get_logger().warning("종료 sentinel을 넣지 못해 worker의 timeout 종료를 기다림")

            self._worker.join(timeout=2.0)

            if self._worker.is_alive():
                self.get_logger().warning("LLM worker가 2초 안에 종료되지 않음")

        if self.runtime_log.active:
            self.runtime_log.end_session(
                reason="node_shutdown",
                final_session=copy.deepcopy(self.graph_state),
                robot_state=self._build_robot_state(),
            )
        self.runtime_log.close(timeout=2.0)

        return super().destroy_node()


#################### ros node 실행 진입점 ####################

def main(args=None):
    rclpy.init(args=args)
    node = LLMLangGraphNode()

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
    
    