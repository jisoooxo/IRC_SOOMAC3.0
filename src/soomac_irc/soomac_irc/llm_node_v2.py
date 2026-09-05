import queue # 모델 추론은 ros 콜백에서 돌리면 막혀서 스레드랑 큐로 빼야함
import json
import threading
import copy


from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String, Int16, Bool
from sensor_msgs.msg import Image
from collections import deque

from order_v2 import new_order, missing_required, next_section, build_section_plan, validate_delta, apply_delta, enforce_constraints, SECTION_ORDER, SECTION_LABELS, VEGGIES, MEATS, EXTRAS, EARLY_TOPPINGS, CONSTRAINT_BLOCKS
from call_model_v2 import load_model, make_call_model, make_generate_reply, make_call_vlm, SCENE_SYSTEM
from agent_v2 import build_graph, HISTORY_TURNS
from vlm_rag import VlmRag


ENABLE_VLM = True # 일단 LLM부터 완성. False면 VLM 함수 만들지 않음

ENABLE_TOOL_LORA = True # Tool 전용 어뎁터
TOOL_ADAPTER_PATH = "/home/roma/ros2_ws/src/soomac_irc/outputs/gemma4_tool_lora_v3"

ENABLE_RUNTIME_LOG = True
# True: Tool을 사용한 실제 대화를 JSONL로 계속 누적한다.
# False: 파일을 만들거나 기록하지 않는다.

RUNTIME_LOG_PATH = Path.home() / "soomac_runtime_logs" / "tool_turns.jsonl"
# 홈 디렉터리에 저장하므로 ros2_ws와 IRC_SOOMAC3.0 중 어디서 실행해도 같은 파일에 모인다.

SECTION_SEQUENCE_VLM = True
# True : 현재 섹션의 모든 재료를 담은 뒤 VLM 판정
# False : 소스까지 모두 담은 뒤 뚜껑 닫기 전에 전체 VLM 판정

SECTION_START_MODE = "WAIT_USER_CONFIRM"
# WAIT_USER_CONFIRM : 사용자가 "다 골랐어"라고 해야 로봇 작업 시작
# AUTO_WHEN_VALID : 필수값이 다 채워지면 바로 로봇 작업 시작

WAIT_FOR_ROBOT = True # 로봇이 현재 섹션을 다 담은 뒤 다음 섹션으로 넘어감

CONFIRM_LID = False # cover는 main이 바로 실행하므로 손님에게 포장 여부를 다시 묻지 않음

ALLOW_CANCEL_AFTER_ROBOT_START = False
# False : 로봇이 재료를 담기 시작한 뒤에는 음성으로 전체 주문 초기화 금지
# True : 물리 작업이 시작됐어도 재확인 후 주문 상태 초기화

AFFIRM_WORDS = frozenset({"네", "예", "응", "맞아", "맞아요", "그래", "좋아", "그대로 해주세요"})
# 확정 언어
# frozenset은 set과 같은데 한 번 만들면 내부 값을 바꿀 수 없음

WORLD_CAM_TOPIC = "/world_camera/image_raw"
BY_VLM_NUM = 3 # 완료 토픽

# publish 할때만 영어로 바꿈(코드 내에선 한글로 가져감)

TOPIC_CLASS_NAMES = {
    "소시지": "sausage",
    "게살": "crab",
    "넓적면": "noodle_thick",
    "얇은면": "noodle_thin",
    "버섯": "mushroom",
    "치즈": "cheese",
    "페퍼론치노": "pepperoncino",
    "양파": "onion",
    "크림": "sauce_cream",
    "오일": "sauce_oil",
    "토마토": "sauce_tomato",
    "뚜껑": "cover",
}

CONFIRM_MULTI_IMAGE = True # VLM 판단 시 여러 프레임 큐에 저장해서 비교(순간적으로 들어온 1프레임이 아니라 ㅇㅇ)
NUMBER_IMAGE_FOR_CONFIRM = 5 # 이미지 개수


class LLMNode(Node):
    def __init__(self):
        super().__init__("soomac_llm_node_v2")


        if SECTION_START_MODE not in ("WAIT_USER_CONFIRM", "AUTO_WHEN_VALID"):
            raise ValueError(f"SECTION_START_MODE 이상함 : {SECTION_START_MODE}")


        # 대화 전체 상태는 llm node가 들고 있음
        # agent graph는 한 턴만 처리하고 끝남

        self.order = new_order() # order deep copy ㄱㄱ염

        self.section = "noodle" # noodle → veggie → meat → extra → sauce → lid

        self.history = []
        self.action_history = [] # 실제 수행한 history
        self.task_queue = [] # 아직 로봇한테 안 보낸 작업
        self.active_task = None # 지금 로봇이 하고 있는 작업(방금 보낸거)
        self.completed_tasks = [] # 작업 성공까지 확인된 작업을 담을 리스트
        self.verification_history = [] # VLM 최초 실패와 한 번 재시도 결과

        # 섹션 거절·추천·반복 안내 상태
        self.refusal_prompted_section = None
        self.recommendation_state = {}
        self.last_recommendation = {}
        self.remaining_item_prompted_section = None
        self.skipped_sections = set()

        self.waiting_pick = False # 로봇이 현재 섹션을 담는 중인지

        self.robot_started = False # /llm/plan 발행 여부 ㅇㅇ

        # 현재 active_task가 VLM PASS를 받았는지에 대한 여부
        # PASS 뒤에 제어가 home으로 돌아와서 /llm/next=3를 보낼 때 까지 유지
        self.vlm_confirmed = False

        self.awaiting_confirm = None
        # None : 확인 대기 아님
        # final_order : 뚜껑 닫기 전 마지막 확인 대기
        # cancel_order : 전체 주문 취소 확인 대기

        self.order_finished = False

        self.ui_started = False 
        # True : UI 시작 버튼을 눌렀고 main의 첫 /llm/next=3을 기다리는 중
        # False : 아직 UI 시작 전이거나 이미 주문 대화를 시작함

        self.pending_initial_next = False # ui start랑 llm next랑 섞일거 가정해서 예외처리

        self.conversation_started = False
        # True : UI start 뒤 main의 첫 /llm/next=3까지 받아 실제 주문 대화 중
        # False : 시작 전이거나 리셋됨

        # 모델 하나만 로드해서 tool, reply, 나중에 vlm까지 공유

        self.runtime_session_id = ""
        self.runtime_turn_index = 0

        if ENABLE_RUNTIME_LOG:
            RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        self.get_logger().info("모델 로딩중")

        model, processor = load_model(TOOL_ADAPTER_PATH if ENABLE_TOOL_LORA else None)

        call_model = make_call_model(model, processor)

        self.generate_reply = make_generate_reply(model, processor)

        self.graph = build_graph(call_model, self.generate_reply) # 함수 자체를 넘긴다.
        self.call_vlm = None # 얘는 나중에
        self.vlm_reference = {} 
        self.vlm_rag = None

        self.latest_camera_message = None
        self.vlm_camera_messages = deque(maxlen=NUMBER_IMAGE_FOR_CONFIRM)
        self.camera_lock = threading.Lock()
        self.camera_sub = None


        """
        위에는 참고 이미지 박힘

        {
            "양파": 양파_참고이미지,
            "버섯": 버섯_참고이미지,
        }
        """

        if ENABLE_VLM:
            self.call_vlm = make_call_vlm(model, processor) 
            self.vlm_rag = VlmRag()


        self.get_logger().info("모델 로딩 완료")


        # Publish!!

        self.reply_pub = self.create_publisher(String, "/llm_response", 10) # llm 대답
        self.plan_pub = self.create_publisher(String, "/llm/plan", 10) # main한테 현재 로봇이 해야할 작업 전달
        self.done_pub = self.create_publisher(Bool, "/llm/done", 10) # 최종 포장 확정 후 main이랑 ui에 전체 완료 알림
        self.status_pub = self.create_publisher(String, "/agent/status", 10) # 얘는 ui 한테
        self.stt_enable_pub = self.create_publisher(Bool, "/stt/enable", 10) # LLM -> STT, 손님 발화를 받아도 되는 구간인지

        self.vlm_result_pub = self.create_publisher(String, "/vlm/confirm", 10) # vlm 판정 결과를 main에 전달

        # 구독 콜백이 받은 작업을 worker에 넘길 큐
        # 구독 직후 메시지가 들어와도 사용할 수 있도록 구독보다 먼저 만든다.

        self._jobs = queue.Queue()

        # ROS 구독

        self.stt_sub = self.create_subscription(String, "/stt_question", self.stt_callback, 10)
        self.ui_start_sub = self.create_subscription(String, "/ui/start", self.ui_start_callback, 10)
        self.ui_reset_sub = self.create_subscription(String, "/ui/reset", self.ui_reset_callback, 10)
        self.next_trigger_sub = self.create_subscription(Int16, "/llm/next", self.trigger_callback, 10) # main이 제어 작업과 홈 복귀 완료 후 3을 발행
        self.confirm_start_sub = self.create_subscription(Bool, "/llm/confirm_start", self.confirm_start_callback, 10)
        # 이때부터 vlm 판단 시작 -> 이거 받으면 우리쪽에서 판단 후, /vlm/confim 토픽 보냄

        # llm_reset sub

        self.llm_reset_sub = self.create_subscription(String, "/llm/reset", self.llm_reset_callback, 10)

        if ENABLE_VLM:
            self.camera_sub = self.create_subscription(Image, WORLD_CAM_TOPIC, self.camera_callback, qos_profile_sensor_data)


        # 주문 콜백은 상태를 직접 바꾸지 않고 jobs에 작업만 넣음
        # 카메라 콜백만 최신 메시지를 lock으로 교체하고 주문 상태는 건드리지 않음
        # order, history, section은 worker 하나가 순서대로 바꿈

        
        self._worker = threading.Thread(target=self.worker_loop, daemon=True) # daemon=True는 rclpy가 죽으면 worker도 같이 죽게 함

        self._worker.start()


    ## callback 함수
    def ui_start_callback(self, msg: String):
        self._jobs.put(("start", msg.data))


    def stt_callback(self, msg: String):
        self._jobs.put(("turn", msg.data))


    def ui_reset_callback(self, msg: String):
        self._jobs.put(("reset", "UI 홈 또는 뒤로가기"))

    def llm_reset_callback(self, msg: String):
        # main이 cover 작업 완료 후 보내는 최종 완료 신호
        self._jobs.put(("finish", msg.data))

    def confirm_start_callback(self, msg: Bool):
        # main이 제어의 VLM 촬영 준비 완료를 확인한 뒤 True를 보냄
        self._jobs.put(("confirm", msg.data))


    def trigger_callback(self, msg: Int16):
        self._jobs.put(("next", msg.data))

    def camera_callback(self, msg: Image):
        # 카메라는 작업 큐에 계속 쌓지 않고 최신 메시지만 교체한다.
        # 주문 상태는 worker만 수정하고, 카메라 메시지는 lock으로 따로 보호한다.

        with self.camera_lock:
            self.latest_camera_message = msg

            if CONFIRM_MULTI_IMAGE:
                self.vlm_camera_messages.append(msg)


    def _set_stt_enabled(self, enabled: bool):
        # True는 다음 TTS가 끝난 뒤 손님 발화를 받으라는 뜻이다.
        # False는 TTS가 끝나도 로봇 작업 중에는 계속 닫아두라는 뜻이다.
        self.stt_enable_pub.publish(Bool(data=enabled))
        state_text = "열기 대기" if enabled else "닫기"
        self.get_logger().info(f"STT 상태 : {state_text}")


    # 콜백이 넣은 작업을 처리(모델 추론이 길어지면 ros 내부 스레드가 막힐 수 있음)
    def worker_loop(self):
        # 별도 스레드에서 jobs를 하나씩 꺼냄
        # worker가 하나라서 order 수정 순서가 안 꼬임

        while True:

            mode, data = self._jobs.get()

            try:
                if mode == "start":
                    self._prepare_order()

                elif mode == "reset":
                    if self.conversation_started or self.active_task is not None or self.waiting_pick:
                        self.get_logger().warning("주문 또는 로봇 작업 중이라 /ui/reset을 무시함")
                    else:
                        self._reset(data)

                elif mode == "finish":
                    if self.section != "sauce":
                        self.get_logger().warning("소스 작업 중이 아니어서 /llm/reset을 무시함")


                    else:
                        completed_task = self.active_task
                        self.active_task = None

                        if completed_task not in self.completed_tasks and completed_task is not None:
                            self.completed_tasks.append(completed_task.copy())


                        self.section = "lid"

                        self.reply_pub.publish(String(data="포장이 끝났어요. 이용해 주셔서 감사합니다."))
                        self._finish_order()

                elif mode == "confirm":
                    self._process_vlm_confirm(data)

                elif mode == "next":
                    self._process_next(data)

                elif mode == "turn":
                    self._process(data)

            except Exception as exc:
                self.get_logger().error(f"에러 터짐 : {exc}") # 
                self.reply_pub.publish(String(data="처리 중 문제가 생겼어요. 다시 말씀해 주세요."))

            finally:
                self._publish_status()

                self._jobs.task_done()

    def _image_message_to_pil(self, message: Image):
        # PIL(Python Imaging Library)은 Python에서 이미지를 다루는 형식이다.
        # Gemma의 processor는 ROS Image가 아니라 PIL Image를 입력으로 받는다.
        #
        # 변환 흐름:
        # ROS 바이트 → NumPy 배열 → RGB 순서 정리 → PIL 이미지

        # 카메라 이미지가 아직 들어오지 않은 경우
        if message is None:
            return None

        # RGB/BGR은 픽셀 하나가 색상 3개로 구성된다.
        # 예: 빨강 픽셀 RGB = [255, 0, 0]
        if message.encoding in ("rgb8", "bgr8"):
            channels = 3

        # RGBA/BGRA는 RGB 색상 3개와 투명도 Alpha 1개로 구성된다.
        elif message.encoding in ("rgba8", "bgra8"):
            channels = 4

        # 현재 코드가 모르는 이미지 형식이면 판정하지 않는다.
        else:
            return None

        # 실제 픽셀만 계산했을 때 이미지 한 줄에 필요한 바이트 수
        # 예: 너비 640 × RGB 채널 3개 = 1920바이트
        pixel_row_size = message.width * channels

        # step은 ROS 이미지 한 줄의 전체 바이트 수이다.
        # 카메라에 따라 실제 픽셀 뒤에 빈 여백이 붙을 수도 있다.
        if message.step < pixel_row_size:
            return None

        # ROS Image의 연속된 바이트를 NumPy 숫자 배열로 해석한다.
        # uint8은 색상 하나를 0~255 숫자로 저장하는 자료형이다.
        image_bytes = np.frombuffer(message.data, dtype=np.uint8)

        # 전체 이미지에 있어야 하는 최소 바이트 수
        required_size = message.height * message.step

        # 메시지에 들어온 바이트가 부족하면 깨진 이미지로 판단한다.
        if image_bytes.size < required_size:
            return None

        # 한 줄로 이어진 바이트를 [높이, 한 줄 바이트 수] 형태로 나눈다.
        image_rows = image_bytes[:required_size].reshape(
            message.height,
            message.step,
        )

        # 줄 끝 여백을 제외하고 [높이, 너비, 채널] 이미지 모양으로 만든다.
        pixels = image_rows[:, :pixel_row_size].reshape(
            message.height,
            message.width,
            channels,
        )

        # OpenCV 계열 카메라는 BGR 순서를 사용할 수 있다.
        # PIL과 Gemma는 RGB를 사용하므로 색상 채널 순서를 뒤집는다.
        if message.encoding == "bgr8":
            pixels = pixels[:, :, ::-1]

        # Alpha는 투명도 정보이므로 VLM 입력에서는 제거한다.
        elif message.encoding == "rgba8":
            pixels = pixels[:, :, :3]

        # BGRA는 BGR을 RGB로 바꾸면서 Alpha도 제거한다.
        elif message.encoding == "bgra8":
            pixels = pixels[:, :, [2, 1, 0]]

        # NumPy 배열을 Gemma processor가 읽을 수 있는 PIL RGB 이미지로 변환한다.
        return PILImage.fromarray(pixels.copy(), mode="RGB")

    def _describe_scene(self, user_text: str) -> str:
        if not ENABLE_VLM or self.call_vlm is None:
            return "현재는 카메라 확인 기능을 사용할 수 없어요."

        with self.camera_lock:
            camera_message = self.latest_camera_message

        camera_image = self._image_message_to_pil(camera_message)

        if camera_image is None:
            return "아직 카메라 화면을 확인할 수 없어요."

        try:
            reply = self.call_vlm(
                [camera_image],
                SCENE_SYSTEM,
                user_text,
            )
        except Exception as exc:
            self.get_logger().warning(f"장면 묘사 실패 : {exc}")
            return "카메라 화면을 확인하는 중 문제가 생겼어요."

        if not isinstance(reply, str) or not reply.strip():
            return "카메라 화면에서 확실하게 확인되는 대상이 없어요."

        reply = reply.strip()

        # thinking을 꺼도 문자열 앞에 thought가 남는 경우 제거
        if reply.startswith("thought"):
            reply = reply[len("thought"):].lstrip()

        return reply


    def _prepare_order(self):
        # ui start 오면 상태 초기화, main의 int 3을 기다린다

        if self.ui_started or self.conversation_started or self.active_task is not None:
            self.get_logger().warning("이미 주문 세션이 진행 중이라 중복 /ui/start를 무시함")
            return

        pending_initial_next = self.pending_initial_next
        self._clear_state()

        # UI에서 새 주문을 시작할 때마다 로그 세션과 턴 번호를 새로 만든다.
        self.runtime_session_id = datetime.now().astimezone().strftime(
            "%Y%m%dT%H%M%S_%f%z"
        )
        self.runtime_turn_index = 0

        self._set_stt_enabled(False)
        self.ui_started = True

        if pending_initial_next:
            self.get_logger().info("먼저 도착한 /llm/next로 주문 대화를 시작함")
            self._start_order()



    def _start_order(self):
         # UI 시작 뒤 main의 첫 /llm/next=3을 받으면 실제 주문 대화 시작

        self.ui_started = False
        self.conversation_started = True
        self._set_stt_enabled(True)

        greeting_ment= "안녕하세요! 저는 스파게티 밀키트 주문을 도와드리는 로봇이에요. "

        self.reply_pub.publish(String(data=greeting_ment + "먼저 면 종류, 소스, 양을 골라주세요."))
        self.get_logger().info("새 주문 시작")

    def _reset(self, reason: str): # 이유(문자열)
        self._clear_state() # 주문·대화·행동 기록 초기화
        self._set_stt_enabled(False)
        self.reply_pub.publish(
            String(data="주문을 초기화했어요. 처음부터 다시 시작할게요.")
        )
        self.get_logger().info(f"리셋 : {reason}")


    def _clear_state(self):
        # 홈버튼 누르면 싹다 초기화
        self.order = new_order()
        self.section = "noodle"
        self.history = []
        self.task_queue = []
        self.active_task = None
        self.waiting_pick = False
        self.completed_tasks = []
        self.robot_started = False
        self.verification_history = [] # VLM 최초 실패와 한 번 재시도 결과
        self.refusal_prompted_section = None
        self.recommendation_state = {}
        self.last_recommendation = {}
        self.remaining_item_prompted_section = None
        self.skipped_sections = set()
        self.vlm_confirmed = False
        self.awaiting_confirm = None
        self.order_finished = False
        self.ui_started = False
        self.pending_initial_next = False
        self.action_history = []
        self.conversation_started = False
        self.latest_camera_message = None
        self.vlm_camera_messages = deque(maxlen=NUMBER_IMAGE_FOR_CONFIRM)


    def _section_items(self):
        # 현재 섹션에서 선택할 수 있는 재료 반환
        # 식이 제약은 order_v2.enforce_constraints()에서 별도로 검사한다.

        if self.section == "veggie":
            return VEGGIES.copy()

        elif self.section == "meat":
            return MEATS.copy()

        elif self.section == "extra":
            return EXTRAS.copy()

        return []



    def _section_select_prompt(self):
        # 섹션 넘어갈 때 뭐 고를지 알려주는 문장 하드코딩

        if self.section == "noodle":
            return "면 종류, 소스, 양을 골라주세요."

        elif self.section == "sauce":
            return "이제 고르신 소스를 담아드릴게요."

        elif self.section == "lid":
            return "재료를 모두 담았습니다. 마지막 주문을 확인할게요."

        section_label = SECTION_LABELS[self.section]
        blocked_items = {item for constraint in self.order["constraints"] for item in CONSTRAINT_BLOCKS.get(constraint, {}).get("toppings", [])}
        section_items = [item for item in self._section_items() if item not in blocked_items]
        already = []
        remaining = []

        for item in section_items:
            if item in self.order["toppings"]:
                already.append(item)
            else:
                remaining.append(item)

        if already and remaining:
            self.remaining_item_prompted_section = self.section
            return f"다음은 {section_label}입니다. {', '.join(already)}는 골랐어요. {', '.join(remaining)}도 추가할 수 있어요. 더 없으면 다 골랐다고 말씀해 주세요."

        elif already:
            return f"다음은 {section_label}입니다. {', '.join(already)}를 골랐어요. 더 없으면 다 골랐다고 말씀해 주세요."

        elif section_items:
            return f"다음은 {section_label}입니다. {', '.join(section_items)} 중에서 골라주세요."

        return f"다음은 {section_label}입니다. 현재 제약으로 선택할 수 있는 재료가 없어요."

    def _missing_prompt(self, missing):

        # 야채와 육류는 현재 로봇 시나리오상 최소 한 재료를 선택해야 한다.

        if "야채 재료" in missing or "육류 재료" in missing:
            section_items = self._section_items()
            blocked_items = {item for constraint in self.order["constraints"] for item in CONSTRAINT_BLOCKS.get(constraint, {}).get("toppings", [])}
            available_items = [item for item in section_items if item not in blocked_items]

            if len(available_items) >= 2:
                return f"{available_items[0]}와 {available_items[1]} 중 하나는 선택하셔야 해요."

            elif len(available_items) == 1:
                return f"현재 제약에서는 {available_items[0]}를 선택할 수 있어요."

            return f"현재 제약으로 선택할 수 있는 {SECTION_LABELS[self.section]} 재료가 없어요. 모두 빼달라고 말씀해 주세요."

        field_labels = {
            "sauce": "소스",
            "noodle_type": "면 종류",
            "noodle_portion": "면 양",
        }

        needed = []

        for field in missing:
            needed.append(field_labels.get(field, field))

        return f"아직 {', '.join(needed)} 선택이 필요해요."


    def _publish_next_task(self):
        # task queue에서 하나 꺼내서 비전쪽으로 publish

        if not self.task_queue:
            return

        task = self.task_queue.pop(0) # task_queue에서 첫 작업을 꺼내면서 제거


        self.active_task = task # task를 현재 상태로 설정
        self.vlm_confirmed = False

        with self.camera_lock:
            self.vlm_camera_messages.clear()

        plan_for_publish = {
            "class": TOPIC_CLASS_NAMES[task["class"]], # remapping해서 발행
            "repeat_count": task["repeat_count"],
        }

        plan_next = json.dumps(plan_for_publish, ensure_ascii=False)

        self.robot_started = True # 코드 내부 상태 변수


        self.plan_pub.publish(String(data=plan_next)) # 비전한테 현재 할 것을 publish

        self.get_logger().info(
            f"plan 발행 : {plan_for_publish}, "
            f"내부 작업 : {task}, "
            f"남은 작업 : {len(self.task_queue)}"
        )

    def _start_confirmed_section(self, prefix=""): # 기존 확정 문장 공유 함수
        section_tasks = build_section_plan(self.order, self.section)
        task_names = [task["class"] for task in section_tasks]
        section_label = SECTION_LABELS[self.section]

        if len(task_names) == 1:
            reply = (
                f"{section_label} 선택이 완료됐어요. "
                f"이제 {task_names[0]} 담기를 시작할게요."
            )

        elif len(task_names) >= 2:
            task_text = (
                f"{', '.join(task_names[:-1])}와 "
                f"{task_names[-1]}"
            )

            reply = (
                f"{section_label} 선택이 완료됐어요. "
                f"이제 {task_text} 담기를 시작할게요."
            )

        else:
            reply = f"{section_label} 선택이 완료됐어요."

        if prefix:
            reply = f"{prefix} {reply}"

        return self._confirm_current_section(reply)



    def _confirm_current_section(self, reply):
        # 현재 섹션 주문을 로봇 작업으로 변환

        tasks = build_section_plan(self.order, self.section)

        if tasks:
            if WAIT_FOR_ROBOT:
                self._set_stt_enabled(False)

            self.task_queue.extend(tasks) # 다음 테스크 task 추가!
            self._publish_next_task()

            if WAIT_FOR_ROBOT:
                self.waiting_pick = True
                return reply

    # 담을 작업이 없거나 로봇을 기다리지 않는 모드면 바로 다음 섹션 

        return self._commit_section(reply)


    def _commit_section(self, reply, skipped=False):

        # 현재 섹션 끝내고 다음 섹션으로 이동. 마무리 되면 실행할 함수인듯

        done_label = SECTION_LABELS[self.section] 
        next_step = next_section(self.section)

        self.waiting_pick = False

        if next_step is None:
            self._finish_order()
            return f"{reply} 주문이 모두 완료되었습니다. 이용해 주셔서 감사합니다." # 마무리!

        self.refusal_prompted_section = None
        self.remaining_item_prompted_section = None
        self.section = next_step


        self.get_logger().info(f"다음 섹션 : {self.section}")

        # 소스는 면 단계에서 이미 골랐으므로 사용자한테 다시 안 물어보고 바로 담음

        if self.section == "sauce":
            sauce_tasks = build_section_plan(self.order, self.section)


            if not sauce_tasks:
                return f"{reply} 선택된 소스가 없어서 작업을 진행할 수 없어요."

            if WAIT_FOR_ROBOT:
                self._set_stt_enabled(False)

            self.task_queue.extend(sauce_tasks)


            self._publish_next_task()


            if WAIT_FOR_ROBOT:
                self.waiting_pick = True
                return f"{reply} {self._section_select_prompt()}"


            return self._commit_section(
                f"{reply} 소스 작업을 시작했어요."
            )

        # 소스까지 다 담으면 마지막 포장 확인

        elif self.section == "lid":
            self.recommendation_state = {}
            if CONFIRM_LID:
                self.awaiting_confirm = "final_order"
                return f"{reply} {self._order_summary()} 이대로 포장할까요?"


            return self._confirm_current_section(f"{reply} 포장을 시작할게요.")

        # 자동 추천은 사용자가 요청한 마지막 섹션까지만 재확인 없이 진행한다.
        if self.recommendation_state.get("phase") == "running":
            if self.section in self.recommendation_state.get("target_sections", []):
                return self._start_confirmed_section(reply)

            self.recommendation_state = {}

        if skipped:
            return f"{reply} {self._section_select_prompt()}"

        return f"{reply} {done_label} 선택이 끝났어요. {self._section_select_prompt()}"

    def _process_vlm_confirm(self, should_confirm: bool):
        # /llm/confirm_start=True는 VLM 판정만 시작(응답은 뭐... 일단 하드코딩?)
        # 작업 완료와 다음 단계 이동은 제어 home 뒤 /llm/next=3에서 처리

        self.get_logger().info(f"/llm/confirm_start 수신 : {should_confirm}")

        if not should_confirm:
            self.get_logger().warning("confirm_start가 False라서 VLM 판정을 무시함")
            return

        if self.active_task is None:
            self.get_logger().warning("판정할 active_task가 없어서 confirm_start를 무시함")
            return

        if self.section == "lid":
            self.get_logger().warning("cover 작업은 VLM으로 판정하지 않음")
            return

        # VLM OFF 디버깅 모드는 판정을 생략하고 성공으로 통과
        if not ENABLE_VLM:
            self.vlm_result_pub.publish(String(data="success"))
            self.vlm_confirmed = True
            self.get_logger().info("VLM OFF 우회 결과 발행 : success")
            return

        if self.call_vlm is None:
            self.get_logger().warning("VLM 함수가 준비되지 않아 판정할 수 없음")
            return

        self.vlm_confirmed = False

        # 섹션별 판정 모드는 지금 실행 중인 재료 하나만 확인
        if SECTION_SEQUENCE_VLM:
            vlm_tasks = [self.active_task.copy()]

        # 마지막 판정 모드는 소스 단계에서 담은 식재료 전체를 확인
        elif self.section == "sauce":
            vlm_tasks = []

            for section_name in SECTION_ORDER:
                if section_name == "lid":
                    continue

                section_tasks = build_section_plan(self.order, section_name)
                vlm_tasks.extend(section_tasks)

        else:
            self.get_logger().warning("현재 단계는 마지막 VLM 판정 시점이 아님")
            return

        # failed_tasks, vlm_results = self._judge_latest_tasks(vlm_tasks)

        # if failed_tasks:
        #     failed_names = []

        #     for task in failed_tasks:
        #         failed_names.append(task["class"])

        #         fail_for_publish = {
        #             "result": "fail",
        #             "class": TOPIC_CLASS_NAMES[task["class"]],
        #         }

        #         self.vlm_result_pub.publish(
        #             String(data=json.dumps(fail_for_publish, ensure_ascii=False))
        #         )

        #         self.get_logger().info(f"VLM 결과 발행 : {fail_for_publish}")

        #     reply = (
        #         f"{', '.join(failed_names)} 위치를 확인하지 못했어요. "
        #         "해당 작업을 다시 확인할게요."
        #     )

        #     self.reply_pub.publish(String(data=reply))
        #     return

        # # 성공할 때는 class 없이 success 문자열만 보낸다.
        # self.vlm_result_pub.publish(String(data="success"))
        # self.get_logger().info("VLM 결과 발행 : success")

        # self.vlm_confirmed = True

        # checked_names = []

        # for result in vlm_results:
        #     checked_names.append(result["class"])

        # reply = f"{', '.join(checked_names)} 위치를 확인했어요. 로봇 작업을 마무리할게요."
        # self.reply_pub.publish(String(data=reply))

        failed_tasks, vlm_results = self._judge_latest_tasks(vlm_tasks)

        failed_names = []
        policy_success_names = []
        failed_classes = [task["class"] for task in failed_tasks]

        for result in vlm_results:
            class_name = result["class"]

            previous_failures = 0

            for verification in self.verification_history:
                if (verification["class"] == class_name and verification["result"] == "fail"):
                    previous_failures += 1

            attempt = previous_failures + 1

            if class_name not in failed_classes:
                self.verification_history.append({"class": class_name, "attempt": attempt, "result" : "pass"})
                continue

            # 같은 재료의 첫 실패만 main과 제어에 재시도 요청

            if previous_failures == 0:
                self.verification_history.append({"class": class_name, "attempt": attempt, "result" : "fail"})

                failed_names.append(class_name)

                fail_for_publish = {"result": "fail", "class": TOPIC_CLASS_NAMES[class_name]}

                self.vlm_result_pub.publish(String(data=json.dumps(fail_for_publish, ensure_ascii=False)))

                self.get_logger().info(f"VLM 결과 발행 : {fail_for_publish}")

            # 재시도 까지 끝나면 더 막지 않음.
            else:
                self.verification_history.append({"class": class_name, "attempt": attempt, "result" : "policy_success"})
                policy_success_names.append(class_name)

        if failed_names:
            # 재시도 판정에는 첫 시도 이후 들어온 새 카메라 프레임만 사용한다.
            with self.camera_lock:
                self.latest_camera_message = None
                self.vlm_camera_messages.clear()

            reply = (f"{', '.join(failed_names)} 위치를 확인하지 못했어요. " "해당 작업을 한 번 다시 시도할게요.")
            self.reply_pub.publish(String(data=reply))
            return


        # 최초 성공 또는 한 번 재시도 완료 후에는 공정을 통과
        self.vlm_result_pub.publish(String(data="success"))
        self.get_logger().info("VLM 결과 발행 : success")

        self.vlm_confirmed = True

        checked_names = []

        for result in vlm_results:
            checked_names.append(result["class"])

        if policy_success_names:
            reply = (f"{', '.join(policy_success_names)} 재시도를 마쳤어요. " "다음 단계로 진행할게요." )

        else:
            reply = (f"{', '.join(checked_names)} 위치를 확인했어요. " "로봇 작업을 마무리할게요.")

        self.reply_pub.publish(String(data=reply))


        """
        fail → active_task 유지, /llm/plan 재발행 안 함
        pass → vlm_confirmed=True, active_task 유지
        """


    def _process_next(self, result_code: int):
        # 첫 /llm/next=3은 주문 대화를 시작한다.
        # 이후 /llm/next=3은 VLM PASS와 제어 home 복귀가 끝난 작업을 완료한다.

        self.get_logger().info(f"/llm/next 수신 : {result_code}")

        # 지수 / vlm한테 오는 trigger 토픽 (/llm/next) 검증(상단에 BY VLM NUM으로 유동적으로 관리 ㄱㄱㄱㄱ)

        if result_code != BY_VLM_NUM:
            self.get_logger().warning("우리 토픽 규약이랑 맞지 않아용")
            return

        # ui 시작 요청 뒤 처음 받은 3은 로봇 작업 완료가 아닌, 주문 시작
                # /ui/start보다 먼저 도착한 최초 next는 버리지 않고 잠시 보관한다.
        if (not self.ui_started and not self.conversation_started and self.active_task is None and not self.order_finished):
            self.pending_initial_next = True
            self.get_logger().info("/ui/start보다 먼저 도착한 /llm/next를 보관함")
            return

        if self.ui_started:
            self._start_order()
            return
        
        # 중복으로 호출되어도 다음 작업을 잘못하지 않게 사전에 막아버령

        if self.active_task is None:
            self.get_logger().warning("진행 중인 작업이 없어서 /llm/next를 무시함")
            return

        # 현재 main은 소스 작업이 끝나면 cover를 직접 실행하고 /llm/reset을 보낸다.
        # 소스 완료 /llm/next가 잘못 들어와도 LLM의 이전 cover 계획 경로로 넘어가지 않는다.
        if self.section == "sauce":
            self.get_logger().warning("소스와 cover 작업 완료 후 main의 /llm/reset을 기다리는 중")
            return

        # cover 완료는 /llm/next가 아니라 main의 /llm/reset으로 확정한다.
        if self.section == "lid":
            self.get_logger().warning("cover 작업 완료를 위해 /llm/reset을 기다리는 중")
            return

        needs_vlm_confirm = ENABLE_VLM and (
            SECTION_SEQUENCE_VLM or self.section == "sauce"
        )

        if needs_vlm_confirm and not self.vlm_confirmed:
            self.get_logger().warning("VLM PASS 전이라 /llm/next를 무시함")
            return

        completed_task = self.active_task
        self.active_task = None
        self.vlm_confirmed = False

        if completed_task not in self.completed_tasks:
            self.completed_tasks.append(completed_task.copy())

        self.get_logger().info(f"현재 하고 있는 작업 완료 : {completed_task}")



        if self.task_queue:
            self._publish_next_task()
            return


        # task_queue가 비었고 waiting_pick이면 현재 섹션 작업이 전부 끝난 것

        if not self.waiting_pick:
            return

        self.get_logger().info(f"총 작업 완료 : {self.completed_tasks}")

        # 현재 섹션에서 실제 완료된 작업만 처리

        completed_section_tasks = []

        for section_task in build_section_plan(self.order, self.section):
            if section_task in self.completed_tasks:
                completed_section_tasks.append(section_task)

        completed_names = [] # 딕셔너리 넣어둠

        for task in completed_section_tasks:
            completed_names.append(task["class"])

        if len(completed_names) == 1:
            reply = f"{completed_names[0]} 담기가 끝났어요."

        elif len(completed_names) >= 2:
            completed_text = f"{', '.join(completed_names[:-1])}와 {completed_names[-1]}"
            reply = f"{completed_text} 담기가 끝났어요."

        else:
            reply = "현재 단계의 재료 담기가 끝났어요."

        reply = self._commit_section(reply)


        # 다음 선택 단계나 최종 포장 확인을 물었을 때만 다시 듣는다.
        if not self.waiting_pick and not self.order_finished:
            self._set_stt_enabled(True)

        

        self.reply_pub.publish(String(data=reply))
        

    def _finish_order(self):
        # 전체 주문 완료는 한 번만 발행

        if self.order_finished:
            return

        self.active_task = None
        self.waiting_pick = False
        self.awaiting_confirm = None
        self.conversation_started = False
        self._set_stt_enabled(False)
        self.recommendation_state = {}
        self.order_finished = True

        self.done_pub.publish(Bool(data=True))
        self.get_logger().info("총 주문 완료")

    def _order_summary(self, order=None):
        # 뚜껑 닫기 전 최종 주문 문장

        if order is None:
            order = self.order

        amount_word = {
            "low": "적게",
            "normal": "보통",
            "high": "많이",
        }

        parts = []

        if order["sauce"]:
            parts.append(f"{order['sauce']} 소스")

        if order["noodle_type"]:
            portion = amount_word.get(
                order["noodle_portion"],
                "",
            )

            parts.append(
                f"{order['noodle_type']} {portion}".strip()
            )

        for topping, amount in order["toppings"].items():
            amount_text = amount_word.get(amount, amount)
            parts.append(f"{topping} {amount_text}")

        if not parts:
            return "현재 주문"

        return "주문은 " + ", ".join(parts) + "입니다."

    def _continue_prompt(self): # WAIT_USER_CONFIRM에서 추가 주문을 받을 문장 추가
        missing = missing_required(self.order, self.section)

        if missing:
            return self._missing_prompt(missing)

        blocked_items = {item for constraint in self.order["constraints"] for item in CONSTRAINT_BLOCKS.get(constraint, {}).get("toppings", [])}

        if self.section == "noodle":
            section_items = EARLY_TOPPINGS.get("noodle", [])
        else:
            section_items = self._section_items()

        remaining = [item for item in section_items if item not in self.order["toppings"] and item not in blocked_items]

        if remaining and self.remaining_item_prompted_section != self.section:
            self.remaining_item_prompted_section = self.section
            return f"{', '.join(remaining)}도 추가할 수 있어요. 더 없으면 다 골랐다고 말씀해 주세요."

        return "더 없으면 다 골랐다고 말씀해 주세요."

    def _is_affirmative(self, user_text):
        # 문장 끝에 기호가 붙어도 긍정 처리하게 텍스트 전처리

        normalized = user_text.strip().rstrip(".!?,~ ")

        return normalized in AFFIRM_WORDS

    def _recommendation_reply(self, result, recommended_order, scope, target_sections):
        facts = copy.deepcopy(result["facts"])
        facts["order"] = copy.deepcopy(self.order)
        facts["recommended_order"] = copy.deepcopy(recommended_order)
        facts["recommendation_scope"] = scope
        facts["target_sections"] = target_sections.copy()
        facts["flow_state"]["recommendation_state"] = copy.deepcopy(self.recommendation_state)
        facts["flow_state"]["last_recommendation"] = copy.deepcopy(self.last_recommendation)

        try:
            reply = self.generate_reply(facts)
        except Exception:
            reply = ""

        if isinstance(reply, str) and reply.strip():
            return reply.strip()

        return "추천안을 준비했어요. 이대로 진행할까요?"


    def _record_turn(self, user_text, action, reply, result=None):
        has_model_result = result is not None
        result = result or {}
        decision_source = "model_tool" if has_model_result else "python_guard"

        model_input = None
        user_msg = result.get("user_msg", {})

        if isinstance(user_msg, dict) and isinstance(user_msg.get("content"), str):
            try:
                model_input = json.loads(user_msg["content"])

            except json.JSONDecodeError:
                model_input = {"raw_content": user_msg["content"]}

        if ENABLE_RUNTIME_LOG and isinstance(self, LLMNode):
            try:
                self.runtime_turn_index += 1

                log_row = {
                    "schema_version": 3,
                    "record_type": "tool_turn",
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "session_id": self.runtime_session_id,
                    "turn_index": self.runtime_turn_index,
                    "record_id": f"{self.runtime_session_id}_{self.runtime_turn_index:04d}",
                    "prompt_version": "tool_v3_2026-09-04",
                    "adapter_version": Path(TOOL_ADAPTER_PATH).name if ENABLE_TOOL_LORA else "base",
                    "parser_status": result.get("parser_status", "python_guard" if not has_model_result else "unknown"),

                    # 이번 발화 전 모델에 함께 전달된 실제 자연어 대화
                    "history": copy.deepcopy(self.history[-(HISTORY_TURNS * 2):]),

                    # order, section, missing, action_history, 사용자 발화
                    "model_input": model_input,

                    # 모델이 처음 출력한 Tool. Python 검증 전 값이다.
                    "tool_call": copy.deepcopy(result.get("tool_call")),

                    # 실제 주문에 적용된 결과
                    "python_result": {
                        "decision_source": decision_source,
                        "action": action,
                        "clean": copy.deepcopy(result.get("clean", {})),
                        "changed": copy.deepcopy(result.get("changed", [])),
                        "blocked": copy.deepcopy(result.get("blocked", [])),
                        "dropped": copy.deepcopy(result.get("dropped", [])),
                        "missing": copy.deepcopy(result.get("missing", [])),
                        "order_after": copy.deepcopy(self.order),
                        "section_after": self.section,
                        "flow_state_after": {
                            "refusal_prompted_section": self.refusal_prompted_section,
                            "recommendation_state": copy.deepcopy(self.recommendation_state),
                            "last_recommendation": copy.deepcopy(self.last_recommendation),
                            "remaining_item_prompted_section": self.remaining_item_prompted_section,
                            "skipped_sections": sorted(self.skipped_sections),
                        },
                        "recommendation_validation": copy.deepcopy(result.get("recommendation_validation")),
                    },

                    # Reply 모델 원본과 Python 분기까지 끝난 실제 송출 문장
                    "model_reply": result.get("reply") if has_model_result else None,
                    "final_reply": reply,

                    # 나중에 사람이 검수한 뒤 학습 데이터로 바꿀 때 채운다.
                    "review": {
                        "accepted": None,
                        "correct_action": None,
                        "correct_changes": None,
                        "note": "",
                    },
                }

                with RUNTIME_LOG_PATH.open("a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(log_row, ensure_ascii=False) + "\n")

            except Exception as exc:
                # 로그 실패 때문에 주문 서비스까지 멈추지는 않는다.
                self.get_logger().warning(f"런타임 로그 저장 실패 : {exc}")

        # 모델이 고른 행동보다 Python 검증이 끝난 실제 결과를 기억한다.
        self.action_history.append({
            "user_text": user_text,
            "decision_source": decision_source,
            "action": action,
            "changed": copy.deepcopy(result.get("changed", [])),
            "blocked": copy.deepcopy(result.get("blocked", [])),
            "dropped": copy.deepcopy(result.get("dropped", [])),
            "missing": copy.deepcopy(result.get("missing", [])),
        })

        # 현재 주문에서 필요한 최근 행동만 유지
        self.action_history = self.action_history[-HISTORY_TURNS:]

        # 그리고 사용자와 tts에 전달한 문장을 history에 저장
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply}) # 모델의 대답

    def _process(self, user_text: str):
        if not self.conversation_started:
            self.get_logger().warning("주문 대화 시작 전 STT 결과를 무시함")
            return

        if self.order_finished:
            reply = "주문이 완료됐어요. 새 주문은 처음부터 시작해주세요."
            self._record_turn(user_text, "order_finished_guard", reply)
            self.reply_pub.publish(String(data=reply))
            return

        # 여기부터 추가
        # cancel_order는 모델이 골라도 바로 초기화하지 않고 한 번 더 확인하도록

        if self.awaiting_confirm == "cancel_order":

            if self._is_affirmative(user_text):

                # 취소 질문 뒤 로봇 작업이 시작됐을 가능성도 다시 검사
                if self.robot_started and not ALLOW_CANCEL_AFTER_ROBOT_START:
                    self.awaiting_confirm = None
                    guard_action = "cancel_order_blocked"
                    reply = "이미 재료를 담기 시작해서 자동으로 주문을 취소할 수 없어요."

                else:
                    reply = "주문을 초기화했어요. 처음부터 다시 시작할게요."
                    self._record_turn(user_text, "cancel_order_accept", reply)
                    self._reset("손님 취소 확정") # reason 들어감(문자열)
                    return

            else:
                self.awaiting_confirm = None
                guard_action = "cancel_order_reject"
                reply = "주문을 취소하지 않고 계속 진행할게요."

            self._record_turn(user_text, guard_action, reply)
            self.reply_pub.publish(String(data=reply))
            return

        # 뚜껑 닫기 전 마지막 확인은 모델보다 먼저 처리 -> 왜?
        # 마지막 확인은 Python이 처리해야 모델 오판으로 포장이 확정되지 않음

        if self.awaiting_confirm == "final_order":

            if self._is_affirmative(user_text):
                self.awaiting_confirm = None
                guard_action = "final_order_accept"
                reply = self._confirm_current_section("네, 이대로 포장을 시작할게요.")

            else:
                guard_action = "final_order_reconfirm"
                reply = "주문은 이미 담았어요. 이대로 포장할까요?"

            self._record_turn(user_text, guard_action, reply)
            self.reply_pub.publish(String(data=reply))
            return

        # 로봇이 현재 섹션을 담고 있을 때는 주문을 새로 받지 않는다.

        if self.waiting_pick:
            reply = "지금 로봇이 담고 있어요. 잠시만 기다려 주세요."
            self._record_turn(user_text, "waiting_pick_guard", reply)
            self.reply_pub.publish(String(data=reply))
            return

        # 소스·면 종류·면 양은 각각 하나만 선택할 수 있다.
        # 반반처럼 명확한 동시 선택만 막고, 자기 정정은 Tool 모델이 판단한다.
        if self.section == "noodle":
            normalized_text = "".join(user_text.split())
            question_only = any(word in normalized_text for word in ("차이", "뭐가", "어느", "어떤", "추천", "골라줘", "정해줘", "선택해줘", "어때", "어떻게", "가능", "수있", "알려"))
            explicit_conflict = any(word in normalized_text for word in ("반반", "둘다", "둘모두", "같이넣", "섞", "전부", "다넣"))

            single_choice_groups = [
                ("소스", {"오일": ("오일",), "토마토": ("토마토",), "크림": ("크림",)}),
                ("면 종류", {"얇은면": ("얇은면", "얇은거", "가는면", "가는거"), "넓적면": ("넓적면", "넓은면", "넓적한면", "두꺼운면", "두꺼운거", "굵은면", "굵은거", "납작한면", "납작한거")}),
                ("면 양", {"low": ("적게", "조금"), "normal": ("보통", "1인분", "일인분"), "high": ("많이", "넉넉", "푸짐")}),
            ]

            conflict_fields = []

            if explicit_conflict and not question_only:
                for field_label, values in single_choice_groups:
                    selected_values = set()

                    for value, aliases in values.items():
                        if any(alias in normalized_text for alias in aliases):
                            selected_values.add(value)

                    if len(selected_values) >= 2:
                        conflict_fields.append(field_label)

            if conflict_fields:
                if len(conflict_fields) == 1:
                    reply = f"{conflict_fields[0]}는 한 가지만 선택할 수 있어요. 원하시는 하나를 다시 말씀해 주세요."
                else:
                    reply = f"{', '.join(conflict_fields)}는 각각 하나만 선택할 수 있어요. 원하시는 값을 다시 말씀해 주세요."

                self._record_turn(user_text, "single_choice_conflict", reply)
                self.reply_pub.publish(String(data=reply))
                return

        recommendation_edit_pending = False

        # 확정 전 추천안은 모델을 다시 호출하지 않고 Python이 승인·거절을 처리한다.
        if self.recommendation_state.get("phase") == "confirming":
            normalized = user_text.strip().rstrip(".!?,~ ")
            guard_action = None

            if self.recommendation_state.get("started_section") != self.section:
                self.recommendation_state = {}
                guard_action = "recommendation_expired"
                reply = "추천 상태가 현재 단계와 맞지 않아 취소했어요. 다시 추천해 달라고 말씀해 주세요."

            elif self._is_affirmative(user_text):
                recommendation_scope = self.recommendation_state.get("scope")
                recommendation_excluded = self.recommendation_state.get("excluded", []).copy()
                self.order = copy.deepcopy(self.recommendation_state["proposal"])

                if recommendation_scope in ("selected", "all"):
                    self.recommendation_state = {
                        "phase": "running",
                        "scope": recommendation_scope,
                        "target_sections": self.recommendation_state.get("target_sections", []),
                        "started_section": self.section,
                        "excluded": recommendation_excluded,
                    }

                else:
                    self.recommendation_state = {}

                guard_action = "recommendation_accept"
                reply = self._start_confirmed_section("추천한 내용으로 반영했어요.")

            elif normalized in ("아니", "아니요", "싫어", "됐어", "직접 고를게"):
                self.recommendation_state = {}
                guard_action = "recommendation_reject"
                reply = "추천안은 반영하지 않았어요. 원하는 재료를 직접 말씀해 주세요."

            else:
                recommendation_edit_pending = True

            if guard_action is not None:
                self._record_turn(user_text, guard_action, reply)
                self.reply_pub.publish(String(data=reply))
                return

        order_before_tool = copy.deepcopy(self.order) if recommendation_edit_pending else None

        result = self.graph.invoke({
            "user_text": user_text,
            "order": self.order,
            "section": self.section,
            "history": self.history,
            "action_history": self.action_history,
            "completed_tasks": self.completed_tasks,
            "refusal_prompted_section": self.refusal_prompted_section,
            "recommendation_state": self.recommendation_state,
            "last_recommendation": self.last_recommendation,
            "remaining_item_prompted_section": self.remaining_item_prompted_section,
            "skipped_sections": sorted(self.skipped_sections),
        }) # 그래프 내부 state
        action = result["action"]
        reply = result["reply"]
        missing = result["missing"]

        # 추천 범위를 기다리던 중 실패하거나 일반 주문으로 전환되면 이전 추천 상태를 끝낸다.
        if self.recommendation_state.get("phase") == "await_scope" and action in ("error", "set_order", "set_order_and_confirm"):
            self.recommendation_state = {}


        """
        아무거나 추천해줘
        → 추천 범위 대기

        전체 추천 요청 해석 실패
        → 추천 상태 삭제

        토마토 얇은면 보통으로 해줘
        → 이전 버섯·게살 요청 없이 일반 주문 처리
                
        """

        # 추천 확인 중 confirm_section은 임시 추천안을 실제 주문에 확정한다.
        if recommendation_edit_pending and action == "confirm_section":
            recommendation_scope = self.recommendation_state.get("scope")
            recommendation_excluded = self.recommendation_state.get("excluded", []).copy()
            self.order = copy.deepcopy(self.recommendation_state["proposal"])

            if recommendation_scope in ("selected", "all"):
                self.recommendation_state = {
                    "phase": "running",
                    "scope": recommendation_scope,
                    "target_sections": self.recommendation_state.get("target_sections", []),
                    "started_section": self.section,
                    "excluded": recommendation_excluded,
                }

            else:
                self.recommendation_state = {}

            reply = self._start_confirmed_section("추천한 내용으로 반영했어요.")
            self._record_turn(user_text, action, reply, result)
            self.reply_pub.publish(String(data=reply))
            return

        # 잘못된 Tool이 확정 전 실제 주문을 바꾸지 못하게 원래 주문으로 복구한다.
        if recommendation_edit_pending and action in ("set_order", "set_order_and_confirm", "refuse_section"):
            self.order = order_before_tool
            result["changed"] = []
            result["dropped"] = result.get("dropped", []) + ["recommendation_edit"]
            result["missing"] = missing_required(self.order, self.section)
            reply = "추천안 수정 요청을 정확히 처리하지 못했어요. 바꿀 재료를 다시 말씀해 주세요."
            self._record_turn(user_text, "recommendation_edit_blocked", reply, result)
            self.reply_pub.publish(String(data=reply))
            return

        self.order = result["order"]

        self.get_logger().info(f"[action] name={action} " f"clean={result.get('clean')} "  f"dropped={result.get('dropped')} "  f"blocked={result.get('blocked')}")

        if action == "describe_scene":
            reply = self._describe_scene(user_text)

        elif action == "cancel_order":

            if self.robot_started and not ALLOW_CANCEL_AFTER_ROBOT_START:
                reply = "이미 재료를 담기 시작해서 자동으로 주문을 취소할 수 없어요."

            else:
                self.awaiting_confirm = "cancel_order"
                reply = "주문을 전부 취소하고 처음부터 다시 시작할까요?"

        elif action == "refuse_section":
            refusal = result.get("clean", {})
            reason = refusal.get("reason")
            refused_items = refusal.get("items", [])
            section_items = self._section_items()

            # 현재 단계의 모든 재료가 정확히 들어온 경우에만 스킵할 수 있다.
            if not section_items or set(refused_items) != set(section_items):
                reply = "현재 단계의 모든 재료를 제외할지 다시 말씀해 주세요."

            elif (
                reason == "dislike"
                and self.section in ("veggie", "meat")
                and self.refusal_prompted_section != self.section
            ):
                self.refusal_prompted_section = self.section
                item_text = f"{section_items[0]}와 {section_items[1]}"
                reply = f"{item_text} 중 하나는 선택하셔야 해요. 그래도 모두 빼드릴까요?"

            else:
                skipped_section = self.section
                removed_items = []

                for item in section_items:
                    if item in self.order["toppings"]:
                        del self.order["toppings"][item]
                        removed_items.append(f"toppings.{item}")

                self.skipped_sections.add(skipped_section)
                self.refusal_prompted_section = None
                self.remaining_item_prompted_section = None

                result["changed"] = removed_items
                result["missing"] = []
                missing = []

                section_label = SECTION_LABELS[skipped_section]
                reply = self._commit_section(
                    f"{section_label}는 빼드릴게요.",
                    skipped=True,
                )

        elif action == "recommend_order":
            recommendation = result.get("clean", {})
            scope = recommendation.get("scope")

            if recommendation_edit_pending:
                scope = self.recommendation_state.get("scope")
                recommendation["scope"] = scope

                if scope == "selected":
                    recommendation["target_sections"] = self.recommendation_state.get("target_sections", []).copy()

            if scope == "ask":
                self.recommendation_state = {
                    "phase": "await_scope",
                    "started_section": self.section,
                }
                reply = (
                    "현재 단계만 추천해드릴까요, 아니면 남은 주문 전체를 추천해드릴까요? "
                    "못 드시거나 싫어하는 재료가 있으면 함께 말씀해 주세요."
                )

            elif scope == "section":
                proposal = copy.deepcopy(self.order)
                recommended_order = copy.deepcopy(recommendation.get("recommended_order", {}))
                excluded = recommendation.get("excluded", [])
                constraint_changes = recommendation.get("constraints", {})
                proposal_dropped = []

                constraint_clean, constraint_dropped = validate_delta(
                    {"constraints": constraint_changes},
                    self.section,
                )
                apply_delta(proposal, constraint_clean)
                enforce_constraints(proposal)

                # 싫다고 말한 현재 단계 재료는 추천 후보에서도 제외한다.
                for item in excluded:
                    if item in self._section_items():
                        proposal["toppings"].pop(item, None)

                        if isinstance(recommended_order.get("toppings"), dict):
                            recommended_order["toppings"].pop(item, None)

                    elif self.section == "noodle":
                        if proposal["sauce"] == item:
                            proposal["sauce"] = None

                        if proposal["noodle_type"] == item:
                            proposal["noodle_type"] = None

                # 이미 고른 값을 모델이 다시 추천하면 추천안 오류로 기록한다.
                for field in ("sauce", "noodle_type", "noodle_portion"):
                    if field in recommended_order and self.order[field] is not None:
                        proposal_dropped.append(f"already_selected.{field}={recommended_order.pop(field)}")

                if isinstance(recommended_order.get("toppings"), dict):
                    for topping in list(recommended_order["toppings"]):
                        if topping in self.order["toppings"]:
                            amount = recommended_order["toppings"].pop(topping)
                            proposal_dropped.append(f"already_selected.toppings.{topping}={amount}")

                    if not recommended_order["toppings"]:
                        del recommended_order["toppings"]

                clean_proposal, invalid_proposal = validate_delta(
                    recommended_order,
                    self.section,
                )
                proposal_dropped.extend(invalid_proposal)
                apply_delta(proposal, clean_proposal)
                proposal_blocked = enforce_constraints(proposal)
                proposal_missing = missing_required(proposal, self.section)

                proposal_unchanged = recommendation_edit_pending and proposal == self.recommendation_state.get("proposal")

                result["recommendation_validation"] = {
                    "scope": scope,
                    "target_sections": [self.section],
                    "recommended_order": copy.deepcopy(recommended_order),
                    "dropped": copy.deepcopy(constraint_dropped + proposal_dropped),
                    "blocked": copy.deepcopy(proposal_blocked),
                    "missing": copy.deepcopy(proposal_missing),
                    "unchanged": proposal_unchanged,
                }

                if constraint_dropped or proposal_dropped or proposal_blocked or proposal_missing or proposal_unchanged:
                    if recommendation_edit_pending:
                        reply = "추천안 수정 요청을 정확히 처리하지 못했어요. 바꿀 재료를 다시 말씀해 주세요."

                    else:
                        self.recommendation_state = {}
                        reply = "조건에 맞는 추천안을 만들지 못했어요. 원하는 재료를 직접 말씀해 주세요."

                else:
                    self.recommendation_state = {
                        "phase": "confirming",
                        "scope": "section",
                        "started_section": self.section,
                        "proposal": proposal,
                        "recommended_order": copy.deepcopy(recommended_order),
                        "excluded": excluded,
                    }
                    self.last_recommendation = copy.deepcopy(self.recommendation_state)
                    reply = self._recommendation_reply(result, recommended_order, scope, [self.section])
                    result["reply"] = reply

            elif scope in ("selected", "all"):
                proposal = copy.deepcopy(self.order)
                recommended_order = copy.deepcopy(recommendation.get("recommended_order", {}))
                excluded = recommendation.get("excluded", [])
                constraint_changes = recommendation.get("constraints", {})
                current_index = SECTION_ORDER.index(self.section)
                food_sections = ["noodle", "veggie", "meat", "extra"]
                target_sections = recommendation.get("target_sections", []).copy() if scope == "selected" else food_sections[current_index:]
                target_indexes = []
                proposal_dropped = []

                if not target_sections or any(section_name not in food_sections for section_name in target_sections):
                    proposal_dropped.append(f"target_sections={target_sections}")

                else:
                    target_indexes = [SECTION_ORDER.index(section_name) for section_name in target_sections]

                    if target_indexes != list(range(target_indexes[0], target_indexes[-1] + 1)):
                        proposal_dropped.append(f"target_sections={target_sections}")

                    if target_indexes[0] < current_index:
                        proposal_dropped.append(f"target_sections={target_sections}")

                constraint_clean, constraint_dropped = validate_delta({"constraints": constraint_changes}, self.section)
                apply_delta(proposal, constraint_clean)
                enforce_constraints(proposal)

                # 싫다고 한 값은 현재 또는 앞으로 진행할 섹션에서만 제외한다.
                for item in excluded:
                    if recommended_order.get("sauce") == item:
                        recommended_order.pop("sauce")

                    if recommended_order.get("noodle_type") == item:
                        recommended_order.pop("noodle_type")

                    if isinstance(recommended_order.get("toppings"), dict):
                        recommended_order["toppings"].pop(item, None)

                    if item == proposal["sauce"] or item == proposal["noodle_type"]:
                        if self.section == "noodle":
                            if item == proposal["sauce"]:
                                proposal["sauce"] = None

                            if item == proposal["noodle_type"]:
                                proposal["noodle_type"] = None

                        else:
                            proposal_dropped.append(f"excluded.{item}")

                    if item in VEGGIES:
                        item_section = "veggie"

                    elif item in MEATS:
                        item_section = "meat"

                    elif item in EXTRAS:
                        item_section = "extra"

                    else:
                        item_section = None

                    if item_section is not None:
                        item_index = SECTION_ORDER.index(item_section)

                        if item_index < current_index or item_section in self.skipped_sections:
                            if item in proposal["toppings"]:
                                proposal_dropped.append(f"excluded.{item}")

                        else:
                            proposal["toppings"].pop(item, None)

                for field in ("sauce", "noodle_type", "noodle_portion"):
                    if field not in recommended_order:
                        continue

                    if self.order[field] is not None:
                        proposal_dropped.append(f"already_selected.{field}={recommended_order.pop(field)}")

                    elif "noodle" not in target_sections:
                        proposal_dropped.append(f"{field}={recommended_order.pop(field)}")

                    elif self.section != "noodle":
                        proposal_dropped.append(f"{field}={recommended_order.pop(field)}")

                scalar_changes = {}

                for field in ("sauce", "noodle_type", "noodle_portion"):
                    if field in recommended_order:
                        scalar_changes[field] = recommended_order[field]

                scalar_clean, scalar_dropped = validate_delta(scalar_changes, "noodle")
                proposal_dropped.extend(scalar_dropped)
                apply_delta(proposal, scalar_clean)

                recommended_toppings = recommended_order.get("toppings", {})

                if not isinstance(recommended_toppings, dict):
                    proposal_dropped.append(f"toppings={recommended_toppings}")
                    recommended_toppings = {}

                for section_name, section_items in (
                    ("veggie", VEGGIES),
                    ("meat", MEATS),
                    ("extra", EXTRAS),
                ):
                    section_changes = {}

                    for topping, amount in recommended_toppings.items():
                        if topping not in section_items:
                            continue

                        section_index = SECTION_ORDER.index(section_name)

                        if topping in self.order["toppings"]:
                            proposal_dropped.append(f"already_selected.toppings.{topping}={amount}")

                        elif section_name not in target_sections:
                            proposal_dropped.append(f"toppings.{topping}={amount}")

                        elif section_index < current_index or section_name in self.skipped_sections:
                            proposal_dropped.append(f"toppings.{topping}={amount}")

                        elif topping not in excluded:
                            section_changes[topping] = amount

                    clean_toppings, dropped_toppings = validate_delta(
                        {"toppings": section_changes},
                        section_name,
                    )
                    proposal_dropped.extend(dropped_toppings)
                    apply_delta(proposal, clean_toppings)

                for topping, amount in recommended_toppings.items():
                    if topping not in VEGGIES + MEATS + EXTRAS:
                        proposal_dropped.append(f"toppings.{topping}={amount}")

                proposal_blocked = enforce_constraints(proposal)
                proposal_missing = []

                sections_to_check = SECTION_ORDER[current_index:target_indexes[-1] + 1] if target_indexes else []

                if "noodle" not in sections_to_check:
                    sections_to_check.insert(0, "noodle")

                for section_name in sections_to_check:
                    if section_name not in self.skipped_sections:
                        proposal_missing.extend(missing_required(proposal, section_name))

                available_extras = [item for item in EXTRAS if item not in excluded and not (item == "치즈" and ("유제품" in proposal["constraints"] or "비건" in proposal["constraints"]))]

                if "extra" in target_sections and available_extras and not any(item in proposal["toppings"] for item in available_extras):
                    proposal_missing.append("추가 재료")

                proposal_unchanged = recommendation_edit_pending and proposal == self.recommendation_state.get("proposal")

                result["recommendation_validation"] = {
                    "scope": scope,
                    "target_sections": target_sections.copy(),
                    "recommended_order": copy.deepcopy(recommended_order),
                    "dropped": copy.deepcopy(constraint_dropped + proposal_dropped),
                    "blocked": copy.deepcopy(proposal_blocked),
                    "missing": copy.deepcopy(proposal_missing),
                    "unchanged": proposal_unchanged,
                }

                if constraint_dropped or proposal_dropped or proposal_blocked or proposal_missing or proposal_unchanged:
                    if recommendation_edit_pending:
                        reply = "추천안 수정 요청을 정확히 처리하지 못했어요. 바꿀 재료를 다시 말씀해 주세요."

                    else:
                        self.recommendation_state = {}
                        reply = "조건에 맞는 추천안을 만들지 못했어요. 원하는 재료를 직접 말씀해 주세요."
                else:
                    self.recommendation_state = {
                        "phase": "confirming",
                        "scope": scope,
                        "target_sections": target_sections,
                        "started_section": self.section,
                        "proposal": proposal,
                        "recommended_order": copy.deepcopy(recommended_order),
                        "excluded": excluded,
                    }
                    self.last_recommendation = copy.deepcopy(self.recommendation_state)

                    reply = self._recommendation_reply(result, recommended_order, scope, target_sections)
                    result["reply"] = reply

            else:
                reply = "추천 범위를 현재 단계 또는 남은 주문 전체 중에서 말씀해 주세요."



        elif action == "set_order_and_confirm":
            # 하나라도 반영되지 않았으면 현재 섹션을 자동 확정하지 않는다.
            if (
                missing or result.get("blocked", []) or result.get("dropped", [])):
                reply = (f"{reply} 아직 확정되지 않았어요. " "다 고르셨으면 다 골랐다고 말씀해 주세요.")

            else:
                reply = self._start_confirmed_section(reply)

        elif action == "confirm_section":
            if missing:
                reply = self._missing_prompt(missing)

            else:
                reply = self._start_confirmed_section()

        elif action == "set_order":

            if SECTION_START_MODE == "AUTO_WHEN_VALID" and not missing:
                reply = self._confirm_current_section(reply)

            elif SECTION_START_MODE == "WAIT_USER_CONFIRM": # user의 확답 받기
                reply = f"{reply} {self._continue_prompt()}"


        self._record_turn(user_text, action, reply, result)

        self.reply_pub.publish(String(data=reply))

        self.get_logger().info(f"[{self.section}] {user_text} -> {reply}")


    def _parse_vlm_predict(self, raw_text: str)-> str:
        # 모델 답변의 마지막 줄만 실제 판정으로 사용한다

        if not isinstance(raw_text, str):
            return "unknown_retake" # 재시도 해야함

        raw_text = raw_text.strip()

        if not raw_text:
            return "unknown_retake"

        last_line = raw_text.splitlines()[-1].strip()

        if last_line == "판정: PASS":
            return "pass"

        elif last_line == "판정: WRONG_INGREDIENT":
            return "wrong_ingredient"

        elif last_line == "판정: UNKNOWN_RETAKE":
            return "unknown_retake"


        return "unknown_retake"

    def _judge_latest_tasks(self, tasks):
        # 카메라 콜백이 저장한 최신 ROS Image를 PIL 이미지로 바꿔 VLM에 전달 ㅇㅇ -> gemma는 pil을 먹음

        with self.camera_lock:
            if CONFIRM_MULTI_IMAGE:
                camera_messages = list(self.vlm_camera_messages)

            elif self.latest_camera_message is not None:
                camera_messages = [self.latest_camera_message]

            else:
                camera_messages = []

        camera_images = []

        for camera_message in camera_messages:
            camera_image = self._image_message_to_pil(camera_message)

            if camera_image is not None:
                camera_images.append(camera_image)

        failed_tasks = []
        results = []

        # 카메라 이미지가 없거나 깨졌으면 현재 작업을 전부 재시도 대상으로 반환
        if not camera_images:

            for task in tasks:
                failed_tasks.append(task.copy())

                results.append({"class" : task["class"], "repeat_count": task["repeat_count"], "predict": "unknown_retake", "raw": ""})


            return failed_tasks, results

        # 현재 섹션에 재료가 여러 개면 월드카메라 사진으로 하나씩 확인

        for task in tasks:
            expected = task["class"] # 지금 실행되고 있는 class key를 가져옴

            predict, raw_text = self._judge_place(camera_images, expected)

            results.append({"class": expected, "repeat_count": task["repeat_count"], "predict": predict, "raw": raw_text})

            if predict != "pass":
                failed_tasks.append(task.copy())


        return failed_tasks, results

    def _judge_place(self, camera_images, expected):
        # 현재 로봇 작업이 없거나 vlm 함수가 없으면 판정 x

        if self.call_vlm is None or not expected or not camera_images:
            return "unknown_retake", ""

         
        reference_image = self.vlm_reference.get(expected) # key값 넣음

        # 처음 보는 재료면 Chroma에서 대표 이미지를 한 번 가져와 저장
        # 같은 재료를 다시 판정할 때는 저장한 이미지를 재사용(재료별로 캐싱)

        """
        처음 상태
        vlm_reference = {}

        양파 첫 판정
        → Chroma 조회
        → {"양파": 양파_참고이미지}

        버섯 첫 판정
        → Chroma 조회
        → {"양파": 양파_참고이미지, "버섯": 버섯_참고이미지}

        양파 재시도
        → Chroma 조회 안 함
        → 저장된 양파 이미지 재사용
        """


        if reference_image is None and self.vlm_rag is not None: # reference image가 없고, vlm 모드가 켜져 있을때
            reference_image = self.vlm_rag.get_reference(expected)

            if reference_image is not None:
                self.vlm_reference[expected] = reference_image

        if reference_image is None:
            return "unknown_retake", ""

        vlm_response = self.call_vlm(
            [reference_image, *camera_images],
            (
                "너는 식재료 배치 상태를 확인하는 판정기다.\n"
                "사진1은 확인할 식재료의 참고 사진이다.\n"
                "사진2부터 마지막 사진은 로봇 작업 뒤 연속으로 촬영한 월드카메라 사진이다.\n"
                "사진 중 한 장 이상에서 사진1과 같은 종류의 식재료가 명확히 보이면 PASS로 판정한다.\n"
                "명확히 다른 재료가 있거나 해당 재료가 보이지 않으면 WRONG_INGREDIENT로 판정한다.\n"
                "사진이 흐리거나 가려져 확신할 수 없으면 UNKNOWN_RETAKE로 판정한다.\n"
                "마지막 줄에는 반드시 판정: PASS, 판정: WRONG_INGREDIENT, "
                "판정: UNKNOWN_RETAKE 중 하나만 출력한다."
            ),
            f"월드카메라 사진 {len(camera_images)}장에서 {expected}가 올바르게 들어있는지 확인해.",
        )

        return self._parse_vlm_predict(vlm_response), vlm_response



    def _publish_status(self): # 현재 주문 상태를 json화 해서 ui에 발행

        amount_word = {"low": "적게", "normal": "보통", "high": "많이",}

        selected = []


        if self.order["sauce"]:
            selected.append(f"{self.order['sauce']} 소스")

        if self.order["noodle_type"]:
            portion = amount_word.get(self.order["noodle_portion"], "")

            selected.append(f"{self.order['noodle_type']} {portion}".strip())

        for topping, amount in self.order["toppings"].items():
            amount_text = amount_word.get(amount, amount)

            selected.append(f"{topping} {amount_text}")

        for constraint in self.order["constraints"]:
            selected.append(f"제약 : {constraint}")


        section_index = SECTION_ORDER.index(self.section)

        section_label = SECTION_LABELS[self.section]

        if self.order_finished:

            status_text = "포장 완료"
            completed_sections = SECTION_ORDER

        elif self.awaiting_confirm == "cancel_order":
            status_text = "주문 취소 확인 중"
            completed_sections = SECTION_ORDER[:section_index]


        elif self.awaiting_confirm == "final_order":

            status_text = "뚜껑 닫기 전 최종 확인"
            completed_sections = SECTION_ORDER[:section_index]


        elif self.active_task is not None:

            status_text = f"{self.active_task['class']} 담는 중"
            completed_sections = SECTION_ORDER[:section_index]


        else:
            status_text = f"{section_label} 선택 중"
            completed_sections = SECTION_ORDER[:section_index]


        if self.active_task is not None:
            current_task = self.active_task["class"]

        else:
            current_task = section_label

        ui_json = {
            "phase": "완료" if self.order_finished else "주문중",
            "section": section_label,
            "section_status": "완료" if self.order_finished else "진행 중",
            "current_task": current_task,
            "target_list": [],
            "completed": [],
            "selected": selected,
            "auto_running": self.recommendation_state.get("phase") == "running",
            "skipped": [SECTION_LABELS[name] for name in SECTION_ORDER if name in self.skipped_sections],
            "status_text": status_text,
        }

        for task in self.completed_tasks:
            task_text = f"{task['class']} × {task['repeat_count']}"
            ui_json["completed"].append(task_text)

        for section_name in SECTION_ORDER:
            ui_json["target_list"].append(SECTION_LABELS[section_name])


        for section_name in completed_sections:
            if section_name not in self.skipped_sections:
                ui_json["completed"].append(SECTION_LABELS[section_name])


        ui_string = json.dumps(ui_json, ensure_ascii=False) 

        # ensure_ascii=False는 한글을 \uXXXX로 바꾸지 않고 그대로 JSON에 넣음
        # json.dumps는 json을 string으로

        self.status_pub.publish(String(data=ui_string))


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
