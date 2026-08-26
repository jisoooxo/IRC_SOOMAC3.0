import queue # 모델 추론은 ros 콜백에서 돌리면 막혀서 스레드랑 큐로 빼야함
import json
import threading

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String, Int16, Bool
from sensor_msgs.msg import Image

from order_v2 import new_order, missing_required, next_section, build_section_plan, SECTION_ORDER, SECTION_LABELS, VEGGIES, MEATS, EXTRAS
from call_model_v2 import load_model, make_call_model, make_generate_reply, make_call_vlm, SCENE_SYSTEM
from agent_v2 import build_graph
from vlm_rag import VlmRag

ENABLE_VLM = True # 일단 LLM부터 완성. False면 VLM 함수 만들지 않음

ENABLE_TOOL_LORA = True # Tool 전용 어뎁터
TOOL_ADAPTER_PATH = "/home/roma/ros2_ws/src/soomac_irc/outputs/gemma4_tool_lora_v2"

SECTION_SEQUENCE_VLM = True
# True : 현재 섹션의 모든 재료를 담은 뒤 VLM 판정
# False : 소스까지 모두 담은 뒤 뚜껑 닫기 전에 전체 VLM 판정

SECTION_START_MODE = "WAIT_USER_CONFIRM"
# WAIT_USER_CONFIRM : 사용자가 "다 골랐어"라고 해야 로봇 작업 시작
# AUTO_WHEN_VALID : 필수값이 다 채워지면 바로 로봇 작업 시작

WAIT_FOR_ROBOT = True # 로봇이 현재 섹션을 다 담은 뒤 다음 섹션으로 넘어감

CONFIRM_LID = True # 뚜껑 닫기 전 마지막으로 주문 확인

ALLOW_CANCEL_AFTER_ROBOT_START = False
# False : 로봇이 재료를 담기 시작한 뒤에는 음성으로 전체 주문 초기화 금지
# True : 물리 작업이 시작됐어도 재확인 후 주문 상태 초기화

ENABLE_RECOMMENDATION = False # 추천 기능은 애들이랑 얘기하기 전까지 OFF

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
        self.task_queue = [] # 아직 로봇한테 안 보낸 작업
        self.active_task = None # 지금 로봇이 하고 있는 작업(방금 보낸거)
        self.completed_tasks = [] # 작업 성공까지 확인된 작업을 담을 리스트

        self.waiting_pick = False # 로봇이 현재 섹션을 담는 중인지

        self.robot_started = False # /llm/plan 발행 여부 ㅇㅇ

        self.awaiting_confirm = None
        # None : 확인 대기 아님
        # final_order : 뚜껑 닫기 전 마지막 확인 대기
        # cancel_order : 전체 주문 취소 확인 대기

        self.order_finished = False

        self.ui_started = False 
        # True : UI 시작 버튼을 눌렀고 main의 첫 /llm/next=3을 기다리는 중
        # False : 아직 UI 시작 전이거나 이미 주문 대화를 시작함
        self.conversation_started = False
        # True : UI start 뒤 main의 첫 /llm/next=3까지 받아 실제 주문 대화 중
        # False : 시작 전이거나 리셋됨

        # 모델 하나만 로드해서 tool, reply, 나중에 vlm까지 공유

        self.get_logger().info("모델 로딩중")

        model, processor = load_model(TOOL_ADAPTER_PATH if ENABLE_TOOL_LORA else None)

        call_model = make_call_model(model, processor)

        generate_reply = make_generate_reply(model, processor)

        self.graph = build_graph(call_model, generate_reply) # 함수 자체를 넘긴다.
        self.call_vlm = None # 얘는 나중에
        self.vlm_reference = {} 
        self.vlm_rag = None

        self.latest_camera_message = None
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

        self.vlm_result_pub = self.create_publisher(String, "/vlm/result", 10) # vlm 판정 결과를 main에 전달

        # 구독 콜백이 받은 작업을 worker에 넘길 큐
        # 구독 직후 메시지가 들어와도 사용할 수 있도록 구독보다 먼저 만든다.

        self._jobs = queue.Queue()

        # ROS 구독

        self.stt_sub = self.create_subscription(String, "/stt_question", self.stt_callback, 10)
        self.ui_start_sub = self.create_subscription(String, "/ui/start", self.ui_start_callback, 10)
        self.ui_reset_sub = self.create_subscription(String, "/ui/reset", self.ui_reset_callback, 10)
        self.next_trigger_sub = self.create_subscription(Int16, "/llm/next", self.trigger_callback, 10) # main이 제어 작업과 홈 복귀 완료 후 3을 발행

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


    def trigger_callback(self, msg: Int16):
        self._jobs.put(("next", msg.data))

    def camera_callback(self, msg: Image):
        # 카메라는 작업 큐에 계속 쌓지 않고 최신 메시지만 교체한다.
        # 주문 상태는 worker만 수정하고, 카메라 메시지는 lock으로 따로 보호한다.

        with self.camera_lock:
            self.latest_camera_message = msg


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
                    self._reset(data)

                elif mode == "finish":
                    if self.section != "lid" or self.active_task is None:
                        self.get_logger().warning("포장 작업 중이 아니어서 /llm/reset을 무시함")

                    elif self.active_task["class"] != "뚜껑":
                        self.get_logger().warning("현재 작업이 cover가 아니어서 /llm/reset을 무시함")

                    else:
                        completed_task = self.active_task
                        self.active_task = None

                        if completed_task not in self.completed_tasks:
                            self.completed_tasks.append(completed_task.copy())

                        self.reply_pub.publish(String(data="포장이 끝났어요. 이용해 주셔서 감사합니다."))
                        self._finish_order()

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

        self._clear_state()
        self._set_stt_enabled(False)
        self.ui_started = True
        self.get_logger().info("UI 시작 수신, main의 /llm/next=3 대기")



    def _start_order(self):
         # UI 시작 뒤 main의 첫 /llm/next=3을 받으면 실제 주문 대화 시작

        self.ui_started = False
        self.conversation_started = True
        self._set_stt_enabled(True)

        greeting_ment= "안녕하세요! 저는 스파게티 밀키트 주문을 도와드리는 로봇이에요. "

        self.reply_pub.publish(String(data=greeting_ment + "먼저 면 종류, 소스, 양을 골라주세요."))
        self.get_logger().info("새 주문 시작")


    def _reset(self, reason: str): # 이유(문자열)
        self._clear_state() # 내부 변수 전부 초기화
        self._set_stt_enabled(False)
        self.reply_pub.publish(String(data="주문을 초기화했어요. 처음부터 다시 시작할게요."))
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
        self.awaiting_confirm = None
        self.order_finished = False
        self.ui_started = False
        self.conversation_started = False


    def _section_items(self):
        # 현재 섹션에서 선택할 수 있는 재료 반환

        if self.section == "veggie":
            return VEGGIES

        elif self.section == "meat":
            return MEATS

        elif self.section == "extra":
            return EXTRAS


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

        section_items = self._section_items()

        already = []

        remaining = []

        for item in section_items:

            if item in self.order["toppings"]:
                already.append(item)

            else:
                remaining.append(item)


        if already and remaining:
            return f"다음은 {section_label}입니다. {', '.join(already)}는 골랐어요. {', '.join(remaining)}도 추가할까요?"

        elif already:
            return f"다음은 {section_label}입니다. {', '.join(already)}를 골랐어요. 더 없으면 다 골랐다고 말씀해 주세요."

        return f"다음은 {section_label}입니다. {', '.join(section_items)} 중에서 골라주세요."


    def _missing_prompt(self, missing):

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


    def _commit_section(self, reply): 

        # 현재 섹션 끝내고 다음 섹션으로 이동. 마무리 되면 실행할 함수인듯

        done_label = SECTION_LABELS[self.section] 
        next_step = next_section(self.section)

        self.waiting_pick = False

        if next_step is None:
            self._finish_order()
            return f"{reply} 주문이 모두 완료되었습니다. 이용해 주셔서 감사합니다." # 마무리!

        self.section = next_step

        self.get_logger().info(f"다음 섹션 : {self.section}")

        # 소스는 면 단계에서 이미 골랐으므로 사용자한테 다시 안 물어보고 바로 담음

        if self.section == "sauce":
            sauce_tasks = build_section_plan(self.order, self.section)


            if not sauce_tasks:
                return f"{reply} 선택된 소스가 없어서 작업을 진행할 수 없어요."


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

            if CONFIRM_LID:
                self.awaiting_confirm = "final_order"
                return f"{reply} {self._order_summary()} 이대로 포장할까요?"


            return self._confirm_current_section(f"{reply} 포장을 시작할게요.")


        return (
            f"{reply} "
            f"{done_label} 선택이 끝났어요. "
            f"{self._section_select_prompt()}"
        )


    def _process_next(self, result_code: int):
        # 현재 재료 작업과 홈 복귀를 마치면 /llm/next에 3을 보냄(vlm 전에)
        # VLM은 현재 작업 성공 시 /llm/next를 발행
        # 성공한 작업은 active_task에서 빼고 다음 작업을 확인

        self.get_logger().info(f"/llm/next 수신 : {result_code}")

        # 지수 / vlm한테 오는 trigger 토픽 (/llm/next) 검증(상단에 BY VLM NUM으로 유동적으로 관리 ㄱㄱㄱㄱ)

        if result_code != BY_VLM_NUM:
            self.get_logger().warning("우리 토픽 규약이랑 맞지 않아용")
            return

        # ui 시작 요청 뒤 처음 받은 3은 로봇 작업 완료가 아닌, 주문 시작

        if self.ui_started:
            self._start_order()
            return
        
        # 중복으로 호출되어도 다음 작업을 잘못하지 않게 사전에 막아버령

        if self.active_task is None:
            self.get_logger().warning("진행 중인 작업이 없어서 /llm/next를 무시함")
            return

        # cover 완료는 /llm/next가 아니라 main의 /llm/reset으로 확정한다.
        if self.section == "lid":
            self.get_logger().warning("cover 작업 완료를 위해 /llm/reset을 기다리는 중")
            return

        completed_task = self.active_task
        self.active_task = None

        self.get_logger().info(f"현재 하고 있는 작업 완료 : {completed_task}")



        if self.task_queue:

            if not ENABLE_VLM or not SECTION_SEQUENCE_VLM:

                if completed_task not in self.completed_tasks:
                    self.completed_tasks.append(completed_task.copy())

            self._publish_next_task()
            return


        # task_queue가 비었고 waiting_pick이면 현재 섹션 작업이 전부 끝난 것

        if not self.waiting_pick:
            return

        vlm_tasks = []

        # 섹션 마다 판정하는 경우엔 현재 섹션 작업만 확인한다

        # 식재료 작업만 VLM으로 확인하고 cover는 제어 완료 신호만 기다린다.
        if ENABLE_VLM and SECTION_SEQUENCE_VLM and self.section != "lid":
            vlm_tasks = build_section_plan(self.order, self.section)


        # 마지막에 판정하면 소스 작업 끝났을 때 전체 주문 확인

        elif ENABLE_VLM and not SECTION_SEQUENCE_VLM and self.section == "sauce":

            for section_name in SECTION_ORDER:
                section_tasks = build_section_plan(self.order, section_name)
                vlm_tasks.extend(section_tasks)

        if vlm_tasks:
            failed_tasks, vlm_results = self._judge_latest_tasks(vlm_tasks)

            # VLM return 결과를 main에 재료별로 발행

            for result in vlm_results:
                result_for_publish = {
                    "class": TOPIC_CLASS_NAMES[result["class"]],
                    "repeat_count": result["repeat_count"],
                    "success": result["predict"] == "pass",
                    "status": result["predict"],
                }


                self.vlm_result_pub.publish(
                    String(data=json.dumps(result_for_publish, ensure_ascii=False))
                )

                self.get_logger().info(f"VLM 결과 발행 : {result_for_publish}")

            # 실패 작업만 다시 plan으로 ㅇㅇ

            if failed_tasks:
                failed_names = []

                for task in failed_tasks:
                    failed_names.append(task["class"])

                completed_after_vlm = []

                for task in self.completed_tasks:

                    if task["class"] not in failed_names:
                        completed_after_vlm.append(task)

                self.completed_tasks = completed_after_vlm
                self.task_queue.extend(failed_tasks)
                self._publish_next_task()


                reply = (f"{', '.join(failed_names)} 위치를 확인하지 못했어요. "
                         "해당 작업을 다시 시도할게요.")

                self.reply_pub.publish(String(data=reply))
                return


        # 전부 PASS일 때만 완료 목록에 추가

            for task in vlm_tasks:

                if task not in self.completed_tasks:
                    self.completed_tasks.append(task.copy())

            # VLM을 사용하지 않는 기존 모드

        elif completed_task not in self.completed_tasks:
            self.completed_tasks.append(completed_task.copy())

        self.get_logger().info(f"총 작업 완료 : {self.completed_tasks}")


        if self.section == "lid":
            reply = "포장이 끝났어요."

        else:
            reply = f"{completed_task['class']} 담기가 끝났어요."

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
        self.order_finished = True

        self.done_pub.publish(Bool(data=True))
        self.get_logger().info("총 주문 완료")


    def _order_summary(self):
        # 뚜껑 닫기 전 최종 주문 문장

        amount_word = {
            "low": "적게",
            "normal": "보통",
            "high": "많이",
        }

        parts = []


        if self.order["sauce"]:
            parts.append(f"{self.order['sauce']} 소스")


        if self.order["noodle_type"]:

            portion = amount_word.get(
                self.order["noodle_portion"],
                "",
            )

            parts.append(
                f"{self.order['noodle_type']} {portion}".strip()
            )


        for topping, amount in self.order["toppings"].items():
            amount_text = amount_word.get(amount, amount)
            parts.append(f"{topping} {amount_text}")


        if not parts:
            return "현재 주문"


        return "주문은 " + ", ".join(parts) + "입니다."


    def _continue_prompt(self): # WAIT_USER_CONFIRM에서 추가 주문을 받을 문장 추가

        missing = missing_required(self.order, self.section)

        if missing:
            return self._missing_prompt(missing)

        if self.section == "noodle":
            if ENABLE_RECOMMENDATION:
                return "치즈와 페퍼론치노도 미리 추가할 수 있어요. 더 없으면 다 골랐다고 말씀해 주세요."

            return "더 없으면 다 골랐다고 말씀해 주세요."
                
        section_items = self._section_items()

        remaining = []

        for item in section_items:
            if item not in self.order["toppings"]:
                remaining.append(item)


        if remaining:
            return f"{', '.join(remaining)}도 추가할 수 있어요. 더 없으면 다 골랐다고 말씀해 주세요."


        return "현재 단계의 재료를 모두 선택했어요. 이대로 담으려면 다 골랐다고 말씀해 주세요."

    def _is_affirmative(self, user_text):
        # 문장 끝에 기호가 붙어도 긍정 처리하게 텍스트 전처리

        normalized = user_text.strip().rstrip(".!?,~ ")

        return normalized in AFFIRM_WORDS

    def _process(self, user_text: str):
        if not self.conversation_started:
            self.get_logger().warning("주문 대화 시작 전 STT 결과를 무시함")
            return


        if self.order_finished:
            self.reply_pub.publish(String(data="주문이 완료됐어요. 새 주문은 처음부터 시작해주세요."))
            return

        # 여기부터 추가
        # cancel_order는 모델이 골라도 바로 초기화하지 않고 한 번 더 확인하도록

        if self.awaiting_confirm == "cancel_order":

            if self._is_affirmative(user_text):

                # 취소 질문 뒤 로봇 작업이 시작됐을 가능성도 다시 검사
                if self.robot_started and not ALLOW_CANCEL_AFTER_ROBOT_START:
                    self.awaiting_confirm = None
                    reply = "이미 재료를 담기 시작해서 자동으로 주문을 취소할 수 없어요."

                else:
                    self._reset("손님 취소 확정") # reason 들어감(문자열)
                    return

            else:
                self.awaiting_confirm = None
                reply = "주문을 취소하지 않고 계속 진행할게요."


            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})

            self.reply_pub.publish(String(data=reply))
            return


        # 뚜껑 닫기 전 마지막 확인은 모델보다 먼저 처리 -> 왜?
        # 마지막 확인은 Python이 처리해야 모델 오판으로 포장이 확정되지 않음

        if self.awaiting_confirm == "final_order":

            if self._is_affirmative(user_text):
                self.awaiting_confirm = None
                reply = self._confirm_current_section("네, 이대로 포장을 시작할게요.")

            else:
                reply = "주문은 이미 담았어요. 이대로 포장할까요?"


            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})

            self.reply_pub.publish(String(data=reply))
            return


        # 로봇이 현재 섹션을 담고 있을 때는 주문을 새로 받지 않는다.

        if self.waiting_pick:

            self.reply_pub.publish(String(data="지금 로봇이 담고 있어요. 잠시만 기다려 주세요."))
            return


        result = self.graph.invoke({
            "user_text": user_text,
            "order": self.order,
            "section": self.section,
            "history": self.history,
            "completed_tasks": self.completed_tasks,
        }) # 그래프 내부 state

        self.order = result["order"]
        action = result["action"]
        reply = result["reply"]
        missing = result["missing"]

        self.get_logger().info(f"[action] name={action} " f"clean={result.get('clean')} "  f"dropped={result.get('dropped')} "  f"blocked={result.get('blocked')}")

        if action == "describe_scene":
            reply = self._describe_scene(user_text)

        elif action == "cancel_order":

            if self.robot_started and not ALLOW_CANCEL_AFTER_ROBOT_START:
                reply = "이미 재료를 담기 시작해서 자동으로 주문을 취소할 수 없어요."

            else:
                self.awaiting_confirm = "cancel_order"
                reply = "주문을 전부 취소하고 처음부터 다시 시작할까요?"

        elif action == "confirm_section":
            if missing:
                reply = self._missing_prompt(missing) 

            else:
                reply = self._confirm_current_section(reply)

        elif action == "set_order":

            if SECTION_START_MODE == "AUTO_WHEN_VALID" and not missing:
                reply = self._confirm_current_section(reply)

            elif SECTION_START_MODE == "WAIT_USER_CONFIRM": # user의 확답 받기
                reply = f"{reply} {self._continue_prompt()}"


        # 그리고 사용자와 tts에 전달한 문장을 history에 저장

        self.history.append(result["user_msg"])

        self.history.append({"role":"assistant", "content": reply}) # 모델의 대답

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
            camera_message = self.latest_camera_message

        camera_image = self._image_message_to_pil(camera_message)

        failed_tasks = []
        results = []

        # 카메라 이미지가 없거나 깨졌으면 현재 작업을 전부 재시도 대상으로 반환
        if camera_image is None:

            for task in tasks:
                failed_tasks.append(task.copy())

                results.append({"class" : task["class"], "repeat_count": task["repeat_count"], "predict": "unknown_retake", "raw": ""})


            return failed_tasks, results

        # 현재 섹션에 재료가 여러 개면 월드카메라 사진으로 하나씩 확인

        for task in tasks:
            expected = task["class"] # 지금 실행되고 있는 class key를 가져옴

            predict, raw_text = self._judge_place(camera_image, expected)

            results.append({"class": expected, "repeat_count": task["repeat_count"], "predict": predict, "raw": raw_text})

            if predict != "pass":
                failed_tasks.append(task.copy())


        return failed_tasks, results

    def _judge_place(self, camera_image, expected):
        # 현재 로봇 작업이 없거나 vlm 함수가 없으면 판정 x

        if self.call_vlm is None or not expected:
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
            [reference_image, camera_image],
            (
                "너는 식재료 배치 상태를 확인하는 판정기다.\n"
                "사진1은 확인할 식재료의 참고 사진이다.\n"
                "사진2는 로봇이 재료를 담은 뒤 촬영한 월드카메라 사진이다.\n"
                "사진2 안에 사진1과 같은 종류의 식재료가 명확히 보이면 PASS로 판정한다.\n"
                "명확히 다른 재료가 있거나 해당 재료가 보이지 않으면 WRONG_INGREDIENT로 판정한다.\n"
                "사진이 흐리거나 가려져 확신할 수 없으면 UNKNOWN_RETAKE로 판정한다.\n"
                "마지막 줄에는 반드시 판정: PASS, 판정: WRONG_INGREDIENT, "
                "판정: UNKNOWN_RETAKE 중 하나만 출력한다."
            ),
            f"사진2 안에 {expected}가 올바르게 들어있는지 확인해.",
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
            "status_text": status_text,
        }

        for task in self.completed_tasks:
            task_text = f"{task['class']} × {task['repeat_count']}"
            ui_json["completed"].append(task_text)

        for section_name in SECTION_ORDER:
            ui_json["target_list"].append(SECTION_LABELS[section_name])


        for section_name in completed_sections:
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