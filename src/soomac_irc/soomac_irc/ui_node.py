#!/home/roma/miniconda3/envs/cosyvoice3/bin/python

import json
import signal
import socket
import threading
import traceback
from collections import deque
from pathlib import Path

from flask import Flask, render_template, request
from flask_socketio import SocketIO


try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from std_msgs.msg import Bool, String
except Exception as error:
    rclpy = None
    Node = object
    String = None
    Bool = None
    ExternalShutdownException = type(
        'ExternalShutdownException', (Exception,), {})
    ROS_IMPORT_ERROR = error
else:
    ROS_IMPORT_ERROR = None



PUBLISH_QUEUE_SIZE = 10
SUBSCRIPTION_QUEUE_SIZE = 10
WEB_HOST = '0.0.0.0'
WEB_PORT = 5000
ROS_SPIN_TIMEOUT_SECONDS = 0.1
ROS_THREAD_JOIN_TIMEOUT_SECONDS = 2.0
UI_START_MESSAGE = 'start'
MAX_CACHED_DIALOGUE_COUNT = 200

TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / 'templates'
# Socket.IO 클라이언트 JS 를 여기서 직접 서빙한다. flask-socketio 는 서버만 제공하고
#   클라이언트 라이브러리는 안 준다. CDN 을 걸면 대회장에 인터넷이 없을 때 화면이 통째로 죽는다.
STATIC_DIRECTORY = Path(__file__).resolve().parent / 'static'

app = Flask(__name__,
            template_folder=str(TEMPLATE_DIRECTORY),
            static_folder=str(STATIC_DIRECTORY))
socketio = SocketIO(app, async_mode='threading')

# python-socketio의 emit은 동시 호출 때 여러 패킷의 순서가 섞일 수 있어 한 잠금으로 직렬화한다.
socket_emit_lock = threading.RLock()
ros_node_lock = threading.Lock()
ros_stop_event = threading.Event()

latest_mic_state = 'idle'
latest_stt_enabled = False
ros_connected = False
latest_agent_status = {}
ui_session_active = False
cached_dialogue = deque(maxlen=MAX_CACHED_DIALOGUE_COUNT)
ros_node = None


def emit_mic_state(state):
    """마지막 상태를 저장하고 현재 브라우저 모두에 보낸다."""
    global latest_mic_state

    with socket_emit_lock:
        latest_mic_state = state
        socketio.emit('mic_state', {'state': state})

def sync_mic_with_stt(enabled=None, *, tts_done=False):
    """STT 허용값을 저장하고 실제 듣기 전환 시점에 맞춰 표시한다."""
    global latest_mic_state, latest_stt_enabled

    with socket_emit_lock:
        previous_enabled = latest_stt_enabled

        if enabled is not None:
            latest_stt_enabled = bool(enabled)

        if not ui_session_active:
            return

        if tts_done:
            next_state = (
                'listening' if latest_stt_enabled else 'waiting'
            )

        elif enabled is True:
            if previous_enabled:
                return

            if latest_mic_state == 'speaking':
                return

            next_state = 'waiting'

        else:
            next_state = 'waiting'

        latest_mic_state = next_state
        socketio.emit(
            'mic_state',
            {'state': latest_mic_state},
        )


def emit_dialogue(event_name, text):
    """Chroma 저장과 별개로 현재 UI 세션의 최근 대화를 메모리에 보관한다."""
    payload = {'text': text}
    with socket_emit_lock:
        cached_dialogue.append((event_name, payload))
        socketio.emit(event_name, payload)


def emit_agent_status(status):
    """진행 상태 전체를 한 번에 저장하고 브라우저에 보낸다."""
    global latest_agent_status

    with socket_emit_lock:
        latest_agent_status = status
        socketio.emit('agent_status', status)


def emit_ros_status(connected):
    """ROS 연결 여부를 저장해 새 브라우저에도 같은 상태를 보낸다."""
    global ros_connected

    with socket_emit_lock:
        ros_connected = connected
        socketio.emit('ros_status', {'connected': connected})


def is_ui_session_active():
    with socket_emit_lock:
        return ui_session_active


def activate_ui_session():
    global ui_session_active

    with socket_emit_lock:
        ui_session_active = True


def reset_ui_session(action):
    """영구 Chroma 기록은 보존하고 현재 화면 세션만 비운다."""
    global latest_mic_state, latest_stt_enabled
    global latest_agent_status, ui_session_active

    with socket_emit_lock:
        cached_dialogue.clear()
        latest_mic_state = 'idle'
        latest_stt_enabled = False
        latest_agent_status = {}
        ui_session_active = False
        socketio.emit('dialogue_snapshot', {'items': []})
        socketio.emit('agent_status', latest_agent_status)
        socketio.emit('mic_state', {'state': latest_mic_state})
        socketio.emit('work_reset', {'action': action})



UI_START_TOPIC = '/ui/start'           # 손님이 시작할 때 다른 ROS 노드가 대화를 열도록 알린다.
UI_RESET_TOPIC = '/ui/reset'           # 뒤로가기·홈에서 에이전트와 제어기의 현재 작업을 초기화한다.
AGENT_STATUS_TOPIC = '/agent/status'    # 에이전트의 선택·목표·현재 작업·완료 상태를 한 payload로 받는다.
RESET_ACTIONS = {'back', 'home'}

class UiNode(Node):
    def __init__(self):
        super().__init__('soomac_ui_node')

        self.lifecycle_lock = threading.Lock()
        self.destroying = False

        self.start_publisher = self.create_publisher(String, UI_START_TOPIC, PUBLISH_QUEUE_SIZE)
        self.reset_publisher = self.create_publisher(String, UI_RESET_TOPIC, PUBLISH_QUEUE_SIZE)

        # STT가 손님의 최종 문장을 확정했을 때 화면에 보여준다.
        self.question_subscription = self.create_subscription(String, '/stt_question', self.stt_question_callback, SUBSCRIPTION_QUEUE_SIZE)

        # LLM 답변은 TTS 시작과 순서가 다를 수 있어 대사만 갱신한다.
        self.response_subscription = self.create_subscription(String, '/llm_response', self.llm_response_callback, SUBSCRIPTION_QUEUE_SIZE)

        # TTS가 재생 직전에 보내므로 마이크를 말하기 상태로 바꾼다.
        self.stt_stop_subscription = self.create_subscription(String, '/stt_stop', self.stt_stop_callback,SUBSCRIPTION_QUEUE_SIZE)

        self.stt_enable_subscription = self.create_subscription(
            Bool,
            '/stt/enable',
            self.stt_enable_callback,
            SUBSCRIPTION_QUEUE_SIZE,
        )

        # 재생 성공과 실패 모두 다시 들을 수 있다는 신호로 사용한다.
        self.tts_done_subscription = self.create_subscription(String, '/tts_done', self.tts_done_callback, SUBSCRIPTION_QUEUE_SIZE)
        self.agent_status_subscription = self.create_subscription(String, AGENT_STATUS_TOPIC, self.agent_status_callback,SUBSCRIPTION_QUEUE_SIZE)

        self.get_logger().info('손님용 UI ROS 노드가 준비됐어요.')

    def publish_start(self):
        # Flask 스레드와 종료가 겹쳐도 폐기된 발행기를 만지지 않도록 생명주기를 잠근다.
        with self.lifecycle_lock:
            if self.destroying:
                return False
            self.start_publisher.publish(String(data=UI_START_MESSAGE))

        self.get_logger().info('손님이 시작 버튼을 눌러 /ui/start를 보냈어요.')
        return True

    def publish_reset(self, action):
        if action not in RESET_ACTIONS:
            return False

        with self.lifecycle_lock:
            if self.destroying:
                return False
            payload = json.dumps({'action': action}, ensure_ascii=False)
            self.reset_publisher.publish(String(data=payload))

        self.get_logger().warning(
            f'손님이 {action} 버튼을 눌러 {UI_RESET_TOPIC}을 보냈어요.')
        return True

    def stt_question_callback(self, message):
        try:
            if not is_ui_session_active():
                self.get_logger().info('비활성 UI 세션의 손님 문장을 표시하지 않았어요.')
                return
            question_text = message.data.strip()
            if not question_text:
                self.get_logger().info('빈 손님 문장은 화면에 보내지 않았어요.')
                return

            # 말풍선 다음에 thinking을 보내야 브라우저가 같은 발화 흐름으로 그린다.
            emit_dialogue('user_said', question_text)
            emit_mic_state('thinking')
            self.get_logger().info(f'손님 문장을 화면에 보냈어요: {question_text}')
        except Exception as error:
            self.get_logger().error(
                f'손님 문장을 화면에 보내다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}')

    def llm_response_callback(self, message):
        try:
            if not is_ui_session_active():
                self.get_logger().info('비활성 UI 세션의 로봇 대사를 표시하지 않았어요.')
                return
            response_text = message.data.strip()
            if not response_text:
                self.get_logger().info('빈 로봇 대사는 화면에 보내지 않았어요.')
                return

            # TTS 시작 신호와 순서가 보장되지 않으므로 여기서는 마이크 상태를 건드리지 않는다.
            emit_dialogue('bot_say', response_text)
            self.get_logger().info(f'로봇 대사를 화면에 보냈어요: {response_text}')
        except Exception as error:
            self.get_logger().error(
                f'로봇 대사를 화면에 보내다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}')

    def stt_stop_callback(self, _message):
        try:
            if not is_ui_session_active():
                return
            # 이 신호가 LLM 답변보다 먼저 와도 마지막 신호가 상태를 결정해야 한다.
            emit_mic_state('speaking')
            self.get_logger().info('TTS가 시작되어 화면을 말하는 중으로 바꿨어요.')
        except Exception as error:
            self.get_logger().error(
                f'말하기 상태를 화면에 보내다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}')

    def stt_enable_callback(self, message):
        try:
            enabled = bool(message.data)
            sync_mic_with_stt(enabled)
            self.get_logger().info(
                f'/stt/enable={enabled}를 화면 상태로 저장했어요.'
            )
        except Exception as error:
            self.get_logger().error(
                f'STT 허용 상태를 화면에 보내다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}'
            )

    def tts_done_callback(self, _message):
        try:
            if not is_ui_session_active():
                return

            sync_mic_with_stt(tts_done=True)
            self.get_logger().info(
                'TTS 종료 후 마지막 /stt/enable 상태를 화면에 반영했어요.'
            )
        except Exception as error:
            self.get_logger().error(
                f'TTS 종료 상태를 화면에 보내다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}'
            )

    def agent_status_callback(self, message):
        try:
            if not is_ui_session_active():
                return
            status = json.loads(message.data)
            if not isinstance(status, dict):
                raise ValueError('최상위 JSON은 객체여야 합니다.')

            emit_agent_status(status)
            self.get_logger().info('에이전트 진행 상태를 화면에 보냈어요.')
        except (json.JSONDecodeError, ValueError) as error:
            self.get_logger().warning(
                f'{AGENT_STATUS_TOPIC} 메시지를 표시하지 않았어요: {error}')
        except Exception as error:
            self.get_logger().error(
                f'에이전트 진행 상태를 화면에 보내다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}')

    def destroy_node(self):
        # 발행을 먼저 막고 ROS 자원을 닫아 Flask 스레드가 종료 중인 발행기를 쓰지 않게 한다.
        with self.lifecycle_lock:
            self.destroying = True
            super().destroy_node()


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    try:
        # 기록과 전송을 같은 잠금에서 처리해야 새 이벤트 뒤에 낡은 상태가 덮이지 않는다.
        with socket_emit_lock:
            dialogue_items = [
                {'event': event_name, 'text': payload['text']}
                for event_name, payload in cached_dialogue
            ]
            socketio.emit(
                'dialogue_snapshot', {'items': dialogue_items}, to=client_id)
            socketio.emit(
                'mic_state', {'state': latest_mic_state}, to=client_id)
            socketio.emit(
                'ros_status', {'connected': ros_connected}, to=client_id)
            socketio.emit(
                'agent_status', latest_agent_status, to=client_id)
        print('브라우저가 붙었어요.')
    except Exception as error:
        print(
            f'브라우저 초기 상태를 보내다가 터졌어요: '
            f'{error}\n{traceback.format_exc()}')


@socketio.on('disconnect')
def handle_disconnect():
    print('브라우저 연결이 끊겼어요.')


@socketio.on('start')
def handle_start(_payload=None):
    try:
        with ros_node_lock:
            current_ros_node = ros_node

        if current_ros_node is None:
            print('ROS가 연결되지 않아 /ui/start는 보내지 못했어요.')
        else:
            current_ros_node.publish_start()

        # ROS가 없어도 디자인을 확인할 수 있어야 하므로 화면 상태는 독립적으로 전환한다.
        activate_ui_session()
        sync_mic_with_stt()
    except Exception as error:
        print(
            f'시작 요청을 처리하다가 터졌어요: '
            f'{error}\n{traceback.format_exc()}')


@socketio.on('reset_work')
def handle_reset_work(payload=None):
    try:
        action = payload.get('action') if isinstance(payload, dict) else None
        if action not in RESET_ACTIONS:
            return {'ok': False, 'message': '지원하지 않는 초기화 요청입니다.'}

        with ros_node_lock:
            current_ros_node = ros_node

        if current_ros_node is None:
            return {'ok': False, 'message': 'ROS가 연결되지 않아 초기화하지 못했습니다.'}
        if not current_ros_node.publish_reset(action):
            return {'ok': False, 'message': 'ROS 노드가 종료 중이라 초기화하지 못했습니다.'}

        reset_ui_session(action)
        return {'ok': True}
    except Exception as error:
        print(
            f'작업 초기화 요청을 처리하다가 터졌어요: '
            f'{error}\n{traceback.format_exc()}')
        return {'ok': False, 'message': '초기화 처리 중 오류가 발생했습니다.'}


def run_ros(args=None):
    """ROS 초기화 실패가 웹 서버를 막지 않도록 별도 스레드에서 실행한다."""
    global ros_node

    node = None
    try:
        if rclpy is None:
            raise RuntimeError(f'ROS 모듈을 불러오지 못했어요: {ROS_IMPORT_ERROR}')

        rclpy.init(args=args)
        node = UiNode()
        with ros_node_lock:
            ros_node = node
        emit_ros_status(True)

        while not ros_stop_event.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=ROS_SPIN_TIMEOUT_SECONDS)

    except ExternalShutdownException:
        pass
    except Exception as error:
        print(f'ROS를 시작하거나 실행하다가 터졌어요: {error}\n{traceback.format_exc()}')

    finally:
        with ros_node_lock:
            ros_node = None
        emit_ros_status(False)

        # stt_nemotron_node와 같은 순서로 노드를 먼저 버리고 rclpy 문맥을 닫는다.
        if node is not None:
            try:
                node.destroy_node()
            except Exception as error:
                print(
                    f'UI ROS 노드를 정리하다가 터졌어요: '
                    f'{error}\n{traceback.format_exc()}')

        if rclpy is not None and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as error:
                print(
                    f'ROS 문맥을 정리하다가 터졌어요: '
                    f'{error}\n{traceback.format_exc()}')


def is_web_port_free():
    """웹 포트가 비어 있는지 미리 확인한다.

    ROS 스레드를 먼저 띄우면 웹 서버가 포트 충돌로 못 뜨는데도 같은 이름의 ROS 노드가
    하나 더 붙는다. 그러면 토픽을 두 노드가 같이 받아 화면 상태가 어긋난다.
    붙잡지 않고 확인만 하므로 그 사이에 남이 채갈 수는 있지만, 실수로 두 번 띄우는
    흔한 경우는 이걸로 걸린다.
    """
    probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Flask 서버가 SO_REUSEADDR로 뜨므로 프로브도 동일하게 → Ctrl+C 직후 TIME_WAIT 오탐 방지
    try:
        probe_socket.bind((WEB_HOST, WEB_PORT))
        return True
    except OSError:
        return False
    finally:
        probe_socket.close()


def install_termination_handler():
    """SIGTERM 을 Ctrl+C 와 같은 경로로 흘린다.

    기본 상태에서는 SIGTERM 이 오면 파이썬이 그냥 죽어서 아래 finally 가 안 돌고,
    ROS 노드와 포트가 남는다. 실제로 timeout 으로 띄운 프로세스가 안 죽고 남아
    다음 실행이 "Address already in use" 로 실패했다.
    KeyboardInterrupt 로 바꿔 던지면 이미 있는 종료 경로를 그대로 탄다.
    """
    def 종료요청(signal_number, frame):
        print('종료 신호를 받았어요. 정리하고 내려갑니다.')
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, 종료요청)


def main(args=None):
    install_termination_handler()

    if not is_web_port_free():
        print(f'{WEB_HOST}:{WEB_PORT} 가 이미 쓰이고 있어요. '
              '먼저 떠 있는 UI 를 끄고 다시 실행해 주세요.')
        return 1

    ros_stop_event.clear()
    ros_thread = threading.Thread(
        target=run_ros,
        args=(args,),
        name='ui-ros-spin',
        daemon=True)
    ros_thread.start()

    try:
        socketio.run(
            app,
            host=WEB_HOST,
            port=WEB_PORT,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'웹 서버가 터졌어요: {error}\n{traceback.format_exc()}')
        raise
    finally:
        ros_stop_event.set()
        ros_thread.join(timeout=ROS_THREAD_JOIN_TIMEOUT_SECONDS)
        if ros_thread.is_alive():
            print(
                f'ROS 스레드가 {ROS_THREAD_JOIN_TIMEOUT_SECONDS:.0f}초 안에 '
                '끝나지 않았어요.')
    return 0


if __name__ == '__main__':
    # 포트 충돌로 못 떴을 때 종료 코드가 0 이면 스크립트나 launch 가 성공으로 오해한다.
    raise SystemExit(main())
