#!/home/roma/miniconda3/envs/ros_env/bin/python

import json
import os
import queue
import threading
import time
import traceback
from collections import deque

import grpc
import pyaudio
import webrtcvad

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

# 이 노드는 `python3 stt_node.py` 로 직접 돌린다.
#   ros2 run 은 colcon 이 shebang 을 /usr/bin/python3 로 고정해버려서
#   conda 환경(ros_env)의 pyaudio·webrtcvad 를 못 찾는다 -> ros2 run 은 애초에 못 쓴다.
#   토픽은 publisher·subscription 생성 위치에서 직접 확인할 수 있게 문자열로 적는다.
#   아래 nest_pb2 도 protoc 생성 코드라 절대 import 다.

import nest_pb2
import nest_pb2_grpc


CLOVA_HOST = 'clovaspeech-gw.ncloud.com:50051'
SECRET_ENV_NAME = 'CLOVA_SPEECH_SECRET_KEY'

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_MS = 30
FRAMES_PER_BUFFER = 480     # 30ms 분량. webrtcvad 는 10/20/30ms 만 받는다
AUDIO_FORMAT = pyaudio.paInt16
INPUT_DEVICE_INDEX = None   # hw:0,0 은 16kHz 를 못 연다. 기본 플러그인 경로를 써야 리샘플해준다

VAD_MODE = 2
PRE_ROLL_FRAMES = 10        # 300ms. VAD 가 반응한 시점엔 첫 음절이 이미 지나갔다
END_SILENCE_FRAMES = 33     # 990ms. 서버 epFlag 가 안 올 때의 백스톱
GUARD_TIME_SEC = 0.3        # tts_done 받고 이만큼 뒤에 연다. 방 잔향이 빠질 시간
QUEUE_WAIT_SEC = 0.1

PUBLISH_QUEUE_SIZE = 10
SUBSCRIPTION_QUEUE_SIZE = 10


CONFIG_JSON = json.dumps({'transcription': {'language': 'ko'}})


class SttStats:
    """발화 하나의 지연과 결과 품질을 한곳에서 관리한다."""

    def __init__(self):
        self.speech_started = time.perf_counter()
        self.first_partial = None
        self.endpoint_received = None
        self.published = None
        self.pieces = 0
        self.confidences = []

    def add_partial(self, confidence=None):
        self.pieces += 1

        if self.first_partial is None:
            self.first_partial = time.perf_counter()

        if isinstance(confidence, (int, float)):
            self.confidences.append(float(confidence))

    def mark_endpoint(self):
        if self.endpoint_received is None:
            self.endpoint_received = time.perf_counter()

    def mark_published(self):
        self.published = time.perf_counter()

    def summary(self):
        if self.first_partial is None:
            first_partial_ms = '없음'
        else:
            first_partial_ms = f'{(self.first_partial - self.speech_started) * 1000:.0f} ms'

        if self.endpoint_received is None or self.published is None:
            publish_ms = '없음'
        else:
            publish_ms = f'{(self.published - self.endpoint_received) * 1000:.1f} ms'

        if self.confidences:
            confidence_text = f'{sum(self.confidences) / len(self.confidences):.3f}'
        else:
            confidence_text = '없음'

        return (f'첫 부분결과 {first_partial_ms} | 끝점부터 발행 {publish_ms} | '
                f'조각 {self.pieces}개 | 평균 confidence {confidence_text}')


class ClovaSttNode(Node):
    def __init__(self):
        super().__init__('soomac_clova_stt_node')

        secret = os.environ.get(SECRET_ENV_NAME)
        if not secret:
            raise RuntimeError(f'{SECRET_ENV_NAME} 환경변수가 없어요. CLOVA Speech 비밀키를 넣고 다시 실행해 주세요.')

        self.metadata = (('authorization', f'Bearer {secret}'),)   # 키 이름은 반드시 소문자

        self.audio_queue = queue.Queue()
        self.pre_roll = deque(maxlen=PRE_ROLL_FRAMES)

        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.pre_roll_lock = threading.Lock()
        self.call_lock = threading.Lock()

        self.session = 0            # /stt_stop 때마다 ++. 늦게 온 응답을 락 없이 버리는 용도
        self.listening_enabled = False # 주문 흐름상 지금 손님 말을 들어도 되는지
        self.gate_open = False # TTS 중이거나 응답 대기 중이면 False
        self._gate_open_at = None
        self.active_call = None

        self.vad = webrtcvad.Vad(VAD_MODE)

        # 채널(TCP/TLS)은 노드 수명 동안 유지하고 recognize() 스트림만 발화 단위로 열고 닫는다.
        #   스트림 새로 여는 비용은 실측 40ms 라 싸다. 반면 스트림을 계속 열어두면
        #   TTS 가 말하는 동안 로봇 목소리가 서버로 흘러들어가 초당 과금 + 세션 오염이 된다.
        self.channel = grpc.secure_channel(CLOVA_HOST, grpc.ssl_channel_credentials())
        self.stub = nest_pb2_grpc.NestServiceStub(self.channel)

        self.pyaudio_instance = None
        self.microphone_stream = None

        self.question_pub = self.create_publisher(String, '/stt_question', PUBLISH_QUEUE_SIZE)

        self.create_subscription(String, '/tts_done', self.tts_done_callback, SUBSCRIPTION_QUEUE_SIZE)
        self.create_subscription(String, '/stt_stop', self.stt_stop_callback, SUBSCRIPTION_QUEUE_SIZE)
        self.create_subscription(
            Bool, '/stt/enable', self.stt_enable_callback,
            SUBSCRIPTION_QUEUE_SIZE)

        self._open_microphone()

        self.microphone_thread = threading.Thread(target=self._microphone_loop, name='clova-microphone', daemon=True)
        self.worker_thread = threading.Thread(target=self._worker_loop, name='clova-worker', daemon=True)

        self.microphone_thread.start()
        self.worker_thread.start()

        self.get_logger().info('클로바 STT 준비됐어요. 주문 대화 시작 신호를 기다립니다.')

    def _open_microphone(self):
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.microphone_stream = self.pyaudio_instance.open(
                format=AUDIO_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True,
                input_device_index=INPUT_DEVICE_INDEX, frames_per_buffer=FRAMES_PER_BUFFER)

            self.get_logger().info(f'마이크를 {SAMPLE_RATE}Hz, 1채널, 16비트로 열었어요.')

        except Exception as error:
            self.get_logger().error(f'마이크를 열다가 터졌어요: {error}\n{traceback.format_exc()}')
            self._close_audio()
            raise

    def _drain_audio_queue(self):
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _clear_pre_roll(self):
        with self.pre_roll_lock:
            self.pre_roll.clear()

    def _snapshot_and_clear_pre_roll(self):
        with self.pre_roll_lock:
            frames = list(self.pre_roll)
            self.pre_roll.clear()
        return frames

    def _cancel_active_call(self):
        with self.call_lock:
            call = self.active_call

        if call is None:
            return

        try:
            call.cancel()   # 응답 루프를 깨우는 유일한 수단이다
        except Exception as error:
            self.get_logger().warning(f'클로바 스트림 취소하다가 터졌어요: {error}\n{traceback.format_exc()}')

    def stt_enable_callback(self, message):
        enabled = bool(message.data)

        with self.state_lock:
            self.listening_enabled = enabled
            self.gate_open = False
            self._gate_open_at = None
            self.session += 1
            changed_session = self.session

        # 상태가 바뀐 순간 이전 발화와 늦게 온 서버 응답을 모두 버린다.
        self._drain_audio_queue()
        self._clear_pre_roll()
        self._cancel_active_call()

        if enabled:
            self.get_logger().info(
                f'STT 듣기를 허용했어요. TTS 종료 후 문을 엽니다. '
                f'세션은 {changed_session}번입니다.')
        else:
            self.get_logger().info(
                f'STT 듣기를 막았어요. TTS가 끝나도 열지 않습니다. '
                f'세션은 {changed_session}번입니다.')

    def stt_stop_callback(self, message):
        with self.state_lock:
            self.gate_open = False
            self._gate_open_at = None
            self.session += 1
            stopped_session = self.session

        # TTS 직전에 이미 잡혀 있던 사람 목소리도 다음 질문에 섞이면 안 된다.
        self._drain_audio_queue()
        self._clear_pre_roll()
        self._cancel_active_call()

        self.get_logger().info(f'STT 잠깐 닫았어요. 로봇 목소리는 서버로 안 보낼게요. 세션은 {stopped_session}번입니다.')

    def tts_done_callback(self, message):
        # 콜백에서 자면 ROS executor 전체가 멈춘다. 시각만 적어두고 마이크 스레드가 열게 한다.
        with self.state_lock:
            if self.listening_enabled:
                self.gate_open = False
                self._gate_open_at = time.monotonic() + GUARD_TIME_SEC
                should_reopen = True
            else:
                self.gate_open = False
                self._gate_open_at = None
                should_reopen = False

        if not should_reopen:
            self.get_logger().info('TTS는 끝났지만 로봇 작업 중이라 STT는 계속 닫아둡니다.')
            return

        self.get_logger().info('TTS가 끝났어요. 잔향이 빠진 뒤 다시 듣겠습니다.')

    def _update_gate_from_clock(self):
        with self.state_lock:
            if (not self.listening_enabled
                    or self._gate_open_at is None
                    or time.monotonic() < self._gate_open_at):
                return

            # 큐를 비운 뒤에 문을 열어야 가드타임 동안 들어온 잔향이 살아남지 않는다.
            self._drain_audio_queue()
            self._clear_pre_roll()
            self.gate_open = True
            self._gate_open_at = None

        self.get_logger().info('가드타임 끝났어요. 다시 듣고 있습니다.')

    def _is_gate_open(self):
        with self.state_lock:
            return self.listening_enabled and self.gate_open

    def _current_session(self):
        with self.state_lock:
            return self.session

    def _microphone_loop(self):
        try:
            while not self.stop_event.is_set():
                pcm = self.microphone_stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)

                self._update_gate_from_clock()

                # 닫혀 있을 때도 read 해야 드라이버 안에 로봇 목소리가 쌓이지 않는다.
                if not self._is_gate_open():
                    continue

                self.audio_queue.put(pcm)

        except Exception as error:
            if not self.stop_event.is_set():
                self.get_logger().error(f'마이크 스레드가 터졌어요: {error}\n{traceback.format_exc()}')
                self.stop_event.set()
                self._cancel_active_call()

    def _worker_loop(self):
        try:
            while not self.stop_event.is_set():
                try:
                    pcm = self.audio_queue.get(timeout=QUEUE_WAIT_SEC)
                except queue.Empty:
                    continue

                if not self._is_gate_open():
                    continue

                with self.pre_roll_lock:
                    self.pre_roll.append(pcm)

                try:
                    speech_detected = self.vad.is_speech(pcm, SAMPLE_RATE)
                except Exception as error:
                    self.get_logger().warning(f'VAD가 프레임을 못 읽었어요: {error}\n{traceback.format_exc()}')
                    continue

                if not speech_detected:
                    continue

                my_session = self._current_session()
                initial_frames = self._snapshot_and_clear_pre_roll()

                self.get_logger().info(f'말이 시작됐어요. 클로바 세션 {my_session}번을 엽니다.')

                self._recognize_utterance(my_session, initial_frames)

                # 한 발화의 프리롤이 다음 발화 앞에 다시 붙지 않도록 세션 뒤에도 비운다.
                self._clear_pre_roll()

        except Exception as error:
            if not self.stop_event.is_set():
                self.get_logger().error(f'STT 워커 스레드가 터졌어요: {error}\n{traceback.format_exc()}')
                self.stop_event.set()
                self._cancel_active_call()

    def _request_generator(self, my_session, initial_frames):
        # 이 generator 는 grpc 송신 스레드에서 돈다. 워커는 그동안 응답 루프에 막혀 있어서
        #   같은 audio_queue 를 동시에 꺼내가는 일은 없다.
        yield nest_pb2.NestRequest(type=nest_pb2.RequestType.CONFIG,
                                   config=nest_pb2.NestConfig(config=CONFIG_JSON))

        seq_id = 0
        silence_frames = 0

        for pcm in initial_frames:
            if self.stop_event.is_set() or my_session != self._current_session():
                return

            yield nest_pb2.NestRequest(type=nest_pb2.RequestType.DATA,
                                       data=nest_pb2.NestData(chunk=pcm, extra_contents=json.dumps({'seqId': seq_id, 'epFlag': False})))
            seq_id += 1

        while not self.stop_event.is_set():
            if my_session != self._current_session() or not self._is_gate_open():
                return

            try:
                pcm = self.audio_queue.get(timeout=QUEUE_WAIT_SEC)
            except queue.Empty:
                continue

            if my_session != self._current_session():
                return

            try:
                speech_detected = self.vad.is_speech(pcm, SAMPLE_RATE)
            except Exception as error:
                self.get_logger().warning(f'발화 중 VAD가 프레임을 못 읽었어요: {error}\n{traceback.format_exc()}')
                speech_detected = True

            silence_frames = 0 if speech_detected else silence_frames + 1
            local_endpoint = silence_frames >= END_SILENCE_FRAMES

            yield nest_pb2.NestRequest(type=nest_pb2.RequestType.DATA,
                                       data=nest_pb2.NestData(chunk=pcm, extra_contents=json.dumps({'seqId': seq_id, 'epFlag': local_endpoint})))
            seq_id += 1

            # 서버 끝점 검출이 늦거나 빠져도 1초 무음 뒤에는 요청 쪽을 닫아 준다.
            if local_endpoint:
                self._local_endpoint_sent = True
                self.get_logger().info('서버 끝점이 아직 안 와서 1초 무음으로 발화를 닫았어요.')
                return

    def _recognize_utterance(self, my_session, initial_frames):
        stats = SttStats()
        text_pieces = []
        endpoint_received = False
        call = None

        # 무음 백스톱으로 우리가 발화를 닫았는지. generator 가 세운다.
        #   서버가 epFlag 를 안 돌려줘도 인식된 글자를 버리지 않기 위해 필요하다.
        self._local_endpoint_sent = False

        try:
            call = self.stub.recognize(self._request_generator(my_session, initial_frames), metadata=self.metadata)

            with self.call_lock:
                self.active_call = call

            for resp in call:
                if self.stop_event.is_set() or my_session != self._current_session():
                    return

                try:
                    contents = json.loads(resp.contents)
                except Exception as error:
                    self.get_logger().warning(f'클로바 응답 JSON이 이상해요: {error}, 원문={resp.contents!r}\n{traceback.format_exc()}')
                    continue

                response_types = contents.get('responseType') or []
                transcription = contents.get('transcription') or {}

                if 'transcription' not in response_types and not transcription:
                    continue

                text = transcription.get('text') or ''
                confidence = transcription.get('confidence')
                ep_flag = bool(transcription.get('epFlag', False))

                # 서버는 전체 문장을 다시 주지 않는다. 조각으로 오니 이어붙여야 한다
                if text:
                    text_pieces.append(text)
                    stats.add_partial(confidence)
                    self.get_logger().info(f'들리는 중이에요: {text}')

                if ep_flag:
                    stats.mark_endpoint()
                    endpoint_received = True
                    break

            if self.stop_event.is_set() or my_session != self._current_session():
                return

            if not endpoint_received:
                # 서버가 epFlag 를 안 돌려주는 경우가 있는지 아직 실측 못 했다.
                #   우리가 무음 백스톱으로 닫았다면 인식된 글자는 살린다.
                #   그것도 아니면 진짜로 끊긴 것이니 버린다.
                if not self._local_endpoint_sent:
                    self.get_logger().warning('클로바 스트림이 끝났는데 epFlag도 없고 우리가 닫은 것도 아니라 버립니다.')
                    return

                self.get_logger().warning('서버 epFlag 없이 무음 백스톱으로 닫혔어요. 그래도 인식된 건 발행합니다.')
                stats.mark_endpoint()

            final_text = ''.join(text_pieces).strip()

            if not final_text:
                self.get_logger().info('발화 끝은 받았는데 알아들은 글자가 없어요.')
                return

            # 한 발화를 보낸 뒤에는 LLM·TTS 응답이 끝날 때까지 다음 발화를 받지 않는다.
            with self.state_lock:
                self.gate_open = False
                self._gate_open_at = None
            self._drain_audio_queue()
            self._clear_pre_roll()

            self.question_pub.publish(String(data=final_text))
            stats.mark_published()

            self.get_logger().info(f'질문으로 보냈어요: {final_text}')
            self.get_logger().info(stats.summary())

        except grpc.RpcError as error:
            if (self.stop_event.is_set() or my_session != self._current_session()
                    or error.code() == grpc.StatusCode.CANCELLED):
                self.get_logger().info(f'클로바 세션 {my_session}번을 바로 취소했어요.')
                return

            self.get_logger().error(f'클로바 gRPC가 터졌어요: {error.code()} / {error.details()}\n{traceback.format_exc()}')

        except Exception as error:
            self.get_logger().error(f'음성 인식하다가 터졌어요: {error}\n{traceback.format_exc()}')

        finally:
            if call is not None:
                try:
                    call.cancel()   # generator 도 여기서 같이 닫힌다. 안 닫으면 과금이 계속된다
                except Exception as error:
                    self.get_logger().warning(f'끝난 클로바 스트림 정리하다가 터졌어요: {error}\n{traceback.format_exc()}')

            with self.call_lock:
                if self.active_call is call:
                    self.active_call = None

    def _close_audio(self):
        if self.microphone_stream is not None:
            try:
                if self.microphone_stream.is_active():
                    self.microphone_stream.stop_stream()
                self.microphone_stream.close()
            except Exception as error:
                self.get_logger().warning(f'마이크 스트림 정리하다가 터졌어요: {error}\n{traceback.format_exc()}')
            finally:
                self.microphone_stream = None

        if self.pyaudio_instance is not None:
            try:
                self.pyaudio_instance.terminate()
            except Exception as error:
                self.get_logger().warning(f'PyAudio 정리하다가 터졌어요: {error}\n{traceback.format_exc()}')
            finally:
                self.pyaudio_instance = None

    def destroy_node(self):
        # 순서가 중요하다. 플래그 -> 스트림 취소 -> 스레드 join -> 오디오/채널 정리
        self.stop_event.set()

        with self.state_lock:
            self.gate_open = False
            self._gate_open_at = None
            self.session += 1

        self._cancel_active_call()

        try:
            if hasattr(self, 'microphone_thread'):
                self.microphone_thread.join(timeout=2.0)
                if self.microphone_thread.is_alive():
                    self.get_logger().warning('마이크 스레드가 2초 안에 안 끝났어요.')
        except Exception as error:
            self.get_logger().warning(f'마이크 스레드 기다리다가 터졌어요: {error}\n{traceback.format_exc()}')

        try:
            if hasattr(self, 'worker_thread'):
                self.worker_thread.join(timeout=2.0)
                if self.worker_thread.is_alive():
                    self.get_logger().warning('STT 워커 스레드가 2초 안에 안 끝났어요.')
        except Exception as error:
            self.get_logger().warning(f'STT 워커 기다리다가 터졌어요: {error}\n{traceback.format_exc()}')

        self._close_audio()

        try:
            if self.channel is not None:
                self.channel.close()
        except Exception as error:
            self.get_logger().warning(f'gRPC 채널 정리하다가 터졌어요: {error}\n{traceback.format_exc()}')

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = ClovaSttNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error(f'STT 노드가 터졌어요: {error}\n{traceback.format_exc()}')
        else:
            print(f'STT 노드를 시작하지 못했어요: {error}\n{traceback.format_exc()}')
        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
