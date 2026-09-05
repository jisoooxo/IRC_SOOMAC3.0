#!/home/roma/miniconda3/envs/gemma4_env/bin/python

import math
import queue
import shutil
import subprocess
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

# 이 노드는 conda 환경을 지키려고 파일로 직접 실행한다.
#   토픽은 publisher·subscription 생성 위치에서 직접 확인할 수 있게 문자열로 적는다.


# 모델 자체는 캐시 기반 스트리밍 추론을 지원하지만 현재 노드는 그 모드를 쓰지 않는다.
#   Silero VAD 로 발화 끝을 찾은 뒤 완성된 음성 전체를 generate 에 한 번 넣는 구조이다.
MODEL_DIR = '/home/roma/models/audio/stt/nemotron-3.5-asr-streaming-0.6b'
# auto 는 CUDA 를 먼저 고르고 없으면 CPU 를 쓴다. CPU+bfloat16 경로는 아직 실측하지 않았다.
COMPUTE_DEVICE = 'auto'
# 모델이 한국어 표기와 디코딩 규칙을 선택할 때 사용하는 언어 코드이다.
LANGUAGE = 'ko-KR'
# 생성되는 글자 수의 안전 상한이다. 음성 청크 길이나 스트리밍 지연을 정하는 값은 아니다.
#   긴 발화가 실제로 잘리는지 확인하기 전에는 임의로 줄이지 않는다.
MAX_NEW_TOKENS = 1024
# 모델 설정의 기본값은 float32 지만 가중치가 2,550MiB 로 두 배가 되고 얻는 게 없다.
#   2026-07-31 실측에서 bfloat16 가중치 1,217MiB / 추론 피크 1,290MiB 로 정확도가 같았다.
MODEL_DTYPE = torch.bfloat16

# PulseAudio 가 USB 마이크를 잡고 있으므로 'default' 로 두는 게 맞다.
#   plughw:CARD=... 처럼 하드웨어 경로를 직접 지정하면 "Device or resource busy" 로 못 연다.
#   ⚠ USB 마이크를 안 꽂은 상태에서는 기본 소스가 S/PDIF 출력의 모니터로 떨어져
#   전부 0인 무음이 잡힌다. 레벨이 -120데시벨로 고정되면 마이크가 안 꽂힌 것이다.
AUDIO_DEVICE = 'default'
# Nemotron processor 와 Silero VAD 모두 16킬로헤르츠 단일 채널 입력을 기준으로 한다.
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
# 이 값은 Nemotron 스트리밍 청크가 아니라 Silero VAD 입력 프레임이다.
#   16킬로헤르츠에서 32밀리초가 정확히 512샘플이므로 다른 값으로 바꾸지 않는다.
CHUNK_MS = 32
# 청크를 음성으로 볼 Silero 확률 기준이다. 높이면 잡음은 줄지만 작은 목소리를 놓칠 수 있다.
SPEECH_PROBABILITY = 0.5
# 참조 구현 값 그대로 0.8 이다. 줄이면 응답이 빨라지지만 문장 사이에서 발화가 쪼개진다.
#   실측: 한국어 두 문장 사이 무음이 0.704초라 0.7 이하면 두 발화로 갈라져 LLM 에 따로 들어간다.
#   0.8 이 지금 확인된 최소값이다. 더 줄이려면 그 전에 실제 주문 발화로 무음 길이를 다시 재라.
SILENCE_SECONDS = 0.8
# 실제 음성 청크의 합이 이보다 짧으면 잡음으로 보고 버린다.
#   '네', '응', '끝'도 짧을 수 있으므로 값을 바꾸기 전에 실제 음성 시간을 먼저 확인한다.
MIN_SPEECH_SECONDS = 0.35
# 한 발화가 끝나지 않을 때 메모리와 추론 시간을 제한하는 안전 상한이다.
MAX_SPEECH_SECONDS = 30.0
# 말 시작 직전 소리를 함께 보관하여 첫 음절이 VAD 경계에서 잘리지 않게 한다.
PRE_ROLL_MS = 300
# 마이크 연결 진단용 로그 주기이다. 0이면 레벨 로그와 RMS 계산을 모두 끈다.
LEVEL_LOG_SECONDS = 1.0
# TTS 종료 직후 남은 스피커 잔향을 다시 받아쓰지 않도록 기다리는 시간이다.
GUARD_TIME_SEC = 0.3
QUEUE_WAIT_SEC = 0.1
SILERO_REPO_DIR = '/home/roma/.cache/torch/hub/snakers4_silero-vad_master'

PUBLISH_QUEUE_SIZE = 10
SUBSCRIPTION_QUEUE_SIZE = 10


ARECORD_WAIT_SEC = 2.0
THREAD_JOIN_TIMEOUT_SEC = 2.0
DBFS_FLOOR = -120.0
PCM_FULL_SCALE = float(1 << (SAMPLE_WIDTH_BYTES * 8 - 1))

FRAMES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = FRAMES_PER_CHUNK * SAMPLE_WIDTH_BYTES * CHANNELS
SILENCE_CHUNKS = max(1, math.ceil(SILENCE_SECONDS * 1000 / CHUNK_MS))
PRE_ROLL_CHUNKS = max(1, math.ceil(PRE_ROLL_MS / CHUNK_MS))
MAX_SPEECH_BYTES = int(
    MAX_SPEECH_SECONDS * SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS)
LEVEL_LOG_CHUNKS = (
    max(1, math.ceil(LEVEL_LOG_SECONDS * 1000 / CHUNK_MS))
    if LEVEL_LOG_SECONDS > 0 else 0)


class NemotronSttStats:
    """발화 하나에서 실제로 잴 수 있는 길이와 처리 지연을 관리한다."""

    def __init__(self, audio_seconds, utterance_ended_at):
        self.audio_seconds = audio_seconds
        self.utterance_ended_at = utterance_ended_at
        self.inference_started_at = None
        self.inference_finished_at = None
        self.published_at = None

    def mark_inference_started(self):
        self.inference_started_at = time.perf_counter()

    def mark_inference_finished(self):
        self.inference_finished_at = time.perf_counter()

    def mark_published(self):
        self.published_at = time.perf_counter()

    def summary(self):
        if self.inference_started_at is None or self.inference_finished_at is None:
            inference_seconds = None
            realtime_factor = None
        else:
            inference_seconds = self.inference_finished_at - self.inference_started_at
            realtime_factor = self.audio_seconds / max(inference_seconds, 1e-9)

        if self.published_at is None:
            publish_delay_ms = None
        else:
            publish_delay_ms = (self.published_at - self.utterance_ended_at) * 1000

        inference_text = '없음' if inference_seconds is None else f'{inference_seconds:.3f}초'
        realtime_text = '없음' if realtime_factor is None else f'{realtime_factor:.1f}배'
        publish_text = '없음' if publish_delay_ms is None else f'{publish_delay_ms:.1f}밀리초'

        return (f'발화 {self.audio_seconds:.2f}초 | 추론 {inference_text} | '
                f'실시간 배율 {realtime_text} | 발화 끝부터 발행 {publish_text}')


class NemotronSttNode(Node):
    def __init__(self):
        super().__init__('soomac_nemotron_stt_node')

        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        # 추론은 마이크 스레드만 하지만 ROS 콜백도 상태를 초기화하므로 LSTM 접근을 직렬화한다.
        self.silero_vad_lock = threading.Lock()

        self.session = 0
        self.listening_enabled = False # 주문 흐름상 지금 손님 말을 들어도 되는지
        self.gate_open = False # TTS 중이거나 응답 대기 중이면 False
        self._gate_open_at = None

        self.utterance_queue = queue.Queue()
        self.pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)
        self.utterance_pcm_parts = []
        self.utterance_byte_count = 0
        self.speaking = False
        self.silent_chunk_count = 0
        self.voice_chunk_count = 0
        self.capture_session = None

        self.processor = None
        self.model = None
        self.silero_vad_model = None
        self.arecord_process = None

        self._validate_constants()
        self._load_silero_vad()
        self._load_model()

        self.question_pub = self.create_publisher(
            String, '/stt_question', PUBLISH_QUEUE_SIZE)
        self.create_subscription(
            String, '/tts_done', self.tts_done_callback,
            SUBSCRIPTION_QUEUE_SIZE)
        self.create_subscription(
            String, '/stt_stop', self.stt_stop_callback,
            SUBSCRIPTION_QUEUE_SIZE)
        self.create_subscription(
            Bool, '/stt/enable', self.stt_enable_callback,
            SUBSCRIPTION_QUEUE_SIZE)

        self._open_arecord()

        self.microphone_thread = threading.Thread(
            target=self._microphone_loop,
            name='nemotron-microphone',
            daemon=True)
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name='nemotron-worker',
            daemon=True)

        self.microphone_thread.start()
        self.worker_thread.start()

        self.get_logger().info(
            '네모트론 STT 준비됐어요. /stt/enable 신호를 기다립니다.')

    def _validate_constants(self):
        if CHUNK_MS <= 0:
            raise ValueError('마이크 청크 길이는 1밀리초 이상이어야 해요.')
        if SAMPLE_RATE * CHUNK_MS % 1000 != 0:
            raise ValueError('마이크 청크 길이는 샘플 수가 정수가 되도록 정해야 해요.')
        if FRAMES_PER_CHUNK != 512:
            raise ValueError('silero-vad 입력은 청크마다 정확히 512샘플이어야 해요.')
        if SAMPLE_WIDTH_BYTES != 2 or CHANNELS != 1:
            raise ValueError('현재 변환과 레벨 계산은 16비트 단일 채널 입력만 지원해요.')
        if not 0.0 <= SPEECH_PROBABILITY <= 1.0:
            raise ValueError('silero-vad 음성 확률 기준은 0과 1 사이여야 해요.')
        if SILENCE_SECONDS <= 0 or MAX_SPEECH_SECONDS <= 0:
            raise ValueError('무음 길이와 최대 발화 길이는 0보다 커야 해요.')
        if MIN_SPEECH_SECONDS < 0:
            raise ValueError('최소 발화 길이는 0 이상이어야 해요.')

    def _choose_compute_device(self):
        if COMPUTE_DEVICE == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if COMPUTE_DEVICE == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('그래픽 연산 장치를 지정했지만 CUDA를 사용할 수 없어요.')
        if COMPUTE_DEVICE not in ('cuda', 'cpu'):
            raise ValueError(f'지원하지 않는 연산 장치예요: {COMPUTE_DEVICE}')
        return COMPUTE_DEVICE

    def _load_silero_vad(self):
        silero_repo_path = Path(SILERO_REPO_DIR).expanduser().resolve()
        if not silero_repo_path.is_dir():
            raise FileNotFoundError(
                f'silero-vad 저장소 디렉터리가 없어요: {silero_repo_path}')

        loading_started_at = time.perf_counter()
        self.get_logger().info(
            f'silero-vad 모델을 CPU로 불러옵니다. 경로 {silero_repo_path}입니다.')

        try:
            silero_vad_model, _ = torch.hub.load(
                str(silero_repo_path),
                'silero_vad',
                source='local',
                onnx=False,
                trust_repo=True)
            silero_vad_model.to('cpu')
            silero_vad_model.eval()
            silero_vad_model.reset_states()
            self.silero_vad_model = silero_vad_model

            loading_seconds = time.perf_counter() - loading_started_at
            self.get_logger().info(
                f'silero-vad 모델을 {loading_seconds:.2f}초 만에 불러왔어요. '
                f'장치 CPU, 프레임 {FRAMES_PER_CHUNK}샘플입니다.')

        except Exception as error:
            self.get_logger().error(
                f'silero-vad 모델을 불러오다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}')
            raise

    def _load_model(self):
        model_path = Path(MODEL_DIR).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f'모델 디렉터리가 없어요: {model_path}')

        compute_device = self._choose_compute_device()
        loading_started_at = time.perf_counter()
        self.get_logger().info(
            f'네모트론 모델을 불러옵니다. 경로 {model_path}, 연산 장치 {compute_device}입니다.')

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path, local_files_only=True)
            processor_sample_rate = self.processor.feature_extractor.sampling_rate
            if processor_sample_rate != SAMPLE_RATE:
                raise RuntimeError(
                    f'모델 입력은 {processor_sample_rate}헤르츠인데 '
                    f'마이크 설정은 {SAMPLE_RATE}헤르츠예요.')

            # dtype 을 안 주면 config 의 float32 가 그대로 먹는다.
            #   _recognize_utterance 가 입력을 model.dtype 으로 맞추므로
            #   여기만 바꾸면 전처리부터 추론까지 전부 따라온다.
            self.model = AutoModelForRNNT.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=MODEL_DTYPE,
                device_map=compute_device)
            self.model.eval()

            loading_seconds = time.perf_counter() - loading_started_at
            model_device = self.model.device
            if model_device.type == 'cuda':
                allocated_vram_gib = torch.cuda.memory_allocated(model_device) / (1024 ** 3)
                reserved_vram_gib = torch.cuda.memory_reserved(model_device) / (1024 ** 3)
                vram_text = (f'할당 {allocated_vram_gib:.2f}기가바이트, '
                             f'예약 {reserved_vram_gib:.2f}기가바이트')
            else:
                vram_text = '사용 안 함'

            self.get_logger().info(
                f'네모트론 모델을 {loading_seconds:.2f}초 만에 불러왔어요. '
                f'자료형 {self.model.dtype}, 장치 {model_device}, 그래픽 메모리 {vram_text}입니다.')

        except Exception as error:
            self.get_logger().error(
                f'네모트론 모델을 불러오다가 터졌어요: {error}\n{traceback.format_exc()}')
            raise

    def _open_arecord(self):
        arecord_path = shutil.which('arecord')
        if arecord_path is None:
            raise RuntimeError('arecord를 찾지 못했어요. alsa-utils 설치를 확인해 주세요.')

        command = [
            arecord_path,
            '--quiet',
            '-D',
            AUDIO_DEVICE,
            '-t',
            'raw',
            '-f',
            'S16_LE',
            '-c',
            str(CHANNELS),
            '-r',
            str(SAMPLE_RATE),
        ]

        try:
            self.arecord_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=CHUNK_BYTES * 4)

            if (self.arecord_process.stdout is None
                    or self.arecord_process.stderr is None):
                raise RuntimeError('arecord 입출력 통로를 열지 못했어요.')

            self.get_logger().info(
                f'마이크를 {AUDIO_DEVICE} 장치에서 {SAMPLE_RATE}헤르츠, '
                f'{CHANNELS}채널, {SAMPLE_WIDTH_BYTES * 8}비트로 열었어요.')
            self.get_logger().info(
                f'음성 확률 기준은 {SPEECH_PROBABILITY:.2f}이고 '
                f'{SILENCE_SECONDS:.1f}초 조용하면 발화를 닫습니다.')

        except Exception as error:
            self.get_logger().error(
                f'마이크를 열다가 터졌어요: {error}\n{traceback.format_exc()}')
            self._stop_arecord()
            raise

    def _drain_utterance_queue(self):
        while True:
            try:
                self.utterance_queue.get_nowait()
            except queue.Empty:
                break

    def _clear_capture_locked(self):
        # LSTM 상태도 함께 지워야 이전 발화와 닫힌 동안의 소리가 다음 판정에 새지 않는다.
        silero_vad_model = self.silero_vad_model
        if silero_vad_model is not None:
            with self.silero_vad_lock:
                silero_vad_model.reset_states()

        self.pre_roll.clear()
        self.utterance_pcm_parts = []
        self.utterance_byte_count = 0
        self.speaking = False
        self.silent_chunk_count = 0
        self.voice_chunk_count = 0
        self.capture_session = None

    def stt_enable_callback(self, message):
        enabled = bool(message.data)

        with self.state_lock:
            # 같은 상태가 중복으로 들어오면 현재 발화와 세션을 그대로 유지한다.
            if enabled == self.listening_enabled:
                self.get_logger().info(
                    f'STT 상태가 이미 {"열기 대기" if enabled else "닫기"}입니다.')
                return

            self.listening_enabled = enabled
            self.gate_open = False
            self._gate_open_at = None
            self.session += 1
            changed_session = self.session

            # 세션을 바꾼 잠금 안에서 이전 발화와 대기 중인 추론을 모두 버린다.
            self._drain_utterance_queue()
            self._clear_capture_locked()

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

            # 세션을 바꾼 잠금 안에서 모두 비워야 오래된 청크가 닫힌 문 뒤로 다시 들어오지 않는다.
            self._drain_utterance_queue()
            self._clear_capture_locked()

        self.get_logger().info(
            f'STT 잠깐 닫았어요. 로봇 목소리는 인식하지 않을게요. '
            f'세션은 {stopped_session}번입니다.')

    def tts_done_callback(self, message):
        # 콜백에서 자면 ROS 실행기 전체가 멈추므로 시각만 적고 마이크 스레드가 문을 연다.
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

    def _update_gate_from_clock_locked(self):
        if (not self.listening_enabled or self._gate_open_at is None
                or time.monotonic() < self._gate_open_at):
            return False

        # 문을 열기 직전에 모두 비워야 가드 시간 동안 들어온 방 잔향이 살아남지 않는다.
        self._drain_utterance_queue()
        self._clear_capture_locked()
        self.gate_open = True
        self._gate_open_at = None
        return True

    def _is_gate_open(self):
        with self.state_lock:
            return self.listening_enabled and self.gate_open

    def _current_session(self):
        with self.state_lock:
            return self.session

    @staticmethod
    def _rms_dbfs(pcm):
        samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32)
        if samples.size == 0:
            return DBFS_FLOOR

        rms = float(np.sqrt(np.mean(np.square(samples)))) / PCM_FULL_SCALE
        return 20.0 * math.log10(max(rms, 1e-6))

    def _speech_probability(self, pcm):
        if len(pcm) != CHUNK_BYTES:
            raise ValueError(
                f'silero-vad 입력은 {CHUNK_BYTES}바이트여야 하는데 {len(pcm)}바이트예요.')

        silero_vad_model = self.silero_vad_model
        if silero_vad_model is None:
            raise RuntimeError('silero-vad 모델 자원이 이미 정리됐어요.')

        frame_float32 = (
            np.frombuffer(pcm, dtype='<i2').astype(np.float32) / PCM_FULL_SCALE)
        if frame_float32.size != FRAMES_PER_CHUNK:
            raise ValueError(
                f'silero-vad 입력은 {FRAMES_PER_CHUNK}샘플이어야 하는데 '
                f'{frame_float32.size}샘플이에요.')

        # CPU 추론을 상태 잠금 밖에서 끝내야 ROS 콜백과 발화 큐 처리를 막지 않는다.
        with self.silero_vad_lock, torch.inference_mode():
            return silero_vad_model(
                torch.from_numpy(frame_float32), SAMPLE_RATE).item()

    def _microphone_loop(self):
        chunk_number = 0

        try:
            while not self.stop_event.is_set():
                process = self.arecord_process
                if process is None or process.stdout is None or process.stderr is None:
                    raise RuntimeError('arecord 입출력 통로가 닫혔어요.')

                pcm = process.stdout.read(CHUNK_BYTES)
                if len(pcm) != CHUNK_BYTES:
                    error_text = process.stderr.read().decode(
                        'utf-8', errors='replace').strip()
                    raise RuntimeError(
                        f'마이크 입력이 중단됐어요: {error_text or "arecord가 종료됐어요."}')

                chunk_number += 1
                should_log_level = (
                    LEVEL_LOG_CHUNKS > 0
                    and chunk_number % LEVEL_LOG_CHUNKS == 0)
                level_dbfs = (
                    self._rms_dbfs(pcm)
                    if should_log_level else None)

                with self.state_lock:
                    gate_reopened = self._update_gate_from_clock_locked()
                    gate_open = self.gate_open

                if gate_reopened:
                    self.get_logger().info('가드 시간이 끝났어요. 다시 듣고 있습니다.')

                # 닫혀 있을 때도 위의 read는 계속하되 로봇 목소리를 LSTM 상태에는 넣지 않는다.
                if not gate_open:
                    continue

                speech_probability = self._speech_probability(pcm)
                is_voice = speech_probability >= SPEECH_PROBABILITY

                with self.state_lock:
                    # 확률 계산 중 문이 닫혔다면 이 청크가 새 세션에 들어가지 않게 버린다.
                    if not self.gate_open:
                        continue

                    if should_log_level:
                        level_log_text = (
                            f'마이크 레벨 {level_dbfs:.1f}데시벨 | '
                            f'음성확률 {speech_probability:.2f} | '
                            f'{"음성" if is_voice else "무음"}')
                    else:
                        level_log_text = None

                    speech_started = self._capture_chunk_locked(pcm, is_voice)

                if level_log_text is not None:
                    self.get_logger().info(level_log_text)
                if speech_started:
                    self.get_logger().info(
                        f'말이 시작됐어요. 네모트론 세션 {speech_started}번에 담습니다.')

        except Exception as error:
            if not self.stop_event.is_set():
                self.get_logger().error(
                    f'마이크 스레드가 터졌어요: {error}\n{traceback.format_exc()}')
                self.stop_event.set()
                rclpy.try_shutdown()

    def _capture_chunk_locked(self, pcm, is_voice):
        if not self.speaking:
            self.pre_roll.append(pcm)
            if not is_voice:
                return None

            self.speaking = True
            self.silent_chunk_count = 0
            self.voice_chunk_count = 1
            self.utterance_pcm_parts = list(self.pre_roll)
            self.utterance_byte_count = sum(
                len(pcm_part) for pcm_part in self.utterance_pcm_parts)
            self.pre_roll.clear()
            self.capture_session = self.session
            return self.capture_session

        self.utterance_pcm_parts.append(pcm)
        self.utterance_byte_count += len(pcm)
        self.silent_chunk_count = (
            0 if is_voice else self.silent_chunk_count + 1)
        if is_voice:
            self.voice_chunk_count += 1

        reached_silence = self.silent_chunk_count >= SILENCE_CHUNKS
        reached_limit = self.utterance_byte_count >= MAX_SPEECH_BYTES
        if not reached_silence and not reached_limit:
            return None

        utterance_pcm = b''.join(self.utterance_pcm_parts)
        captured_session = self.capture_session
        captured_seconds = len(utterance_pcm) / (
            SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS)
        voice_seconds = self.voice_chunk_count * CHUNK_MS / 1000
        utterance_ended_at = time.perf_counter()
        ending_reason = '무음' if reached_silence else '최대 길이'

        self.get_logger().info(
            f'말이 끝났어요. 전체 {captured_seconds:.2f}초, '
            f'음성 {voice_seconds:.2f}초, 이유 {ending_reason}입니다.')

        if voice_seconds < MIN_SPEECH_SECONDS:
            self._clear_capture_locked()
            self.get_logger().info(
                f'실제 음성이 {voice_seconds:.2f}초라 너무 짧아서 건너뜁니다.')
            return None

        # 잠금 안에서 넣어야 직후의 중지 콜백이 이 발화까지 확실하게 찾아 버릴 수 있다.
        self.utterance_queue.put_nowait((
            utterance_pcm,
            captured_session,
            utterance_ended_at,
            captured_seconds))
        self._clear_capture_locked()
        return None

    def _worker_loop(self):
        try:
            while not self.stop_event.is_set():
                try:
                    (utterance_pcm,
                     captured_session,
                     utterance_ended_at,
                     captured_seconds) = self.utterance_queue.get(
                        timeout=QUEUE_WAIT_SEC)
                except queue.Empty:
                    continue

                if (captured_session != self._current_session()
                        or not self._is_gate_open()):
                    continue

                self._recognize_utterance(
                    utterance_pcm,
                    captured_session,
                    utterance_ended_at,
                    captured_seconds)

        except Exception as error:
            if not self.stop_event.is_set():
                self.get_logger().error(
                    f'STT 워커 스레드가 터졌어요: {error}\n{traceback.format_exc()}')
                self.stop_event.set()
                rclpy.try_shutdown()

    def _recognize_utterance(self, utterance_pcm, captured_session,
                             utterance_ended_at, captured_seconds):
        stats = NemotronSttStats(captured_seconds, utterance_ended_at)

        try:
            # 취소할 수 없는 그래픽 연산을 시작하기 직전에도 세션을 봐야 불필요한 추론을 줄인다.
            if (self.stop_event.is_set()
                    or captured_session != self._current_session()
                    or not self._is_gate_open()):
                return

            processor = self.processor
            model = self.model
            if processor is None or model is None:
                raise RuntimeError('네모트론 모델 자원이 이미 정리됐어요.')

            audio = np.frombuffer(
                utterance_pcm, dtype='<i2').astype(np.float32) / PCM_FULL_SCALE

            stats.mark_inference_started()
            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                language=LANGUAGE,
                return_tensors='pt')
            inputs = inputs.to(model.device, dtype=model.dtype)

            # 다음 단계에서는 이 자리만 모델 카드의 스트리밍 입력 생성기로 바꿀 수 있다.
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    return_dict_in_generate=True)

            decoded = processor.decode(
                output.sequences, skip_special_tokens=True)
            final_text = decoded[0] if isinstance(decoded, list) else decoded
            final_text = ' '.join(final_text.split())
            stats.mark_inference_finished()

            if not final_text:
                self.get_logger().info('말은 끝났는데 알아들은 글자가 없어요.')
                self.get_logger().info(stats.summary())
                return

            # 확인과 발행 사이에 중지 콜백이 끼면 늦은 결과가 나가므로 같은 잠금 안에서 끝낸다.
            with self.state_lock:
                if (self.stop_event.is_set()
                        or captured_session != self.session
                        or not self.listening_enabled
                        or not self.gate_open):
                    result_is_stale = True
                else:
                    self.question_pub.publish(String(data=final_text))
                    stats.mark_published()
                    # LLM·TTS 응답이 끝날 때까지 다음 발화는 받지 않는다.
                    self.gate_open = False
                    self._gate_open_at = None
                    result_is_stale = False

            # 그래픽 연산은 중간 취소가 안 되므로 끝난 뒤 세션을 다시 확인해 로봇 음성 결과를 버린다.
            if result_is_stale:
                self.get_logger().info(
                    f'네모트론 세션 {captured_session}번 결과는 문이 닫혀서 버렸어요.')
                return

            self.get_logger().info(f'질문으로 보냈어요: {final_text}')
            self.get_logger().info(stats.summary())

        except Exception as error:
            self.get_logger().error(
                f'음성 인식하다가 터졌어요: {error}\n{traceback.format_exc()}')

    def _stop_arecord(self):
        process = self.arecord_process
        if process is None or process.poll() is not None:
            return

        try:
            process.terminate()
            try:
                process.wait(timeout=ARECORD_WAIT_SEC)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=ARECORD_WAIT_SEC)
        except Exception as error:
            self.get_logger().warning(
                f'arecord를 멈추다가 터졌어요: {error}\n{traceback.format_exc()}')

    def _close_arecord_pipes(self):
        process = self.arecord_process
        if process is None:
            return

        for pipe_name in ('stdout', 'stderr'):
            pipe = getattr(process, pipe_name, None)
            if pipe is None:
                continue
            try:
                pipe.close()
            except Exception as error:
                self.get_logger().warning(
                    f'arecord {pipe_name} 통로를 닫다가 터졌어요: '
                    f'{error}\n{traceback.format_exc()}')

        self.arecord_process = None

    def _release_model(self):
        self.processor = None
        self.model = None
        self.silero_vad_model = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as error:
                self.get_logger().warning(
                    f'그래픽 메모리를 정리하다가 터졌어요: '
                    f'{error}\n{traceback.format_exc()}')

    def destroy_node(self):
        # 순서가 중요하다. 플래그 -> arecord 종료 -> 스레드 대기 -> 자원 정리 순서다.
        self.stop_event.set()

        with self.state_lock:
            self.gate_open = False
            self._gate_open_at = None
            self.session += 1
            self._drain_utterance_queue()
            self._clear_capture_locked()

        self._stop_arecord()

        microphone_thread_stopped = True
        try:
            if hasattr(self, 'microphone_thread'):
                self.microphone_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
                microphone_thread_stopped = not self.microphone_thread.is_alive()
                if not microphone_thread_stopped:
                    self.get_logger().warning(
                        f'마이크 스레드가 {THREAD_JOIN_TIMEOUT_SEC:.0f}초 안에 안 끝났어요.')
        except Exception as error:
            microphone_thread_stopped = False
            self.get_logger().warning(
                f'마이크 스레드를 기다리다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}')

        worker_thread_stopped = True
        try:
            if hasattr(self, 'worker_thread'):
                self.worker_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
                worker_thread_stopped = not self.worker_thread.is_alive()
                if not worker_thread_stopped:
                    self.get_logger().warning(
                        f'STT 워커 스레드가 {THREAD_JOIN_TIMEOUT_SEC:.0f}초 안에 안 끝났어요.')
        except Exception as error:
            worker_thread_stopped = False
            self.get_logger().warning(
                f'STT 워커를 기다리다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}')

        self._close_arecord_pipes()
        self._drain_utterance_queue()

        # 워커가 아직 그래픽 연산 중이면 참조를 지우지 않아 종료 경로의 속성 접근 실패를 막는다.
        if worker_thread_stopped:
            self._release_model()

        if not microphone_thread_stopped:
            self.get_logger().warning('마이크 스레드는 종료 중인 프로세스와 함께 정리됩니다.')

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = NemotronSttNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error(
                f'STT 노드가 터졌어요: {error}\n{traceback.format_exc()}')
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
