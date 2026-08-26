#!/home/roma/miniconda3/envs/cosyvoice3/bin/python 

# 가상환경 박아두기

from __future__ import annotations
import os
import sys
import time # 시간 체크 용

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly
import threading, queue # 청크 공백 사이에 순간적으로 끊기는 것을 방지하기 위해 병렬 스레드로 돌려야 함
import re


import traceback
# 에러가 어디서 났는지 알려주는 호출 경로 ㅇㅇ 이거 붙이면 왜 에러 터졌는지 알 수 있음.
#   쌓이지는 않는다. format_exc() 는 터진 그 순간의 스택을 문자열로 떠서 돌려주고 끝.
#   except 안에서만 부르니 정상 동작에는 영향 0 이고, 로그 몇 줄 길어질 뿐이다.

from contextlib import contextmanager

COSYVOICE_WEIGHT_ROOT = '/home/roma/CosyVoice_new'

sys.path.insert(0, COSYVOICE_WEIGHT_ROOT)
sys.path.insert(0, os.path.join(COSYVOICE_WEIGHT_ROOT, 'third_party/Matcha-TTS'))


import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# `python3 tts_node.py` 로 직접 돌리고, ros2 run 은 colcon shebang 문제로 못 쓴다.
#   토픽은 publisher·subscription 생성 위치에서 직접 확인할 수 있게 문자열로 적는다.

from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = '/home/roma/models/audio/tts/Fun-CosyVoice3-0.5B-2512'
REFERENCE_WAV = '/home/roma/CosyVoice_new/asset/산사10.wav'
OUTPUT_DIR = '/home/roma/ros2_ws/src/soomac_ai/src/tts_out'
PROMPT_WAV = os.path.join(OUTPUT_DIR, 'prompt_5p5s.wav')
LAST_WAV = os.path.join(OUTPUT_DIR, 'last.wav')
SAMPLE_RATE = 24000
PROMPT_TEXT = (
    'You are a helpful assistant.<|endofprompt|>'
    '반갑습니다. 산사의 아침 공기가 참으로 맑고 고요하지요?'
)

OUT_SR = 48000 # resampleing Hz
# 'sysdefault' 는 card 0(ALC897 아날로그 = 헤드폰)에 고정이다.
#   None 으로 두면 PortAudio 가 켤 때의 상태에 따라 조용히 HDMI 로 폴백한다.
#   sysdefault 는 아날로그가 안 잡히면 에러를 내고 죽는다. 무음보다 그게 낫다.
#   HDMI 로 내보내려면 TTS_DEVICE='hw:1,3' 처럼 덮어쓸 것
OUT_DEVICE = os.environ.get('TTS_DEVICE') or 'sysdefault'


os.makedirs(OUTPUT_DIR, exist_ok=True)

class TextNormalizer:
    """LLM 답변을 TTS 에 넣기 전에 다듬는다.

    숫자를 미리 한글로 바꾸는 게 핵심이다. CosyVoice3 에 숫자를 그대로 주면
    '만이천 / 일이천 / 민이천' 으로 확률적으로 흔들린다. 아예 안 보여주면 흔들릴 일이 없다.
    """

    # ── 한자어 / 고유어 표 ──────────────────────────────────────
    SINO_DIGITS = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
    SINO_UNITS  = ['', '십', '백', '천']
    SINO_BIG    = ['', '만', '억', '조', '경']

    NATIVE_ONES = ['', '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉']
    NATIVE_TENS = ['', '열', '스물', '서른', '마흔', '쉰', '예순', '일흔', '여든', '아흔']

    # 고유어(하나 둘 셋)를 쓰는 단위
    NATIVE_UNITS = ('개', '명', '잔', '시', '그릇', '마리', '살', '판', '병', '접시',
                    '조각', '켤레', '자루', '벌', '장', '권', '대', '통', '봉지', '팩')

    # 앞글자가 고유어 단위와 겹치지만 실제로는 한자어인 것들.
    #   '3개월' 의 '개' 만 보면 '세 개월'(X). '삼 개월'(O)
    #   주의: '시간'/'시경' 은 고유어다(두 시간). 여기 넣으면 안 된다
    SINO_EXCEPTIONS = ('개월', '개국')

    # 전각 CJK 부호 -> ASCII. CosyVoice 가 중국어 모델이라 실제로 섞여 나온다
    PUNCT_MAP = {
        '。': '.', '，': ',', '？': '?', '！': '!', '…': '.', '、': ',',
        '：': ',', ':': ',',          # 콜론은 쉼표로. 그 자리에서 쉬게 한다
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
    }

    def __init__(self):
        self.re_zero_width = re.compile(r'[\u200B-\u200D\uFEFF]')  # 눈에 안 보이는 문자
        self.re_control    = re.compile(r'<<[^>]+>>')              # 나중에 규약 토큰 쓰면 여기서 걸린다
        self.re_bracket    = re.compile(r'[\(\[\{][^\)\]\}]*[\)\]\}]')  # 괄호는 내용째로 버린다
        self.re_markdown   = re.compile(r'[\*\#\_`]')
        self.re_whitelist  = re.compile(r'[^가-힣a-zA-Z0-9\s\.\,\?\!\'\"]')
        self.re_dup_punc   = re.compile(r'([.?!,])\1+')            # '!!!' -> '!'
        self.re_space_punc = re.compile(r'\s+([,.?!])')
        self.re_whitespace = re.compile(r'\s+')
        self.re_dangling   = re.compile(r'[,\s]+$')                # ':)' 지우고 남은 꼬리 쉼표

        # 숫자(콤마 허용) + 바로 뒤 한글 최대 2글자. 2글자를 보는 건 '개월' 때문
        self.re_number     = re.compile(r'(?P<num>\d[\d,]*)\s*(?P<tail>[가-힣]{1,2})?')

    # ── 숫자 -> 한글 ────────────────────────────────────────────
    def _to_sino(self, num):
        """한자어. 일 이 삼 … 십 백 천 만 억"""
        if num == 0:
            return '영'
        parts, big = [], 0
        while num > 0:
            chunk = num % 10000
            if chunk > 0:
                s = ''
                for i, d in enumerate(str(chunk)[::-1]):
                    if int(d) > 0:
                        name = '' if (int(d) == 1 and i > 0) else self.SINO_DIGITS[int(d)]
                        s = name + self.SINO_UNITS[i] + s
                if s == '일' and big > 0:      # '일만이천' 이 아니라 '만 이천'
                    s = ''
                parts.append(s + self.SINO_BIG[big])
            num //= 10000
            big += 1
        return ' '.join(parts[::-1])

    def _to_native(self, num):
        """고유어 관형사형. 1~99 만. 그 이상은 실사용에서 한자어를 쓴다"""
        if not 1 <= num <= 99:
            return None
        tens, ones = divmod(num, 10)
        if tens and not ones:
            return '스무' if tens == 2 else self.NATIVE_TENS[tens]   # '스물 개' 가 아니라 '스무 개'
        return (self.NATIVE_TENS[tens] + self.NATIVE_ONES[ones]).strip()

    def _number_to_korean(self, m):
        raw  = m.group('num').replace(',', '')   # 콤마는 여기서 뗀다. 화이트리스트보다 먼저다
        tail = m.group('tail') or ''
        num  = int(raw)

        if tail[:1] in self.NATIVE_UNITS and not tail.startswith(self.SINO_EXCEPTIONS):
            kor = self._to_native(num)
            if kor:
                return f'{kor} {tail}'
        return f'{self._to_sino(num)} {tail}'.rstrip()

    def normalize(self, text):
        """LLM 답변 -> TTS 에 넣을 문자열. 순서가 중요하다."""
        if not text:
            return ''

        s = self.re_zero_width.sub('', text)
        s = self.re_control.sub(' ', s)
        s = self.re_bracket.sub('', s)
        s = self.re_markdown.sub('', s)

        for src, dst in self.PUNCT_MAP.items():
            s = s.replace(src, dst)

        s = self.re_number.sub(self._number_to_korean, s)   # 화이트리스트보다 먼저. 콤마가 살아있어야 함
        s = self.re_whitelist.sub(' ', s)

        s = self.re_dup_punc.sub(r'\1', s)
        s = self.re_space_punc.sub(r'\1', s)
        s = self.re_whitespace.sub(' ', s)
        return self.re_dangling.sub('', s).strip()



class SynthStats:
    """한 번의 합성+재생에 대한 계측. llm_callback 을 계측 코드로 어지럽히지 않으려고 분리."""

    def __init__(self):
        self.started = time.perf_counter()
        self.starved = 0.0          # 큐가 비어 스피커가 굶은 총 시간
        self.first_packet = None

    @contextmanager
    def waiting(self):
        """q.get() 을 감싼다. 여기서 걸린 시간이 곧 스피커가 논 시간."""
        t0 = time.perf_counter()
        yield
        self.starved += time.perf_counter() - t0

    def mark_first_packet(self):
        if self.first_packet is None:
            self.first_packet = time.perf_counter() - self.started

    def summary(self, n_chunks: int, audio_sec: float, path: str) -> str:
        return (f'첫 패킷 {self.first_packet * 1000:.0f} ms | '
                f'스피커 대기 {self.starved:.2f} s | '
                f'청크 {n_chunks}개 | 오디오 {audio_sec:.2f} s | 저장 {path}')

class TTSNode(Node):
    def __init__(self):
        super().__init__('soomac_tts_node')

        self.get_logger().info('김도현 박지수 이준미 주재영 파이팅')

        self.cosyvoice_tts = AutoModel(model_dir=MODEL_DIR, fp16=False, load_trt=False) # tts 모델 객체!

        self.text_norm = TextNormalizer()

        # 무조건 워밍업 해줘야 빨라짐 ㅇㅇ

        self.get_logger().info("워밍업 시작")

        self.cosyvoice_tts.model.token_hop_len=25

        # token_hop_len: 몇 개의 speech token 이 모이면 오디오 청크 하나를 뱉을지.


        self.speaker = sd.OutputStream(samplerate=OUT_SR, channels=1, dtype='float32', device=OUT_DEVICE)

        self.speaker.start() # 얘는 실시간 아웃풋 용


        self.get_logger().info(f"오디오 출력 {sd.query_devices(self.speaker.device)['name']} / {OUT_SR}Hz로 오픈!!!")

        # 출력 장치 디버깅용


        for _ in self.cosyvoice_tts.inference_zero_shot("워밍업입니다.", PROMPT_TEXT, PROMPT_WAV, stream=True, text_frontend=False):
            pass


        self.get_logger().info("워밍업 종료. llm 토픽을 받아봅시다")

        self.stt_stop_pub = self.create_publisher(String, '/stt_stop', 1) # tts 음성이 stt안에 들어갈 수 있어서 이동안의 쌓인 큐를 없애기 위함
        self.stt_retrigger_pub = self.create_publisher(String, '/tts_done', 1) # 이거 도착하면 큐 내부 싹다 버려버리고 다시 음성 받음

        self.create_subscription(String, '/llm_response', self.llm_callback, 1)


    def llm_callback(self, message):

        raw = message.data.strip()

        llm_response = self.text_norm.normalize(raw)

        # text normalize 디버깅용

        if raw != llm_response:
            self.get_logger().warning(f"전처리에서 바뀜:\n  원본 {raw!r}\n  변환 {llm_response!r}")

        if not llm_response:
            self.get_logger().info("llm 응답 X")
            return

        self.cosyvoice_tts.model.token_hop_len=25

        self.stt_stop_pub.publish(String(data='speaking'))

        #   대신 자동 문장 분할이 꺼지므로 긴 답변은 우리가 쪼개야 한다.
        tts_generator = self.cosyvoice_tts.inference_zero_shot(llm_response, PROMPT_TEXT, PROMPT_WAV, stream=True, text_frontend=False)

        stats = SynthStats()

        chunks = []


        q = queue.Queue() # 큐에 청크 담아서 계속 넣어야함
       
        def producer():
            try:
                for packet in tts_generator:
                    q.put(packet['tts_speech'].squeeze(0).numpy().astype(np.float32)) # 큐에다가 생성한 청크 쌓음

            except Exception as e:
                self.get_logger().error(f"합성 스레드가 터졌어용 : {e}\n{traceback.format_exc()}")

            finally:
                q.put(None) # 종료 표시


        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while True:
                with stats.waiting():
                    chunk=q.get()

                if chunk is None:
                    break

                stats.mark_first_packet()
                chunks.append(chunk)

                out = np.clip(resample_poly(chunk, OUT_SR, SAMPLE_RATE), -1, 1).astype(np.float32)
                self.speaker.write(out.reshape(-1, 1))      # 실시간 용

        except Exception as error:
            self.get_logger().error(f"에러 터짐 샤갈! {error}")
            self.stt_retrigger_pub.publish(String(data='failed'))
            return

        finally:
            producer_thread.join()

        

        if not chunks:
            self.get_logger().error("합성된 청크가 없어요 샤갈!")
            self.stt_retrigger_pub.publish(String(data='failed')) # 
            return

        audio = np.concatenate(chunks) # 생성된 청크들 통합해버려 그냥

        sf.write(LAST_WAV, audio, SAMPLE_RATE) # 합성된거 sample rate로 저장

        total_time = len((audio)) / SAMPLE_RATE

        self.get_logger().info(stats.summary(len(chunks), total_time, LAST_WAV))

        self.stt_retrigger_pub.publish(String(data='finished'))

    def destroy_node(self):
        try:
            self.speaker.stop()
            self.speaker.close() # 실시간 용

        except Exception as error:
            # 여기서 조용히 실패하면 스피커가 안 닫힌 채 종료
            self.get_logger().warning(f"오디오 정리하다 터짐 : {error}")

        super().destroy_node() # destroy_node에 얹어버리기

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
