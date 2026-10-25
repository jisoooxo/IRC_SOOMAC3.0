"""TTS 재생 직전에 문장 사이 무음만 늘린다.

CosyVoice 3 는 쉼표에서도 길게는 0.6 초까지 쉰다. 그래서 무음 길이만 보면
문장 사이(마침표·물음표 뒤)와 쉼표를 가를 수 없다.
합성할 글은 미리 알고 있으니, 지금까지 나온 말소리 시간으로 글의 어디쯤인지 추정하고
문장 끝 근처에서 끝난 무음만 목표 길이까지 늘린다.

- 모델에는 글 전체를 그대로 준다. 문장을 나눠서 합성하지 않는다.
- 10 ms 프레임 단위로 판단하고 청크 경계를 넘어 무음을 이어 센다.
  청크 경계는 문장 경계가 아니다.
- 이미 받은 소리는 바로 내보낸다. 늘릴 무음은 무음이 끝나는 순간(다음 말소리 직전)에 끼운다.
  그래서 첫 소리가 늦어지지 않는다(10 ms 미만 자투리만 다음 청크까지 들고 있다).
- 판단이 애매하면 늘리지 않는다. 틀려도 원래 소리 그대로다.
"""

import re

import numpy as np

# 검증 데이터(cedar_mid 참조, 무음 166개 정답 라벨)로 고른 값이다.
#   글자당 말소리 시간은 참조 목소리마다 다르다. cedar 계열은 약 0.14 초다.
DEFAULT_SEC_PER_CHAR = 0.14
DEFAULT_MIN_PAUSE_SEC = 0.10      # 이보다 짧은 무음은 문장 사이로 보지 않는다
DEFAULT_POSITION_TOLERANCE = 0.15  # 문장 끝 위치 추정 허용 오차(그 위치까지 글자 수의 비율)
SILENCE_DBFS = -45.0
FRAME_SEC = 0.01
MIN_TOLERANCE_CHARS = 8            # 짧은 문장에서도 최소 이만큼은 허용

SPOKEN_CHAR = re.compile(r'[가-힣a-zA-Z0-9]')


def sentence_layout(text):
    """(소리 나는 글자 수, 문장 끝 위치들, 쉼표 위치들). 위치 = 그 앞 글자 수. 마지막 문장 끝은 뺀다."""
    count = 0
    sentence_ends = []
    commas = []
    for character in text:
        if SPOKEN_CHAR.match(character):
            count += 1
        elif character in '.?!':
            sentence_ends.append(count)
        elif character == ',':
            commas.append(count)
    inner_ends = sorted({position for position in sentence_ends if 0 < position < count})
    return count, inner_ends, commas


class SentencePauseStretcher:
    """청크를 받는 대로 내보내되, 문장 사이 무음만 target_pause_sec 까지 늘린다."""

    def __init__(self, text, sample_rate, target_pause_sec,
                 sec_per_char=DEFAULT_SEC_PER_CHAR,
                 min_pause_sec=DEFAULT_MIN_PAUSE_SEC,
                 position_tolerance=DEFAULT_POSITION_TOLERANCE):
        self.sample_rate = sample_rate
        self.frame = max(1, int(round(sample_rate * FRAME_SEC)))
        self.target_frames = int(round(target_pause_sec / FRAME_SEC))
        self.min_pause_frames = int(round(min_pause_sec / FRAME_SEC))
        self.sec_per_char = sec_per_char
        self.position_tolerance = position_tolerance
        self.threshold = 10 ** (SILENCE_DBFS / 20)

        _, self.sentence_ends, self.commas = sentence_layout(text)
        self.used_ends = set()

        self.remainder = np.zeros(0, dtype=np.float32)
        self.frames_seen = 0
        self.voiced_frames = 0
        self.silent_frames = 0
        self.inserted = []   # (원본 기준 무음 시작 초, 원래 길이 초, 끼운 길이 초)

    def _is_sentence_boundary(self):
        """방금 끝난 무음이 문장 사이인가. 지금까지 말소리로 글 위치를 추정해 판단한다."""
        if self.silent_frames < self.min_pause_frames:
            return None
        position = self.voiced_frames * FRAME_SEC / self.sec_per_char
        candidates = [
            end for end in self.sentence_ends
            if end not in self.used_ends
            and abs(end - position) <= self.position_tolerance * max(end, MIN_TOLERANCE_CHARS)
        ]
        if not candidates:
            return None
        nearest_end = min(candidates, key=lambda end: abs(end - position))
        nearest_comma = min((abs(comma - position) for comma in self.commas), default=float('inf'))
        # 쉼표가 더 가까우면 쉼표 자리로 보고 건드리지 않는다
        if abs(nearest_end - position) > nearest_comma:
            return None
        return nearest_end

    def feed(self, chunk):
        """새 청크를 받아 지금 재생할 소리를 돌려준다."""
        samples = np.concatenate([self.remainder, np.asarray(chunk, dtype=np.float32).reshape(-1)])
        whole = len(samples) // self.frame * self.frame
        self.remainder = samples[whole:]
        pieces = []
        for start in range(0, whole, self.frame):
            frame = samples[start:start + self.frame]
            voiced = np.sqrt(np.mean(frame * frame)) > self.threshold
            if voiced:
                # 말소리가 다시 시작됐다 = 앞의 무음이 끝났다. 첫 말소리 전 무음은 세지 않는다
                if self.voiced_frames and self.silent_frames:
                    end = self._is_sentence_boundary()
                    extra = self.target_frames - self.silent_frames
                    if end is not None and extra > 0:
                        self.used_ends.add(end)
                        pieces.append(np.zeros(extra * self.frame, dtype=np.float32))
                        self.inserted.append((
                            round((self.frames_seen - self.silent_frames) * FRAME_SEC, 2),
                            round(self.silent_frames * FRAME_SEC, 2),
                            round(extra * FRAME_SEC, 2)))
                self.voiced_frames += 1
                self.silent_frames = 0
            elif self.voiced_frames:
                self.silent_frames += 1
            pieces.append(frame)
            self.frames_seen += 1
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def flush(self):
        """남은 자투리(10 ms 미만)를 내보낸다. 끝 무음은 늘리지 않는다."""
        tail, self.remainder = self.remainder, np.zeros(0, dtype=np.float32)
        return tail
