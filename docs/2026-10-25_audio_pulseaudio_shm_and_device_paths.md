# 소리가 안 나요: `Connection failure: Connection terminated` 정리

> 작성 2026-10-25 · 브랜치 `llm_final`
> 대상: 오디오를 처음 다루는 사람도 읽을 수 있게 썼다.
> 조사 방식: 테스트 중이라 **설정은 하나도 바꾸지 않고 읽기만** 해서 확인했다. (사운드 서버를 한 번 껐다 켠 것만 예외)

---

## 0. 한 줄 요약

**컴퓨터 안의 "소리 우체국(PulseAudio)"이 "빠른 택배함(공유 메모리)"을 못 쓰게 되어, 택배함을 쓰려는 프로그램을 전부 돌려보내고 있다.**
택배함을 안 쓰는 "느린 일반 우편"으로 보내라고 설정하면 지금 바로 잘 된다.

---

## 1. 먼저 알아야 할 것 (비유로)

컴퓨터에서 소리가 나가고 들어오는 길을 **학교 방송실**에 비유하면 쉽다.

| 진짜 이름 | 비유 | 하는 일 |
|---|---|---|
| **사운드 카드** | 스피커·마이크가 꽂힌 **방송 장비** | 실제로 소리를 내고 받는 기계. 이 PC에는 3개가 있다 (아래 표) |
| **ALSA** | 방송 장비를 직접 만지는 **손** | 리눅스가 사운드 카드를 직접 다루는 기본 방법 |
| **PulseAudio** | 여러 반의 방송 요청을 받아 섞어 주는 **방송실 선생님(우체국)** | 여러 프로그램이 동시에 소리를 내도 섞어서 내보낸다 |
| **공유 메모리 (SHM / memfd)** | 선생님과 학생이 같이 쓰는 **빠른 택배함** | 소리 데이터를 복사하지 않고 함께 보는 빠른 통로 |
| **싱크 (sink)** | **소리가 나가는 곳** (스피커) | 예: 헤드폰 단자, USB 장치 |
| **소스 (source)** | **소리가 들어오는 곳** (마이크) | 예: MATA USB 마이크 |
| **카드 번호** (`hw:2`) | 장비에 붙인 **번호표** | USB를 꽂고 빼면 **바뀔 수 있다** → 위험 |
| **카드 이름** (`CARD=PCH`) | 장비의 **이름표** | 꽂는 순서가 바뀌어도 **그대로** → 안전 |

### 이 PC의 방송 장비 3개

| 번호표 | 이름표 | 무엇 | 우리 쓰임 |
|---|---|---|---|
| card 0 | `PCH` | 메인보드 내장 사운드 (ALC897) | **TTS 스피커/헤드폰** |
| card 1 | `NVidia` | 그래픽카드 HDMI 소리 | 안 씀 |
| card 2 | `C10` | **MATA STUDIO C10** (USB 마이크) | **STT 마이크** |

---

## 2. 무슨 일이 있었나 (증상)

```
roma@roma-Z790-UD:~$ soomac_tts
Connection failure: Connection terminated
```

- `soomac_tts`는 TTS를 켜기 **전에** `pactl`(우체국에 말 거는 명령)로 사운드 설정을 바꾼다 (`~/.bashrc` 220행).
- 그 `pactl`이 우체국에 들어가자마자 쫓겨나서, `|| exit 1` 때문에 **TTS는 아예 켜지지도 않았다.**
- `pactl info`처럼 아무 설정도 안 바꾸는 명령도 똑같이 쫓겨난다.

---

## 3. 원인

### 3-1. 확인된 것 ✅ — 쫓겨나는 정확한 순간

`pactl`에 상세 기록을 켜고 실행하면 이렇게 나온다.

```
Trying to connect to /run/user/1000/pulse/native...   ← 우체국 문 열기: 성공
Protocol version: remote 35, local 35                 ← 인사: 성공
Negotiated SHM: yes                                   ← "택배함 쓸게요": 합의
Negotiated SHM type: shared memfd
Connection failure: Connection terminated             ← 택배함 쓰려는 순간 쫓겨남
```

그래서 택배함만 꺼 보았다.

| 시험 | 결과 |
|---|---|
| 그냥 `pactl info` | ❌ 쫓겨남 |
| **택배함(공유 메모리) 끄고** `pactl info` | ✅ **정상** |
| memfd만 끄고 (다른 종류 택배함 사용) | ❌ 쫓겨남 |
| PulseAudio·PipeWire 전부 껐다 켠 뒤 | ❌ 그대로 |

→ **원인: 우체국과 프로그램 사이의 "공유 메모리 택배함"이 고장.** 택배함 종류와 상관없이 전부 안 된다.

### 3-2. 가장 유력한 계기 ⚠ (추정, 아직 확인 안 함)

오늘 부팅 기록을 비교하면 딱 하나가 다르다.

| 부팅 | `soomac_tts` | MATA STUDIO C10 (USB 마이크) |
|---|---|---|
| 14:17 | ✅ 정상 (TTS가 오후 내내 동작) | **없음** |
| 19:48 | (모름) | **새로 연결됨** |
| 19:53 (지금) | ❌ 쫓겨남 | **있음** |

- 커널(6.8.0-138), PulseAudio(15.99.1), systemd 버전은 14:17 때와 **같다**.
- 그래서 **MATA USB 장치가 붙은 뒤부터** 문제가 생긴 것으로 보인다.
- ⚠ "같은 때 생겼다"는 것이지 "이것 때문"이라고 증명한 것은 아니다. 왜 USB 장치 하나가 택배함을 고장 내는지는 **모른다**. (4-3에서 확인 방법)

### 3-3. 원인이 아닌 것 ❌ (직접 확인해서 지움)

| 의심 | 확인 결과 |
|---|---|
| 좀비 세션 (죽은 프로그램이 자리 차지) | 서버를 새로 켜도 똑같음 → 아님 |
| PulseAudio와 PipeWire가 같은 문을 두고 싸움 | 문(`pulse/native`)은 PulseAudio만 잡고 있음 → 아님 |
| 택배함 공간 부족 (`/dev/shm`) | 32GB 중 1% 사용 → 아님 |
| 열 수 있는 파일 수 제한 | 256개 중 44개 사용 → 아님 |
| 블루투스 25초 대기 (`org.bluez timed out`) | 매 부팅마다 있던 것, 끝난 뒤에도 쫓겨남 → 아님 (블루투스 서비스가 꺼져 있어서 생기는 별개 경고) |
| 설정 파일 변경 | `/etc/pulse/daemon.conf`·`client.conf` 모두 기본값, `~/.config/pulse/` 에 덮어쓰는 설정 없음 → 아님 |

---

## 4. 해결 방법

### 4-1. 지금 당장 (그 터미널에서만, 되돌리기 쉬움) ← 테스트할 때 이걸 쓴다

```bash
printf 'enable-shm = no\n' > ~/pulse_noshm.conf      # 처음 한 번만
export PULSE_CLIENTCONFIG=~/pulse_noshm.conf
soomac_tts        # 또는 soomac_stt, stt_qwen 등
```

- 뜻: "이 터미널에서 켜는 프로그램은 택배함 말고 일반 우편으로 보내라."
- 터미널을 닫으면 원래대로 돌아간다. 시스템 설정은 안 바뀐다.
- 20:44에 켠 STT(Qwen3)는 이미 이 방식으로 켜져서 잘 돌고 있다.

### 4-2. 매번 치기 귀찮으면 (모든 프로그램에 적용)

```bash
mkdir -p ~/.config/pulse
printf 'enable-shm = no\n' > ~/.config/pulse/client.conf
```

- 이 PC의 **모든** 프로그램이 일반 우편을 쓴다. 소리 데이터가 아주 작아서 체감 속도 차이는 거의 없다(추정).
- 되돌리기: `rm ~/.config/pulse/client.conf`

### 4-3. 진짜 원인 확인 (테스트가 끝난 뒤에)

1. MATA USB 마이크를 **뽑는다**
2. `systemctl --user restart pulseaudio`
3. `pactl info` → **되면** MATA가 계기 확정. **안 되면** 다른 원인
4. 다시 꽂고 2~3 반복 → 다시 안 되면 확정
5. 더 자세히 보려면: `pacmd set-log-level 4` → `pactl info` → `journalctl --user -u pulseaudio --since -1min` → `pacmd set-log-level 1`
   (우체국이 **왜** 쫓아냈는지 서버 쪽 이유가 나온다)

---

## 5. 마이크·스피커 하드웨어 경로 점검

"소리가 어느 길로 가는지" 하나씩 따라가 보았다.

### 5-1. 길 그림

```
[TTS 스피커]
 tts_node.py ──(TTS_DEVICE=soomac_speaker)──▶ ALSA "손" ──▶ CARD=PCH (헤드폰 단자)
                                         ↑ PulseAudio(우체국) 안 거침

[STT 마이크]
 MATA USB 마이크 ──▶ PulseAudio(우체국) ──▶ arecord -D pulse ──▶ stt_qwen3_asr_node.py
                     ↑ 우체국을 거침 → 이번 고장의 영향을 받음

[그 외 기본 소리] (VS Code에서 WAV 열기, aplay 장치 안 정한 것 등)
 프로그램 ──▶ ~/.asoundrc: pcm.!default = pulse ──▶ PulseAudio ──▶ 기본 싱크
```

### 5-2. 점검 결과

| 항목 | 지금 상태 | 판정 | 설명 |
|---|---|---|---|
| TTS 출력 장치 | `soomac_speaker` = `hw:CARD=PCH,DEV=0` (`~/.asoundrc`) | ✅ 안전 | **이름표**로 찾아서 USB를 꽂고 빼도 헤드폰 단자를 정확히 찾는다 |
| TTS 출력과 PulseAudio | TTS 소리는 우체국을 안 거침 | ✅ | 소리 자체는 고장과 무관. **켜기 전 `pactl` 설정 줄만** 막힘 |
| STT 입력 장치 | `alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo` | ✅ 안전 | **이름**으로 지정 (`SOOMAC_AUDIO_SOURCE`). 번호표 아님 |
| STT와 PulseAudio | 마이크를 우체국으로 받음 (`stt_nemotron_node.py` 375~376행 `arecord -D pulse`, 시작할 때 `pactl list short sources`) | ⚠ **영향 받음** | 4-1 우회 없이 켜면 STT 시작 단계에서 "입력 장치를 확인하지 못했어요"로 멈춘다 |
| `soomac_stt` / `stt_qwen` 함수 | 시작할 때 `pactl set-default-source …` 실행 (`~/.bashrc` 208·327·337·349행) | ⚠ **영향 받음** | TTS와 똑같이 첫 줄에서 막힌다 |
| 기본 싱크 (기본 스피커) | **MATA USB 장치**로 잡혀 있음 | ⚠ **헷갈림 주의** | 장치를 안 정하고 틀면 PC 헤드폰이 아니라 **MATA 쪽**으로 나간다. 오후에 "VS Code에서 WAV가 안 들린다"의 원인 중 하나로 보인다(추정) |
| 헤드폰 단자 독점 | `tts_node.py`가 켜져 있으면 헤드폰 단자를 혼자 씀 | ℹ 원래 그렇게 설계 | 그동안 다른 프로그램은 헤드폰 단자로 소리를 못 낸다. 테스트 WAV를 들으려면 TTS를 잠깐 꺼야 한다 |
| 코드 속 번호표 하드코딩 | `hw:2` 같은 숫자 지정 **없음** | ✅ | 전부 이름으로 지정되어 있다 |
| `tts_node.py`를 직접 실행할 때 | `TTS_DEVICE`가 없으면 `sysdefault` 사용 (93행) | ⚠ 약간 위험 | `sysdefault`는 card 0을 따라간다. 지금은 PCH가 card 0이라 괜찮지만, 번호표 방식이다. **항상 `soomac_tts`로 켜는 게 안전** |
| 사운드 칩 절전 | `snd_hda_intel power_save = 1` | ℹ 참고 | 1초 넘게 조용하면 칩이 잠든다. `aplay`처럼 매번 장치를 새로 여는 프로그램은 **첫 소리가 살짝 잘릴 수 있다**. TTS 노드는 장치를 계속 열어 두므로 영향이 적다 |

### 5-3. 대회 당일 체크리스트

1. [ ] MATA USB 마이크를 **같은 USB 구멍**에 꽂았다 (이름으로 찾지만, 습관적으로 같은 곳에)
2. [ ] `pactl info`가 되는지 본다. 안 되면 4-1 우회를 먼저 한다
3. [ ] `pactl list short sources | grep MATA` → 마이크가 보인다
4. [ ] `soomac_tts` 시작 로그에 `오디오 출력 soomac_speaker / 48000Hz로 오픈!!!`과 `참조 음성 …`이 찍힌다
5. [ ] 로봇에게 한 문장 말하게 해서 **헤드폰/스피커**로 들린다
6. [ ] 마이크에 한 문장 말해서 화면에 "이렇게 들었어요"가 뜬다

---

## 6. 대회 당일: 경로 확인하기 · 바꾸기 · 꼬였을 때 대처

> 아래 명령은 전부 터미널에 **그대로 복사**해서 쓰면 된다.
> ⚠ PulseAudio 고장(2·3장)이 아직 있으면, 터미널을 열자마자 이 줄을 **먼저** 친다.
> ```bash
> export PULSE_CLIENTCONFIG=~/pulse_noshm.conf
> ```

### 6-1. 지금 소리 길 지도

```
            ┌──────────── 우리 로봇 ────────────┐
 TTS 목소리 ─▶ soomac_speaker ─▶ CARD=PCH 0번 ─▶ 🎧 PC 헤드폰/스피커 단자 (초록 구멍)
                (~/.asoundrc 에 정의)      ↑ 우체국(PulseAudio) 안 거침

 🎤 MATA USB 마이크 ─▶ 우체국(PulseAudio) ─▶ arecord ─▶ STT 노드
     이름: alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo
            └──────────────────────────────────┘
```

- **TTS는 이름표 `soomac_speaker` 하나만 보면 된다.** 실제 장치는 `~/.asoundrc` 한 곳에 적혀 있다.
- **STT는 마이크 이름 하나만 보면 된다.** `~/.bashrc`의 `soomac_stt`·`stt_qwen` 안에 적혀 있다.
- `soomac_tts`는 켤 때 PCH 카드를 **디지털 모드(`output:iec958-stereo`)로 바꾼다.** 우체국이 헤드폰 단자를 놓아주게 해서, TTS가 단자를 직접 잡게 하는 장치다. TTS를 끄면 아날로그로 되돌린다(`.bashrc` 225~233행 `trap`).

### 6-2. 확인하는 법 (5개만 보면 된다)

| # | 무엇을 | 명령 | 정상이면 |
|---|---|---|---|
| 1 | 장비가 다 꽂혔나 | `cat /proc/asound/cards` | `PCH`, `NVidia`, `C10` 세 줄 |
| 2 | 우체국이 살아 있나 | `pactl info \| grep "Server Name"` | `Server Name: pulseaudio` |
| 3 | 마이크가 보이나 | `pactl list short sources \| grep -v monitor` | `alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo` 한 줄 |
| 4 | 헤드폰 단자를 누가 잡고 있나 | `fuser -v /dev/snd/pcmC0D0p` | TTS가 켜져 있으면 `python` 하나. 꺼져 있으면 아무것도 없음 |
| 5 | TTS가 어디로 내보내나 | TTS 시작 로그 | `오디오 출력 soomac_speaker / 48000Hz로 오픈!!!` (이름표가 찍힌다. 이름표가 실제로 가리키는 장치는 `~/.asoundrc`에서 본다) |

**직접 들어 보기** (TTS를 **끈 상태**에서. 켜져 있으면 헤드폰 단자를 혼자 써서 안 된다)
```bash
# 스피커: "Front Left, Front Right" 라고 말한다
speaker-test -D soomac_speaker -c 2 -t wav -l 1

# 마이크: 3초 녹음해서 바로 다시 틀어 본다
PULSE_SOURCE=alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo \
  arecord -D pulse -f S16_LE -r 16000 -c 1 -d 3 ~/mic_test.wav
aplay -D soomac_speaker ~/mic_test.wav
```

### 6-3. 바꾸는 법

**① TTS 소리를 다른 곳으로 내보내고 싶다** (예: 모니터 HDMI 스피커)

1. 장비 이름 찾기: `aplay -l` → `card 1: NVidia [...], device 3: HDMI 0` 처럼 나온다
   - `card 뒤의 영어 이름`(`NVidia`)이 **이름표**, `device 뒤 숫자`(`3`)가 구멍 번호
2. `~/.asoundrc`의 한 줄만 바꾼다
   ```
   slave.pcm "hw:CARD=PCH,DEV=0"      # 지금: PC 헤드폰 단자
   slave.pcm "hw:CARD=NVidia,DEV=3"   # 예: 모니터 HDMI
   ```
3. TTS를 끈 상태에서 `speaker-test -D soomac_speaker -c 2 -t wav -l 1` 로 새 곳에서 소리가 나는지 듣는다 → `soomac_tts`를 다시 켠다
   - 로그에는 계속 `soomac_speaker`로 찍힌다. 이름표는 그대로고 가리키는 곳만 바뀐 것이라 정상이다
- 한 번만 시험하려면 파일을 안 고치고 `TTS_DEVICE='hw:CARD=NVidia,DEV=3'`를 앞에 붙여 `tts_node.py`를 켜도 된다
- ⚠ `hw:1,3` 같은 **숫자만** 쓰지 않는다. USB를 꽂고 빼면 번호가 바뀐다

**② 다른 마이크로 바꾸고 싶다** (예: 마이크가 고장 나서 다른 USB 마이크를 꽂음)

1. 새 마이크 이름 찾기: `pactl list short sources | grep -v monitor`
   - `.monitor`로 끝나는 건 마이크가 아니라 **스피커 소리를 엿듣는 가짜 입력**이니 고르면 안 된다
2. `~/.bashrc`에서 `alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo`를 **새 이름으로 모두 바꾼다** (209·328·338·341·344·349행, 6곳)
3. `source ~/.bashrc` 후 STT를 다시 켠다
- 한 번만 시험하려면: `SOOMAC_AUDIO_SOURCE=<새 이름>` 을 앞에 붙여 STT를 켠다

**③ 크기(볼륨) 조절**
```bash
pactl set-source-volume alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo 80%   # 마이크 (지금 80%)
amixer -c PCH sset Master 70%                                                       # 스피커 (지금 70%)
```

### 6-4. 꼬였을 때 대처 (증상 → 원인 → 해결)

| 증상 | 가장 흔한 원인 | 해결 |
|---|---|---|
| `soomac_tts`/`soomac_stt` 하자마자 `Connection failure: Connection terminated` | 우체국 택배함 고장 (2·3장) | `export PULSE_CLIENTCONFIG=~/pulse_noshm.conf` 후 다시 실행 |
| TTS 시작 중 `Device or resource busy` 또는 `Invalid sample rate` 로 죽음 | 헤드폰 단자를 **다른 프로그램이 잡고 있음** (예전 TTS가 안 꺼짐, 또는 우체국이 아날로그 모드로 잡음) | ① `fuser -v /dev/snd/pcmC0D0p` 로 범인 확인 → 그 프로그램 끄기 ② `pactl set-card-profile alsa_card.pci-0000_00_1f.3 output:iec958-stereo` 후 다시 실행 |
| TTS 로그는 정상인데 **소리가 안 들림** | ① 스피커 선이 **다른 구멍**에 꽂힘 ② 음량 0 | ① 선을 PC 뒤 **초록 구멍**(또는 앞 헤드폰 구멍)에 꽂기 ② `amixer -c PCH sset Master 70% unmute; amixer -c PCH sset PCM 100% unmute; amixer -c PCH sset Front 100% unmute` |
| TTS 로그에 `soomac_speaker`가 아니라 `sysdefault`가 찍힘 | `TTS_DEVICE` 없이 `tts_node.py`를 직접 켬 → `sysdefault`(번호표 방식) 사용 | **항상 `soomac_tts`로 켠다** (`TTS_DEVICE=soomac_speaker`가 들어 있음) |
| VS Code·`aplay`로 튼 소리가 안 들림 | ① 기본 스피커가 **MATA USB**로 잡혀 있음 ② TTS가 헤드폰 단자를 쓰는 중 | ① 장치를 지정해서 튼다: `aplay -D soomac_speaker 파일.wav` (TTS 끄고) ② TTS를 잠깐 끄기 |
| STT 시작 시 `지정한 입력 장치가 없어요` | 마이크가 안 꽂힘 / 인식 안 됨 / 이름이 바뀜 | ① USB 빼고 다시 꽂기 → 6-2의 1·3번 확인 ② 이름이 달라졌으면 6-3 ② |
| STT 시작 시 `입력이 마이크가 아닌 출력 모니터예요` | 마이크 대신 `.monitor` 가 골라짐 | `pactl set-default-source alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo` 후 다시 실행 |
| STT가 켜졌는데 **말을 못 알아들음/너무 작게** 들음 | 마이크 볼륨이 낮거나 음소거 | 6-2 마이크 녹음 테스트로 들어 보기 → 작으면 6-3 ③ |
| 로봇이 말할 때 **첫 글자가 살짝 잘림** (`aplay` 시험 때) | 사운드 칩 절전 (`power_save=1`) | 시험할 때는 파일 앞에 무음을 넣거나 여러 파일을 한 파일로 이어서 튼다. TTS 노드는 장치를 계속 열어 둬서 대부분 괜찮다 |
| 모니터·USB를 꽂은 뒤 **VS Code나 다른 프로그램 소리**가 엉뚱한 곳에서 남 | 우체국 자동 전환이 기본 스피커를 새 장치로 바꿈 | TTS는 영향 없음. 다른 소리는 `aplay -D soomac_speaker` 처럼 장치를 지정하거나 6-6 ② |
| 이어폰을 앞 구멍에 꽂았더니 **스피커가 조용해짐** | Auto-Mute: 앞 헤드폰을 꽂으면 뒤 스피커를 끔 | 이어폰 빼기, 또는 6-6 ③ |
| USB 헤드셋을 꽂은 뒤 STT가 이상함 | 기본 마이크가 헤드셋으로 바뀜 | `pactl get-default-source` 확인 → `pactl set-default-source alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo` → STT 다시 켜기 |
| 다 해도 안 됨 | 우체국 상태 꼬임 | `systemctl --user restart pulseaudio` → 30초 기다림 (블루투스 대기 25초) → 6-2 처음부터 |

### 6-5. 1분 응급 순서 (당황했을 때 이것만)

1. 터미널: `export PULSE_CLIENTCONFIG=~/pulse_noshm.conf`
2. `cat /proc/asound/cards` → 세 장비(`PCH`, `NVidia`, `C10`)가 다 있나? 없으면 **USB 다시 꽂기**
3. TTS·STT 끄기 → `speaker-test -D soomac_speaker -c 2 -t wav -l 1` → 들리나? 안 들리면 **선·음량** (6-4 세 번째 줄)
4. 마이크 3초 녹음 테스트 (6-2) → 들리나? 안 들리면 **마이크 이름·볼륨** (6-3 ②③)
5. `soomac_tts`, `soomac_stt`(또는 `stt_qwen`) 다시 켜기

### 6-6. HDMI 모니터·USB-A 장치가 소리 길을 빼앗는 경우

대회장에서는 **터치 모니터(HDMI)**, **USB 스피커·헤드셋** 같은 것을 새로 꽂을 수 있다.
이것들도 컴퓨터 입장에서는 "새 방송 장비"라서, 소리 길을 빼앗아 갈 수 있다.

**이 PC에서 확인한 상태 (2026-10-25)**

| 설정 | 지금 | 뜻 |
|---|---|---|
| 새 장치 자동 전환 `module-switch-on-connect` (`/etc/pulse/default.pa` 37행) | **켜져 있음** | HDMI·USB 소리 장치를 꽂는 순간 우체국이 **기본 스피커·기본 마이크를 그쪽으로 바꾼다** |
| 꽂힌 구멍 자동 전환 `module-switch-on-port-available` (33행) | **켜져 있음** | 앞 헤드폰 구멍에 꽂으면 우체국이 출력 구멍을 헤드폰으로 바꾼다 |
| 헤드폰 꽂으면 스피커 끄기 `Auto-Mute Mode` | **Enabled** | **앞 헤드폰 구멍에 뭘 꽂으면 뒤 초록 구멍(스피커)이 꺼진다** |
| 지금 꽂힌 구멍 | 뒤 초록 구멍 `Line Out` = **꽂힘**, 앞 헤드폰 = 비어 있음 | 지금은 뒤 초록 구멍으로 나간다 |
| HDMI 모니터 | `24touch75` 연결됨 | 이 모니터도 스피커 후보로 잡혀 있다 |
| USB 카드 번호 | `/etc/modprobe.d/alsa-base.conf`: `snd-usb-audio index=-2` | USB 장치는 **0번(PCH 자리)을 절대 못 가져간다** ✅ |

**누가 무엇을 빼앗을 수 있나**

| 새로 꽂은 것 | TTS 목소리 | STT 마이크 | 그 외 소리 (VS Code, 기본 장치) |
|---|---|---|---|
| HDMI 모니터 / HDMI 선 다시 꽂기 | ✅ 안 뺏김 (`soomac_speaker` = 이름표 `PCH`로 고정) | ✅ 상관없음 | ⚠ 기본 스피커가 **모니터**로 바뀔 수 있음 |
| USB 스피커·USB DAC (USB-A) | ✅ 안 뺏김 | ✅ 상관없음 | ⚠ 기본 스피커가 **USB**로 바뀜 |
| USB 헤드셋 (마이크 달린 것) | ✅ 안 뺏김 | ⚠ **기본 마이크가 헤드셋으로 바뀜** → 아래 설명 | ⚠ 기본 스피커도 바뀜 |
| 앞 헤드폰 구멍에 이어폰 꽂기 | ⚠ **뒤 스피커가 꺼지고 이어폰으로만** 나옴 (Auto-Mute) | ✅ 상관없음 | 같음 |
| 뒤 분홍 구멍(마이크)에 마이크 꽂기 | ✅ 안 뺏김 | ⚠ 기본 마이크가 바뀔 수 있음 | — |

- **TTS는 안전하다.** 우체국을 안 거치고 `~/.asoundrc`의 `hw:CARD=PCH,DEV=0`(이름표)으로 직접 나간다. 모니터나 USB를 꽂아도 길이 안 바뀐다.
  - 단, **헤드폰 구멍 안에서** 앞/뒤가 바뀌는 건 막지 못한다. 앞 구멍에 꽂으면 뒤 스피커가 꺼진다.
- **STT 마이크**: `stt_qwen`·`stt_nemotron` 은 `SOOMAC_AUDIO_SOURCE` 와 `PULSE_SOURCE` 로 **MATA를 이름으로 지정**해서 연다. 그래서 USB 헤드셋을 꽂아도 원래는 MATA를 계속 쓴다.
  - ⚠ (추정) 우체국의 "자동 전환"이 **이미 열린 녹음까지 새 장치로 옮길 수 있는지**는 이 PC에서 확인하지 못했다. 안전하게 하려면 아래 ②를 한다.
  - `soomac_stt`·`soomac_stt_v3`(Clova, `stt_node.py`)는 **기본 마이크**를 쓴다. 헤드셋을 꽂으면 그쪽으로 바뀔 수 있으니 대회에서는 `stt_qwen` 쪽을 쓴다.

**막는 법 (필요할 때만. 되돌리기 쉬운 순서)**

① **그냥 안 꽂는다** (가장 안전): TTS·STT가 켜진 뒤에는 HDMI 선이나 USB 소리 장치를 **새로 꽂거나 빼지 않는다**. 꽂아야 하면 **TTS·STT를 켜기 전에** 꽂는다.

② **자동 전환 끄기** (이번 부팅 동안만, 재부팅하면 다시 켜짐)
```bash
pactl unload-module module-switch-on-connect
```
- 영구로 끄려면 `~/.config/pulse/default.pa` 파일을 만들고 아래 두 줄을 넣은 뒤 `systemctl --user restart pulseaudio`:
  ```
  .include /etc/pulse/default.pa
  unload-module module-switch-on-connect
  ```
- 되돌리기: 파일을 지우고 `systemctl --user restart pulseaudio`

③ **앞 헤드폰 구멍에 꽂아도 뒤 스피커가 안 꺼지게** (앞·뒤 둘 다 소리 남)
```bash
amixer -c PCH sset 'Auto-Mute Mode' Disabled
```
- 되돌리기: `amixer -c PCH sset 'Auto-Mute Mode' Enabled`
- 앞 헤드폰 구멍을 쓸 거면 음량도 같이: `amixer -c PCH sset Headphone 100% unmute`

**빼앗겼는지 확인하는 법**

```bash
# 모니터가 꽂혀 있나 (1 = 꽂힘)
grep monitor_present /proc/asound/NVidia/eld#0.0

# 앞·뒤 어느 구멍에 꽂혀 있나 (values=on 이 꽂힌 곳)
amixer -c PCH contents | grep -A2 "Jack'" | grep -E "name=|values"

# 우체국의 기본 스피커·마이크가 무엇인가
pactl get-default-sink
pactl get-default-source        # MATA 여야 정상: alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo

# 기본 마이크가 바뀌었으면 되돌리기
pactl set-default-source alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo
```

---

## 7. 아직 모르는 것

- MATA USB 장치가 **왜** 공유 메모리를 고장 내는지 (4-3으로 확인 필요)
- 우체국 자동 전환(`module-switch-on-connect`)이 **이미 열려 있는 STT 녹음까지** 새 USB 마이크로 옮기는지 (6-6, 이름으로 지정했으니 안 옮길 것으로 추정)
- 4-2처럼 공유 메모리를 끈 상태로 오래 돌렸을 때 소리가 끊기거나 늦어지는지 (일반적으로 문제없다고 알려져 있지만 이 PC에서는 미확인)
