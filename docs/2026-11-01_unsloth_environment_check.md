# Unsloth 격리 환경 점검

참고: [Gemma 4 설정](/home/roma/Desktop/sLLM/gemma-4-12B-it/config.json), [Unsloth Vision 로더](/tmp/soomac-training-accel-20261101/venv/lib/python3.10/site-packages/unsloth/models/vision.py), [head 512 커널](/tmp/soomac-training-accel-20261101/venv/lib/python3.10/site-packages/unsloth/kernels/large_head_attention.py), [fused LoRA](/tmp/soomac-training-accel-20261101/venv/lib/python3.10/site-packages/unsloth/kernels/fast_lora.py) → 설치 소스가 Gemma 4 Unified 경로를 갖지만 학습 작동·속도는 GPU에서 별도 검증해야 한다.

## 환경과 의존성

- 실행 인터프리터: `/tmp/soomac-training-accel-20261101/venv/bin/python` (Python 3.10.20, `--system-site-packages`). 추가 패키지는 이 venv에만 설치한다.
- venv 설치: `unsloth==2026.10.11`, `unsloth-zoo==2026.10.11`, `xformers==0.0.35`, `xgrammar==0.2.8`. `torch==2.12.1+cu130`, `torchvision==0.27.1+cu130`, `triton==3.7.1`, `transformers==5.13.1`, `peft==0.19.1`, `accelerate==1.14.0`, `bitsandbytes==0.49.2`는 기존 `gemma4_env`에서 읽기 전용으로 상속한다.
- `pip check`: **No broken requirements found.** 처음에는 상속된 `xgrammar==0.2.5`가 `transformers<5`를 요구해 한 건 실패했다. `xgrammar==0.2.8`을 venv에만 `--no-deps`로 설치한 뒤 통과했다. 기존 `gemma4_env`의 패키지는 바꾸지 않았다.

원인: `--system-site-packages`는 기존 xgrammar 메타데이터까지 보이게 했고, 새 버전의 로컬 설치가 해당 요구사항을 대체했다.
결과: 설치 메타데이터의 의존성 충돌은 해소되었으나 GPU 커널·모델 학습 호환성은 아직 입증되지 않았다.

## Gemma 4 Unified와 가속 경로

모델 `config.json`의 `model_type`은 `gemma4_unified`, text 설정은 `gemma4_unified_text`, `global_head_dim=512`, `attention_dropout=0.0`이다. 설치된 Unsloth Vision 로더는 `gemma4_unified`를 명시적으로 인식한다(`vision.py:1289-1339`). 로드 말미에 head 512 라우터를 설치하려 한다(`vision.py:4936-4944`). 해당 커널은 head 512만 지원하며 입력 형태, 마스크, dtype, attention dropout, 자체 검증 결과에 따라 사용하거나 기존 SDPA로 돌아간다(`large_head_attention.py:25-28,870-959,973-1024`). **설치 소스 존재만으로 실제 커널 사용을 확정하지 않는다.**

기존 학습의 `lora_dropout=0.05`는 attention dropout 0.0과 다른 값이다. `FastVisionModel.get_peft_model`의 기본값은 `lora_dropout=0.0`이므로 **0.05를 명시적으로 전달해야** 기존 계약이 유지된다(`vision.py:4947-4953`). Unsloth fused LoRA는 기본 `UNSLOTH_FUSED_LORA_DROPOUT=1`에서 양수 dropout 경로를 갖고, `0`이면 해당 fused 경로 대신 PEFT로 돌아간다(`fast_lora.py:85-125,221-276`). 따라서 LoRA dropout을 0으로 바꿀 필요가 있다는 근거는 없다. 실제 fused 적용과 역전파 동등성은 root의 GPU 시험 대상이다.

원인: head 512 attention과 LoRA dropout은 서로 다른 코드 경로이며 둘 다 실행 조건이 있다.
결과: 기존 prompt·LoRA dropout·모델 구조를 유지한 채 격리 GPU 시험을 설계할 수 있다.

## 남은 제한

`xformers==0.0.35`는 metadata상 `torch>=2.10`을 만족해 `pip check`를 통과하지만, root의 실제 Unsloth import 로그에서는 torch 2.10/cu128용 바이너리라 현재 torch 2.12.1/cu130에서 로드되지 않았다. Unsloth는 이 예외를 잡아 PyTorch attention으로 돌아간다(`models/_utils.py:4755-4805`). 따라서 **xformers는 Unified 로더의 필수 진입 조건이 아니며**, 현재 환경에서 xformers 가속을 기대하지 않는다. head 512의 별도 커널 경로가 실제로 쓰이는지도 아직 확인하지 않았다.

root는 격리 환경에서 Unsloth import와 GPU SM 12.0 식별이 성공했다고 보고했다. 이 문서의 담당자는 GPU import·모델 로드·학습을 실행하지 않았다. 원격 Decision 봉인본, 기존 학습 환경, runtime/VLM, 데이터·checkpoint는 변경하지 않았다.

원인: Python 의존성 검사와 import는 CUDA 확장 바이너리·실제 training forward/backward를 검증하지 못한다.
결과: 준비 상태는 **격리 환경 설치/CPU 의존성 점검 완료**이며, 속도·메모리·동등성은 root의 GPU 시험 결과를 기다린다.

## 용어 정리

- 전역 head 차원(global head dimension): Gemma 4 full attention 층의 head 폭이며 이 모델은 512이다.
- fused LoRA: LoRA 계산을 결합한 구현이다. dropout 0.05 지원 코드가 있어도 실행 중 선택 여부는 별도 확인이 필요하다.
- SDPA(scaled dot product attention): PyTorch의 기본 attention 구현이며 이 환경에서 xformers 실패 시 대체 경로이다.

(내가 의역해보자면, 지금 확인된 것은 가속 시험을 시작할 **격리된 후보 환경**이다. 가속 자체는 아직 측정값이 없다.)
