# 학습 중 공유 접두부 재사용: PrefixFlow·Tree Training 적용성

- [PrefixFlow 원문 v4 (2026-10-15)](https://arxiv.org/html/2606.01143v4) → 접두부 순전파·접미부 역전파·지연된 접두부 역전파를 세 단계로 분리한다.
- [Tree Training 원문 v5 (2026-04-23)](https://arxiv.org/html/2511.00413v5) → 분기 궤적을 트리로 묶고, 손실 가중치와 그래디언트를 복원한다.
- [Gemma 4 12B-it 공개 설정](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json) → 현재 봉인 모델의 설정과 별도로 대조해야 하는 아키텍처 참고값이다.

> **핵심 한 문장**  두 논문은 학습 중 중복 접두부 계산을 줄이는 실재 연구이지만, 현재 Gemma 4 12B-it·QLoRA 드롭아웃 0.05·유효 배치 16의 동일한 학습 목표와 재개 계약을 유지하면서 곧바로 적용할 수 있다는 증거는 없다.

## 1. Introduction / 문제와 적용 범위

두 논문의 대상은 **동일한 토큰·위치·주의 문맥을 공유하는 여러 학습 경로**이다. 추론용 KV 캐시를 그대로 학습에 쓰면 접미부 손실에서 접두부 LoRA 파라미터로 돌아갈 그래디언트가 끊긴다. PrefixFlow는 이 경로를 세 단계 스케줄로, Tree Training은 미분 가능한 트리 분할로 복원한다. [PrefixFlow §1·§3](https://arxiv.org/html/2606.01143v4), [Tree Training §3](https://arxiv.org/html/2511.00413v5)

현재 작업은 RL rollout이 아니라 독립적인 Decision→Response→Recommendation 지도 미세조정(SFT)이다. 동일 프롬프트에서 다수의 분기가 **같은 optimizer update 안**에 모이는지, 실제 토큰화 후 접두부 공유율이 얼마인지는 아직 검증되지 않았다. 사용자가 다른 GPT에서 전달받은 `90–98%`, `7924/145`, `12.6×`는 이 학습의 실측치나 ETA가 아니다. 원본 학습 문장이나 가중치는 외부로 보내지 않는다.

## 2. Background and Motivation / 기존 방식의 비용

현재 `common/core.py`는 `AutoModelForMultimodalLM`의 일반 순전파, NF4/BF16, 텍스트 디코더 LoRA(`r=16`, `alpha=32`, `dropout=0.05`), 그래디언트 체크포인팅, `use_cache=False`를 사용한다. `common/train_agent.py`는 유효 배치 16을 유지하도록 마이크로배치별 누적 횟수를 정한다. 이 경로에 **추론용 KV 캐시만 켜는 변경은 PrefixFlow가 요구하는 gK/gV 누적과 지연된 접두부 역전파를 구현하지 않는다.** [현행 `core.py`](../training/learning_first_v43/training_release_20261101_r5/common/core.py), [현행 `train_agent.py`](../training/learning_first_v43/training_release_20261101_r5/common/train_agent.py)

## 3. Design / Tree Training: 수식과 동등성 조건

Tree Training §3.1은 각 경로의 **토큰 손실 합을 경로 수 K로 나눈 목표**를 전제로 한다. 경로 k가 토큰 t를 포함할 때 토큰 t의 중복 횟수를 g_t라 하면 다음과 같다. [원문 §3.1](https://arxiv.org/html/2511.00413v5)

$$
L_{\mathrm{sep\_avg}}=\frac{1}{K}\sum_{k=1}^{K}\sum_{t\in p_k}\ell_t
=\sum_{t\in\mathrm{tree}}\frac{g_t}{K}\ell_t.
$$

여기서 $p_k$는 k번째 경로, $\ell_t$는 점수화된 토큰의 손실이다. **이는 자동으로 ‘경로별 평균 토큰 손실’이나 ‘전체 점수 토큰 평균’과 같지 않다.** 예를 들어 경로별 점수 토큰 수가 다르면, 경로별 평균은 각 경로에 `1/(K·n_k)`, 전체 토큰 평균은 `1/Σn_k`를 부여한다. 현재 `common/loss.py`는 점수 위치를 선택한 뒤 `num_items_in_batch`를 모델의 기본 손실로 전달한다. 정확한 분모·`-100` 마스크·누적 스텝의 가중치를 작은 합성 입력에서 기준 구현과 비교해야 한다. 접두부가 비점수 위치여도 접미부 손실의 접두부 경유 그래디언트는 남는다. [현행 `loss.py`](../training/learning_first_v43/training_release_20261101_r5/common/loss.py)

Tree Training §3.2는 깊이 우선 직렬화(DFS), 분기 간 주의 차단, 원래 경로의 위치 ID를 요구한다. §3.3의 **미분 가능한 KV 경계와 부모 역전파는 트리 전체가 메모리에 들어가지 않아 분할할 때** 필요하다. 현재 Gemma 4 공개 설정은 48층 중 주로 1024-토큰 슬라이딩 주의와 주기적 전역 주의, 지역/전역 헤드 차원 256/512, 서로 다른 KV 헤드 수 8/1을 사용한다. 따라서 일반적인 ‘하나의 전체 접두부 KV를 이어 붙이는’ 커널이 **두 주의 종류와 위치·윈도 경계를 모두 재현하는지** 확인해야 한다. 공개 설정이 봉인된 실제 config와 같은지도 먼저 확인해야 한다. [Tree Training §3.2–3.3](https://arxiv.org/html/2511.00413v5), [Gemma 설정](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json)

## 4. Implementation / PrefixFlow: 세 단계와 메모리

PrefixFlow §3–4는 (A) 한 번의 접두부 순전파와 층별 K/V 포착, (B) 각 접미부 마이크로배치의 역전파에서 `gK/gV` 누적, (C) 보관한 접두부 계산 그래프의 한 번의 역전파를 사용한다. 논문의 구현은 TorchTitan의 별도 `prefix_flow` 모듈과 `apply_prefixflow_ac` 진입점이다. **현재 Transformers/PEFT 러너에 해당 모듈이 설치되거나 통합되어 있다는 뜻이 아니다.** 원문의 본문에서 두 논문의 원저자 공식 GitHub 저장소 링크는 확인하지 못했다. 검색된 verl RFC/다른 공유 접두부 프로젝트도 원 논문의 즉시 사용 가능한 원본 구현으로 취급하지 않는다. [PrefixFlow §4](https://arxiv.org/html/2606.01143v4), [Tree Training §3](https://arxiv.org/html/2511.00413v5)

현재 LoRA의 드롭아웃 0.05는 핵심 반례다. 독립 경로의 동일 토큰 접두부라도 각각의 LoRA 드롭아웃 마스크가 다르면 같은 K/V를 계산하지 않는다. 이를 한 번만 계산해 공유하면 경로별 순전파·그래디언트와 **표본별 동등성이 깨진다**. 이는 [PEFT의 LoRA dropout 구현](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)에서 도출한 적용성 판단이다. 두 원문에서 이 LoRA 드롭아웃 조건의 동등성 검증은 확인되지 않았다. 드롭아웃을 0으로 바꾸는 것은 기존 학습 설정 변경이므로 해결책으로 간주하지 않는다. `(내가 의역해보자면, 글자가 같은 접두부라도 훈련 중 난수 마스크까지 같아야 '같은 계산'이다.)`

한 optimizer update 내에서 파라미터가 고정되고 동일 접두부가 모일 때만 재사용 후보가 생긴다. 유효 배치 16에서 마이크로배치가 1이면 16개 누적 구간이지만, 마이크로배치가 커지면 구간은 그에 따라 줄어든다. update가 일어나면 LoRA 가중치가 바뀌므로 K/V·접두부 그래프는 **즉시 무효화**해야 한다. 그룹 재정렬은 데이터 순서, RNG, optimizer/scheduler 상태, 체크포인트 재개 위치를 바꿀 수 있어 원래 실행과의 동등성 검증이 필요하다. PrefixFlow의 오프로드는 접두부 활성화 상태를 호스트의 pinned 메모리로 이동시키고 다시 가져오는 설계이다. 논문의 한 실험은 64 GiB pinned pool을 사용한다. **로컬 정책의 32,000,000,000바이트 GPU 예산에서 여유가 생긴다는 보장도, 2026-11-01 원격 작업의 컨테이너 RAM 16 GiB 요청에 같은 오프로드가 들어간다는 보장도 없다.** 이는 각각 [로컬 속도 설정](../training/learning_first_v43/training_release_20261101_r5/local/response_speed.json)과 [원격 작업 스냅샷 기록](2026-11-01_remote_unsloth_feasibility.md)의 수치이다. K/V, `gK/gV`, 살아 있는 접두부 그래프, 전송 버퍼, 체크포인팅 재계산을 함께 측정해야 한다. [PrefixFlow §4.2·§5.5](https://arxiv.org/html/2606.01143v4)

## 5. Evaluation / 논문 수치와 우리 수치의 분리

PrefixFlow의 8×H20·Qwen3-8B·GRPO·접두부 약 95% 실험은 200스텝 평균에서 **전체 RL 스텝 1.90×**, actor update **6.1×**이다. 별도 통제 실험은 접두부 83%에서 actor **4.4×**까지 보고한다. Tree Training은 자체 agentic 트리 실험의 SFT/RL에서 **최대 6.2×**를 보고한다. 모두 각 논문의 하드웨어·데이터·훈련 엔진에서 얻은 결과이며, 현재 Decision/P/Q 학습의 보장 배수나 ETA가 아니다. [PrefixFlow §5.3–5.4](https://arxiv.org/html/2606.01143v4), [Tree Training §4.4](https://arxiv.org/html/2511.00413v5)

## 6. Discussion and Limitations / 적용 판정

**판정: 연구 후보, 현 봉인 학습 경로에는 즉시 적용 불가.** 원인: 원문은 현재 Gemma 4 혼합 주의·QLoRA 드롭아웃·현행 손실 분모·유효 배치 16의 누적 및 전체 상태 재개의 조합을 검증하지 않았다. 결과: 기존 진행 중인 Unsloth 검토나 학습을 이 연구 때문에 중단·재포장하지 않으며, 실측 가속 수치도 주장하지 않는다.

다음 승인 게이트는 원본 데이터의 외부 전송 없이 로컬에서 (1) 토큰 ID·position·mask 기준 동일 접두부 그룹 빈도, (2) 동결된 합성 텐서로 손실/LoRA 그래디언트 기준 대비 오차와 드롭아웃 RNG 계약, (3) 봉인 모델의 지역/전역 주의 및 32 GB급 GPU·호스트 RAM 피크, (4) optimizer update/체크포인트 재개 후 캐시 무효화, (5) 실제 스텝 시간 및 총 시간의 A/B 측정이다. 하나라도 실패하면 적용하지 않는다. [PrefixFlow §6](https://arxiv.org/html/2606.01143v4), [Tree Training §3–4](https://arxiv.org/html/2511.00413v5)

## 용어 정리

- **접두부(prefix)**: 여러 학습 경로 앞부분에서 토큰과 위치·주의 문맥이 동일한 구간이다.
- **KV·gK/gV**: 주의층의 키·값 상태 및 접미부 손실에서 그 상태로 되돌아오는 그래디언트이다.
- **DFS 트리 직렬화**: 분기 노드의 토큰을 한 번만 배치하면서 형제 분기의 상호 주의를 차단하는 배열 방식이다.
- **점수 토큰(scored token)**: `labels != -100`으로 학습 손실에 직접 들어가는 위치이다.
- **오프로드(offload)**: GPU 고대역폭 메모리(HBM)의 비활성 상태를 호스트 메모리에 잠시 옮기는 것이다.
