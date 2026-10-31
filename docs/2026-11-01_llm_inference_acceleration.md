# IRC LLM 추론 가속 — 조사·접두 캐시 적용·시나리오 OFF/ON 결과

[가속 1차 조사](<IRC LLM Final 2 review and acceleration.md>) · [시나리오 회귀](2026-11-01_llm_scenario_regression.md) · [근거 파일](2026-11-01_accel_evidence/) · [구현](../src/soomac_irc/soomac_irc/prefix_cache.py)
→ **같은 모델·같은 프롬프트로 응답 시간을 줄인다. 의미 정확도는 이 문서의 범위가 아니다.**

작성 2026-11-01 KST, 브랜치 `llm_final_2`. 상태: **접두 캐시 구현·GPU 검증 완료. 사용자 결정(2026-11-01): 켜 둔다**(`PREFIX_CACHE_ENABLED = True`). 분리 T8 같은 동점 뒤집힘 위험은 알고 채택했다.

## 0. 역할 분담

| 담당 | 범위 |
| --- | --- |
| **이 문서 작성자 (Claude, 속도 최적화 담당)** | 추론 엔진·캐시·커널·양자화·VRAM. 프롬프트 내용·few-shot·학습 데이터·adapter 학습은 **수정하지 않는다** |
| 데이터 담당 (사용자 + Codex, 별도 진행 중) | 데이터 보강·프롬프트 정렬·재학습. §8의 모델 오류는 이쪽 입력 자료로만 넘긴다 |

속도 변경이 의미 결과를 바꾸면 그 차이를 이 문서에 기록하고 데이터 쪽 판정과 섞지 않는다.

## 1. 결론

1. **병목은 입력 85~97%를 차지하는 고정 system prompt를 매 호출 다시 계산하는 것**이다. FlashAttention은 이 모델·GPU에서 적용 불가하고 효과 상한도 작다.
2. Agent별 접두 KV 캐시를 HF 경로에 넣었다. 실제 시나리오에서 **턴당 10.4~14.1초 → 2.8~6.7초**, Decision 1회 5.1~7.6초 → 1.0~3.5초.
3. 사용자 필수 시나리오 4개(“바로 담아”·복합 의도)는 **캐시 OFF에서도 전부 실패**하며 기존 결과와 17턴이 완전히 같다. 실패는 캐시 문제가 아니다(§8).
4. 캐시 ON에서만 생긴 의미 차이 1건(분리 T8 “소시지만 많이 골라줘” → 소시지를 식단 제한으로 등록)이 있다. 원인은 bf16/NF4 계산 순서 차이로 거의 동점인 토큰이 뒤집힌 것이다(§7).
5. 비용: 최대 VRAM 13.5GB → 21.9GB. transformers 버그로 슬라이딩 층 KV가 잘리지 않아서다(§9).

Cause: 같은 6천~8천 토큰을 매 호출 처음부터 계산하고 있었다.

Effect: 프롬프트를 한 글자도 줄이지 않고 Decision·Response 대기 시간을 약 1/3로 줄였다. 남은 시간은 대부분 출력 생성(decode) 오버헤드다.

## 2. 환경

- GPU: NVIDIA RTX PRO 4500 Blackwell, 32623 MiB, compute capability 12.0 (SM120), driver 580.178.04
- `gemma4_env`: torch 2.12.1+cu130, transformers 5.13.1, peft 0.19.1, bitsandbytes 0.49.2, xgrammar 0.2.5. vLLM·SGLang·flash-attn·FlashInfer 미설치
- 모델: gemma-4-12B-it (`Gemma4UnifiedForConditionalGeneration`). 48층 = 슬라이딩 40층(창 1024, head_dim 256, KV head 8) + 글로벌 8층(head_dim 512, KV head 1, `attention_k_eq_v`)
- 로딩: NF4 + double quant, bf16 계산, `attn_implementation="sdpa"`. LoRA 3개(r16, q/k/v/o/gate/up/down)를 이름으로 전환한다. greedy와 XGrammar를 쓴다

## 3. 왜 느린가 — 실측

### 3.1 입력 중 고정 앞부분 비율 (실제 tokenizer, 로그의 messages 기준)

| Agent | 전체 입력 | 매 호출 동일한 앞부분 |
| --- | ---: | ---: |
| Decision | 6,209~6,481 | 6,040 (93~97%) |
| Recommendation | 3,832 | 3,338 (87%) |
| Response task | 5,962~6,611 | 5,280 (80~89%) |
| Response general | ~2,567 | 1,162 |

system prompt는 import할 때 고정되는 상수다([agent_prompts.py:249-253](../src/soomac_irc/soomac_irc/agent_prompts.py#L249-L253)). 호출마다 바뀌는 history·state·발화는 그 뒤 user message에만 들어간다. 이후 프롬프트 v2에서 예시가 늘어 시나리오 시험 당시 Decision 입력은 8,092~8,370, Response는 7,574~8,124토큰이었다.

### 3.2 Decision 1회(입력 6.3k, 출력 약 20) 시간 분해 — 추정

| 구간 | 시간 | 근거 |
| --- | ---: | --- |
| prefill Linear | ~0.9초 | 행렬곱 벤치. bf16 864ms, NF4 921ms(계산 바운드라 차이 없음) |
| prefill Attention | ~0.75~0.9초 | SDPA 벤치(§4.1) |
| prefill 기타 | ~0.8~1.2초 | **원인 미확인**(LoRA·PEFT·elementwise 추정) |
| decode | ~75ms/토큰 | 로그 78건 기울기. 행렬곱만 재면 NF4 14ms/토큰이므로 나머지 약 60ms는 파이썬·커널 실행 오버헤드로 추정 |

계측기로 직접 나눈 값이 아니라 로그 회귀와 마이크로벤치를 조합한 추정이다(Codex: 조건부).

## 4. 가속 후보 조사

### 4.1 FlashAttention — 적용 불가, 효과 상한 작음

- FA2/FA3는 head_dim 256까지다. FA4는 SM120에서 공유메모리 99KB 제약이 있어 D=512 타일이 들어가지 않는다. **글로벌 8층(head_dim 512)은 어떤 FlashAttention으로도 SM120에서 돌릴 수 없다.**
- PyTorch SDPA 실측(L=6300, bf16): head 256 causal은 FLASH 2.6ms. 하지만 HF는 슬라이딩 층에 mask를 넘기므로 FLASH가 거부되고 EFFICIENT 16.2ms가 쓰인다. head 512는 FLASH가 거부되고("head dim ≤ 256"), EFFICIENT 14~31ms다. cuDNN은 전부 불가다.
- 결과적으로 48층 전부가 mem-efficient 커널을 쓴다(MATH 폴백은 아님). 슬라이딩 층은 창 1024가 아니라 6k×6k 전체를 계산한다.
- 어텐션을 완전히 없애도 Decision 1회의 약 17~21%다(Codex 찬성).

### 4.2 엔진·양자화 후보 (우선순위)

| 순위 | 후보 | 예상 Decision 1회 | 상태 |
| --- | --- | --- | --- |
| 1 | HF + Agent별 접두 KV 캐시 | 실측 1.0~3.5초 | **적용 완료(§5)** |
| 2 | vLLM ≥0.23 (multi-LoRA, LoRA를 구분하는 prefix caching, CUDA graph, xgrammar) | 0.4~0.8초 **추정** | 미착수 |
| 3 | W4A16 Marlin 기반(직접 GPTQ 또는 Google `gemma-4-12B-it-qat-w4a16-ct`) | decode 이론치 ~9ms/토큰 | 2번 이후 |
| 4 | MTP speculative decoding (`gemma-4-12B-it-assistant`) | decode 약 2배(커뮤니티 보고) | 마지막. LoRA·grammar 조합 미확인 |
| 제외 | FP8 | — | 사용자 결정(이전 시험에서 더 느렸음) |
| 보류 | NVFP4 W4A4 | — | 텍스트로만 교정하면 이미지 입력이 깨진다는 커뮤니티 보고. 로컬 근거는 없음. VLM을 같이 쓰므로 후순위 |

### 4.3 vLLM 전환 시 확인된 함정

1. **k_eq_v LoRA 불일치**
   - 현상: HF 글로벌 층은 V를 `k_proj` 출력으로 만든다([modeling_gemma4_unified.py:442](../../miniconda3/envs/gemma4_env/lib/python3.10/site-packages/transformers/models/gemma4_unified/modeling_gemma4_unified.py), 환경 내부 경로). 그래서 `k_proj` LoRA가 K와 V를 둘 다 바꾼다. vLLM `gemma4.py`(main)는 기본 가중치를 K·V 양쪽에 넣지만(110, 555행), LoRA는 `packed_modules_mapping` q/k/v 슬롯을 따른다(1442행). 그래서 V에는 LoRA delta가 들어가지 않을 가능성이 높다.
   - 어댑터 확인: 글로벌 층(5, 11, …)에 `v_proj` 텐서가 없다.
   - 해결안(재학습 불필요): 글로벌 8층의 `k_proj` lora_A/B를 `v_proj`로 복제한 변환본을 쓴다. Codex는 수식상 정확하다고 판정했다. 런타임 로드는 미검증이다.
2. Gemma4Unified의 LoRA 지원은 vLLM 문서 표에 표시가 없다. 코드상으로는 `SupportsLoRA`가 상속돼 있다. 실제로 돌려 봐야 안다.
3. XGrammar `any_order`를 vLLM이 JSON schema 옵션으로 받지 않는다. 대안은 `xgr.Grammar.from_json_schema(..., any_order=True)`를 EBNF 문자열로 넘기는 것이다. 로컬 변환은 확인했고 vLLM 안에서는 미확인이다.
4. 어댑터는 NF4 위에서 QLoRA로 학습됐다. 다른 기반 가중치(bf16·QAT·GPTQ)에 붙이면 출력이 바뀔 수 있으므로 의미 회귀가 필수다.

## 5. 적용 — Agent별 접두 KV 캐시

### 5.1 변경 파일

- 신규 [prefix_cache.py](../src/soomac_irc/soomac_irc/prefix_cache.py): `PrefixKVCache`
- [decision_model.py](../src/soomac_irc/soomac_irc/decision_model.py), [recommendation_model.py](../src/soomac_irc/soomac_irc/recommendation_model.py), [response_model.py](../src/soomac_irc/soomac_irc/response_model.py): `model.generate(...)`를 `prefix_cache.generate(...)`로 교체. trace에 `prefix_cache` 상태(build/hit/miss/off)를 기록
- 프롬프트·few-shot·adapter·graph·policy는 변경 없음

### 5.2 동작

1. 첫 호출: system + 자리표시 user를 렌더링해 실제 입력과 토큰이 처음 갈라지는 지점까지를 접두로 잡는다. 그 접두를 현재 활성 Adapter로 한 번 forward해 KV를 저장한다. 이때 `logits_to_keep=1`이라 전체 logits는 만들지 않는다.
2. 이후 호출: `input_ids` 앞부분이 저장본과 토큰 단위로 완전히 같을 때만 저장본의 deepcopy를 `past_key_values`로 넘긴다. 다르면 종전 경로로 간다.
3. **전체 `input_ids`·`attention_mask`를 넘긴다.** generate가 캐시 길이만큼 잘라 나머지만 계산하고, position은 전체 mask 기준으로 잡힌다(transformers `generation/utils.py:3876-3886`, `:732-735`).
4. key는 (Adapter 이름, sha256(system)) → D / Q / P-task / P-general / P-mixed가 각자 따로 둔다. KV는 Adapter 가중치에 의존하므로 섞지 않는다.
5. 끄기: `PREFIX_CACHE_ENABLED = False`. 생성 시점에 모듈 값을 읽으므로 시험 스크립트에서 모듈 값만 바꿔 OFF/ON을 비교할 수 있다.

### 5.3 안전 조건 (코드 주석에 명시)

- 프로세스 안에서 Adapter를 다시 로드하면 `clear()`가 필요하다. 현재 코드에는 재로드 경로가 없다(노드 시작 시 1회, [llm_langgraph_node.py:99](../src/soomac_irc/soomac_irc/llm_langgraph_node.py#L99)).
- 잠금이 없다. 현재처럼 워커 스레드 하나에서 순차 호출할 때만 안전하다. VLM도 같은 작업 큐 스레드에서 돈다.
- 가짜 모델(`nn.Module`이 아님)과 batch>1이면 캐시를 우회한다. 단위 테스트 894개 통과(rclpy가 필요한 STT 테스트 2개는 변경 전부터 수집 오류).

## 6. 검증 1 — 로그 실제 입력 재생 (29건, 각 OFF→ON)

| Agent | 캐시 적중 시 OFF | ON |
| --- | ---: | ---: |
| Decision | 4.64초 | 1.70초 |
| Response task | 4.94초 | 2.27초 |
| Recommendation | 4.84초 | 3.35초 |
| Response general | 1.94초 | 1.49초 |
| 전체 적중 24건 평균 | 4.42초 | 2.05초 |

- Recommendation은 출력이 약 57토큰이라 decode 비중이 커서 덜 줄었다.
- 토큰 완전 일치는 21/29. 다른 8건 중 Decision 3건은 키 순서만 다르거나 의미가 다르다(§7).
- **VLM**: 캐시를 만들기 전과 후에 같은 이미지·프롬프트를 넣었더니 출력이 글자까지 같았다. VLM 호출은 캐시를 쓰지 않는다. 합성 이미지 1건 확인이며 VLM 회귀 전체를 대신하지 않는다.

## 7. 검증 2 — 출력 차이는 버그인가 (다음 토큰 logits 비교)

| 비교 | logits 최대 차이 |
| --- | ---: |
| 전체 한 번 계산 vs 같은 계산 반복 | 0.0000 |
| 전체 vs **마지막 1토큰만 나눠 계산**(캐시 코드 미사용) | 0.61~0.77 |
| 전체 vs 접두 캐시 이어 계산 | 0.81~1.53 |

- 4건 모두 첫 토큰 1위 후보는 같았다.
- 원인: NF4·bf16 행렬곱은 한 번에 계산하는 행 수에 따라 커널과 누적 순서가 달라진다. 지금 운영 중인 `generate()`도 decode 단계마다 같은 크기의 차이를 겪는다.
- 결과: 캐시 OFF 출력이 "정답"인 것이 아니다. 거의 동점인 지점에서는 어느 쪽으로든 갈릴 수 있다. vLLM·커널·양자화 변경도 같은 종류의 차이를 만든다. **채택 판정은 토큰 일치가 아니라 의미 시나리오 회귀로 한다**(Codex 찬성).

## 8. 검증 3 — 사용자 시나리오 OFF/ON

하네스: 기존 [manual_scenario_stack_probe.py](../src/soomac_irc/test/manual_scenario_stack_probe.py) `--measure-all`을 수정 없이 사용했다. 실제 state·history를 누적하고, 정답 주입·강제 commit·강제 단계 전환은 없다. 물리 완료만 offline fixture로 처리한다. 래퍼는 [run_scenario_cache.py](2026-11-01_accel_evidence/run_scenario_cache.py)이고, 턴별 3자 비교는 [compare.txt](2026-11-01_accel_evidence/compare.txt)에 있다.

비교 대상은 셋이다.
- `prev`: 기존 실행 `/tmp/soomac_closed_loop_all_20261101_v2/`. 고정 해시 5개 파일이 현재와 동일하다
- `off`: 캐시 끔
- `on`: 캐시 켬

### 8.1 재현성

`prev`와 `off`는 **17턴 모두 Decision 원문·주문·실행 여부·응답이 같다.** 아래 실패는 우연이 아니라 현재 모델·코드의 고정된 결과다.

### 8.2 복합 6턴 — OFF와 ON 판정 동일

| 턴 | 발화 | 결과 | 실패 지점 |
| --- | --- | --- | --- |
| 1 | 얇은면 보통 + 느끼한 소스 추천 | 면 ✅ / 추천 ❌ | D가 `recommendation:request`를 내지 않음 → Q 미호출 → 크림 후보 pending 없음. P가 말로만 크림을 추천 |
| 2 | 크림으로 선택할게. 바로 담아. | ❌ | D가 `commit:true`만 냄. sauce가 null이라 실행 차단([llm_policy.py:585](../src/soomac_irc/soomac_irc/llm_policy.py#L585), [llm_langgraph.py:960](../src/soomac_irc/soomac_irc/llm_langgraph.py#L960)). **P는 "소스는 크림으로 선택했어요"라고 사실과 다르게 말함** |
| 3 | 양파랑 버섯 둘 다 적게 | 주문 ✅ | low/low, 실행 안 함. 로봇 단계는 noodle에 멈춤 |
| 4 | 버섯만 보통으로 바꿔서 바로 담아 | 주문·commit ✅ / 실행 ❌ | 모델 처리는 맞음(버섯 normal, 양파 low 보존, commit). 2턴의 연쇄로 차단 |
| 5 | 게살 알레르기 + 소시지 많이 바로 담아 | 주문 ✅ / commit ❌ | allergy·소시지 high 맞음. **commit 누락(독립 오류)** |
| 6 | 페퍼론치노 빼고 치즈 적게 바로 담아 | 주문 ✅ / commit ❌ | 치즈 low 맞음. **commit 누락(독립 오류)** |

분리 대조 11턴도 같은 이유로 실패한다. T2 "크림으로 선택할게"에서 D가 pending 없이 `confirmation:accept`만 내 sauce가 null이고, P는 크림을 선택했다고 거짓으로 말한다. 그 뒤 모든 "바로 담아"는 소스 누락으로 차단된다.

**데이터 담당에게 넘기는 관측(속도 담당은 수정하지 않음)**
- (a) 1턴 D의 추천 요청 누락
- (b) 2턴과 분리 T2에서 크림 선택이 주문에 들어가지 않음, 그리고 그때 P의 허위 확정 발화
- (c) 5·6턴 D의 commit 누락

(b)를 고쳐도 (c)는 남는다.

### 8.3 캐시 때문에 달라진 것

| 위치 | OFF | ON | 판정 |
| --- | --- | --- | --- |
| 분리 T8 "소시지만 많이 골라줘" | D: 소시지 high + 게살 allergy 재기록 | D: 소시지 high + **소시지/dietary_rule 제한** → restriction_conflict, 소시지 제거 | **캐시 ON에서만 생긴 오류.** T9~T11은 이 연쇄 |
| 복합 T5, 분리 T2·T3·T5·T6·T7·T10 | — | 응답 문장 표현만 다름(예: "오일, 토마토, 크림 중 어떤 소스로 할까요?") | 주문·실행 필드 동일. 분리 T2의 허위 크림 확정은 OFF에도 있는 기존 오류 |

복합 6턴의 Decision 원문은 OFF와 ON이 6턴 모두 같았다.

### 8.4 속도·VRAM (시나리오 실측)

| | OFF | ON |
| --- | ---: | ---: |
| 한 턴(D+P) | 10.4~14.1초 | 2.8~6.7초 (Agent 첫 호출만 13초대) |
| Decision 1회 | 5.1~7.6초 | 1.0~3.5초 |
| Response 1회 | 5.3~7.1초 | 1.8~3.5초 |
| 최대 VRAM | 13.5GB | 21.9GB |

## 9. 발견한 transformers 버그 — 슬라이딩 층 KV가 잘리지 않음

- `cache_utils.py:1566`: `layer_types = layer_types[: -decoder_config.num_kv_shared_layers]`. Gemma4는 이 값이 0이므로 `[:-0]` → 빈 리스트가 되고, 모든 층이 `DynamicLayer`(전체 저장)로 대체된다.
- **운영 중인 `generate()`에도 같은 버그가 걸려 있다.** 캐시를 넣어서 새로 생긴 문제가 아니다.
- 영향: 접두 캐시 하나가 Decision 기준 약 2.1GB다(슬라이딩 40층 × 6040토큰). 슬라이딩 층이 1023토큰만 남기도록 우회하면 약 0.43GB로 추정되고, decode 때 어텐션 길이도 줄어든다.
- 미적용 상태다. 우회하면 계산 경로가 또 바뀌므로 §8 같은 OFF/ON 회귀를 다시 해야 한다.
- **후속 확인(§14)**: 5.14.0에서 수정됐다. 하지만 업그레이드만으로는 실제 메모리가 줄지 않는다(view 문제). 5.14.1 + 저장 직후 `clone()`으로 Decision 캐시가 1.97 → 0.44 GiB가 되는 것을 실측했다.

## 10. Codex 교차검증 요약

| 대상 | 판정 |
| --- | --- |
| 조사 주장 8개 | 찬성 2(FlashAttention 효과 작음, 순서), 조건부 6, 반대 0. 시간 분해·예상 배속은 추정이라고 명시하도록 지적 |
| 캐시 구현·벤치 | 수치 차이·transformers 버그·속도 찬성. 구현은 "수정 필요"(어댑터 재로드·동시 호출) → 현재 코드에 해당 경로가 없어 조건을 주석으로 명시 |
| 시나리오 분류 | 4개 중 3개 찬성. "ON 응답이 전부 사실과 일치"라는 문구는 반대 → §8.3처럼 정정 |

Codex 원문: `training/learning_first_v43/final/goals/GL10/accel_prefix_cache_20261101/codex_*.txt`

## 11. 남은 일 — 속도 담당

| 순서 | 작업 | 이유 | 예상 |
| --- | --- | --- | --- |
| 1 | ~~캐시 ON 운영 여부 결정~~ | **완료: 켜 둠**(사용자 결정 2026-11-01) | — |
| 2 | "~만" 표현 등 동점 취약 발화를 OFF/ON 반복 | T8이 캐시 고유 위험인지, 모델 자체가 불안정한지 분리 | 30분 |
| 3 | 슬라이딩 KV 1023 우회(§9) | VRAM 21.9 → 약 14GB대 추정, decode 단축 가능 | 반나절 + 회귀 |
| 4 | prefill/decode 직접 계측(GPU event) | §3.2 추정을 실측으로 교체 | 1~2시간 |
| 5 | vLLM PoC: k_eq_v adapter 변환 → LoRA 로드 → EBNF any_order → 의미 회귀 | decode 오버헤드(약 60ms/토큰) 제거 | 2~4일 |

데이터 쪽 수정으로 프롬프트가 바뀌어도 캐시는 sha256(system)으로 자동 분리되므로 별도 작업이 필요 없다. 다만 프롬프트가 길어지면 캐시 VRAM도 비례해서 늘어난다.

## 12. 작업 기록상 주의

- 테스트 비교를 위해 `git stash`/`pop`을 1회 실행했다. 당시 작업트리에는 다른 세션의 미커밋 변경 13개 파일이 있었다. stash는 비어 있고 충돌 없이 전부 복원됐음을 확인했다. 이후 이 방식은 쓰지 않는다.
- `~/.codex/config.toml`의 model은 `gpt-6.1-sol`인데 계정이 거부한다. 이번 교차검증은 모두 `-m gpt-5.6-sol`로 실행했다.

## 13. 근거 파일

- 저장소: [2026-11-01_accel_evidence/](2026-11-01_accel_evidence/)
  - `local_evidence.md`: 환경·SDPA·Linear 실측
  - `bench_rows.json`: §6 29건
  - `diag_clean.log`: §7
  - `compare.txt`: §8 17턴 3자 비교
  - 실행 스크립트 4개. scratchpad 경로가 하드코딩돼 있으니 재실행할 때 경로를 바꾼다
- 원본(git 제외): `training/learning_first_v43/final/goals/GL10/accel_prefix_cache_20261101/`
  - `scn_off/`, `scn_on/`: turns.jsonl, PLAN, IDENTITY, RESULT, EXIT
  - `bench.log`, `diag.log`, `cache.diff`, Codex 프롬프트·출력

Cause: 속도 변경과 의미 수정이 동시에 진행되면 어떤 변화가 어느 쪽 때문인지 섞인다.

Effect: 이 문서는 캐시 OFF가 기존 결과와 같다는 기준선, ON에서 생긴 차이 1건, 속도·VRAM 수치를 분리해 둔다. 데이터 담당의 수정 후에도 같은 하네스로 OFF/ON을 다시 비교할 수 있다.

## 14. 라이브러리 버전 팩트체크와 도입 가능성 (2026-11-01 추가)

사용자가 외부 분석 2건(vLLM 도입 검토, transformers 업그레이드로 캐시 메모리 해결)을 받아 사실 확인을 요청했다. 사용자의 의도는 **이해하기 어려운 커스텀 캐시 대신 검증된 엔진(vLLM)을 쓰는 것**이다. 근거 파일은 [2026-11-01_accel_evidence/](2026-11-01_accel_evidence/)에 있다(`cache_layers_check.py`, `cache_inspect.py`, `bench_with_tf.py`, `bench_tf51{31,41}.log`, `rows_tf51{31,41}.json`).

### 14.1 외부 분석 주장 → 확인 결과

| 주장 | 판정 | 근거 |
| --- | --- | --- |
| 5.13.x는 `num_kv_shared_layers=0`이면 `[:-0]`으로 캐시 층 목록이 사라진다 | **사실** | 원본 `cache_utils.py` v5.13.1:1565-1566. 로컬 CPU 실행에서 `DynamicCache(config)`가 **0층**(나중에 전부 일반 층으로 채움) |
| 설정을 안 넘겨서가 아니라 라이브러리 결함이다 | **사실** | 모델이 직접 `DynamicCache(config=self.config)`를 만든다(modeling_gemma4_unified.py:646) |
| 5.14.0/5.14.1에서 수정됐다(PR #47118) | **사실** | v5.14.0·v5.14.1:1633-1635에 `> 0` 조건 추가. PR #47118 "[cache] Simplify cache dispatch based on layer_types", 2026-07-07 병합. 5.14.1 CPU 실행에서 48층 = 슬라이딩 40 + 전체 8 |
| **업그레이드하면 메모리 문제가 해결될 가능성이 높다** | **틀림(실측)** | 아래 14.2 |
| 직접 KV를 자르는 방식은 피하라 | 동의 | 라이브러리의 슬라이딩 층을 그대로 쓰고, 이미 잘라 둔 결과를 복사만 하는 방식을 제안(14.3) |
| PyTorch 버전은 무관 | 동의 | 원인은 transformers 코드다 |
| 5.17·5.20의 추가 개선 | 미확인 | 이번에 확인하지 않음 |
| (vLLM 분석) 커스텀 캐시 최대 메모리 16.4GB | 틀림 | 16.4GB는 로그 재생 벤치 값. 실제 시나리오는 21.9GB(§8.4) |
| (vLLM 분석) vLLM에서도 NF4 bitsandbytes 유지 | 조건부 | 동작은 하지만(LoRA 예제 있음) 느린 경로다. 동일성 비교 기준용으로만 쓴다. 이 문서의 이전 판단("bnb+LoRA 미지원")은 틀려서 정정한다 |

### 14.2 업그레이드만으로는 메모리가 줄지 않는다 — 실측

입력 6,040토큰을 접두로 계산한 직후 캐시 1개의 크기를 쟀다.

| 환경 | 슬라이딩 층 저장 길이 | 장부상 크기 | 실제 GPU 메모리 증가 |
| --- | ---: | ---: | ---: |
| 5.13.1 (현재) | 6,040 | 1.935 GiB | 1.968 GiB |
| 5.14.1 | 1,023 | 0.404 GiB | **1.967 GiB** |
| 5.14.1 + 저장 직후 `clone()` | 1,023 | 0.404 GiB | **0.444 GiB** |

- 원인: 5.14.1의 `DynamicSlidingWindowLayer.update`는 `full_key_states[:, :, -1023:, :]`라는 **view**를 저장한다(v5.14.1:231-232). view가 살아 있으면 원본 6,040개 메모리도 해제되지 않는다. 최신 main에도 같은 코드가 있다(main:245-246).
- 추가 영향: PyTorch `deepcopy`는 view의 원본 저장소 전체를 복사한다(torch `_tensor.py:182, 242`, Codex 확인). 그래서 지금은 캐시가 적중할 때마다 큰 복사가 일어난다.
- 로그 재생 14건 비교(5.13.1 → 5.14.1, 캐시 ON): 최대 VRAM 16.20 → 16.20 GiB(동일). 캐시 적중 평균 1.845 → **1.646초**(약 11% 빨라짐). PEFT·NF4·XGrammar·VLM 모두 정상 동작했다.

### 14.3 메모리 해결안 — transformers 5.14.1 + 1줄

`PrefixKVCache._build`에서 접두 forward를 마친 직후, 각 층의 keys/values가 view일 때만(`tensor._base is not None`) `clone()`으로 바꾼다.

- 직접 자르는 게 아니다. 라이브러리가 이미 잘라 둔 1,023개와 `cumulative_length`를 그대로 두고 메모리만 실제 크기로 만든다(Codex: 안전).
- 5.13.1에서는 view가 아니라서 아무 일도 하지 않게 조건을 둔다.
- 기대 효과(추정): 캐시 4~5개 합계 약 8GB → 약 2GB, 시나리오 최대 VRAM 21.9GB → 약 15GB대. 적중 시 복사량도 줄어 약간 빨라진다.
- **아직 적용하지 않았다.** `gemma4_env` 업그레이드 승인이 필요하기 때문이다.

### 14.4 버전 현황 — 학습과 실행이 다르다

| | torch | transformers | peft | bitsandbytes | Python |
| --- | --- | --- | --- | --- | --- |
| 학습 도커 `soomac/full-agent-v4-triple:v2.9` | 2.8.0+cu128 | 5.13.0 | 0.21.0 | 0.50.2 | — |
| 실행 `gemma4_env` | 2.12.1+cu130 | 5.13.1 | 0.19.1 | 0.49.2 | 3.10.20 |
| vLLM 0.32.0 요구 | ==2.13.0 | ≥5.16.1, <5.20 | — | — | ≥3.11 |

- PEFT 차이 때문에 실행 시 `kasa_config / monteclora_config / velora_config 무시` 경고가 나온다. 어댑터 설정에서 세 값은 모두 null이라 영향은 없다.
- bnb 0.49.2 ↔ 0.50.2가 출력에 영향을 주는지는 **미증명**이다. 맞추는 게 바람직하지만 우선순위는 낮다.
- STT 노드(`stt_qwen3_asr_node.py`, `stt_nemotron_node.py`)도 `gemma4_env`를 쓴다. transformers를 올리면 STT도 같이 바뀐다.

### 14.5 도입 가능성 판단

| 안 | 가능성 | 하는 일 | 위험 | 예상 시간 |
| --- | --- | --- | --- | --- |
| **A. transformers 5.14.1 + clone 1줄** | 높음(부분 실측 완료) | 임시 폴더로 시험 → 17턴 시나리오 OFF/ON + STT smoke → `gemma4_env` 업그레이드 | 출력이 미세하게 바뀔 수 있어 회귀 필수. STT 영향 | 반나절 |
| **B. vLLM 별도 환경** | 중간(미실측) | Python 3.11 새 conda 환경 → vLLM 0.32 → Decision LoRA(k_eq_v 변환) → 로그·17턴 비교 | Gemma4Unified LoRA 미검증, k_eq_v, any_order(EBNF), 디스크 여유 37GB | 1~2일 |

- A는 메모리 문제를 해결하고 지금 구조를 유지한다. 커스텀 캐시(약 100줄)는 남는다.
- B는 사용자가 원하는 "검증된 엔진"이다. 캐시를 vLLM이 관리하고, decode 오버헤드(약 60ms/토큰)도 줄일 수 있다. 다만 지금 캐시 ON(≈2초)보다 빨라야 채택한다.
- A와 B는 서로 막지 않는다. A는 지금 운영 경로의 메모리를 바로 줄이고, B의 비교 기준(캐시 ON) 역할을 한다.

Cause: 외부 분석은 원인(5.13.x 버그)은 맞혔지만, 수정 버전에 남아 있는 view 메모리 문제는 놓쳤다.

Effect: "업그레이드만 하면 된다"가 아니라 "업그레이드 + 1줄"이라는 실측 근거가 생겼다. vLLM은 별도 환경이 필수라는 조건(Python·torch·transformers·xgrammar 고정)이 확정됐다.

## 15. PyTorch 버전 팩트체크와 현재 로직의 라이브러리 충돌 (2026-11-01 추가)

외부 분석("PyTorch 2.15로 올리면 빨라지나")을 PyPI 메타데이터, PyTorch 공식 릴리스 노트(GitHub), 로컬 `pip check`·import 실행으로 확인했다. Codex 교차검증 결과는 찬성 3 / 조건부 3 / 반대 0이다(`codex_env.txt`).

### 15.1 PyTorch 주장 → 확인 결과

| 주장 | 판정 | 근거 |
| --- | --- | --- |
| 2.15.0이 2026-10-28 정식 출시 | 사실 | PyPI·GitHub 릴리스 |
| 2.15는 Python 3.11 이상 | 사실 | `requires_python >=3.11`. cp310 wheel 없음 |
| 2.15 CUDA wheel은 13.2/13.4 | 사실 | cu132·cu134만 있고 cu130은 없음. Triton ~=3.9 |
| CUDA 13.2는 R595 드라이버가 필요 | **틀림(PyTorch wheel 기준)** | 2.15 릴리스 노트: "CUDA 13 wheels need NVIDIA driver 580 or newer". 현재 580.178.04 |
| 2.14.1은 CUDA 13.0 wheel이 있음 | 사실 | cu130·cp310 wheel 존재(2026-09-30). **Python 3.10 환경에 깔 수 있는 최신 torch** |
| 2.13 FlexAttention 최대 12배는 Apple Silicon 한정 | 사실 | 릴리스 노트: MPS, 희소 패턴 기준 |
| 2.13 CuTeDSL Inductor | 사실 | 블로그 분류는 "API Unstable"(Codex 정정) |
| 2.14 NVGEMM, 2.15 NVFP4 decode 가속 | 사실, **우리 경로엔 해당 없음** | Inductor(torch.compile)와 NVFP4 가중치가 전제다. 우리는 컴파일하지 않고 bnb NF4를 쓴다 |
| (추가 확인) 2.14 mem-efficient SDPA의 GQA 지원 | 효과 없음 | HF는 mask가 있으면 여전히 KV를 직접 복제한다(`sdpa_attention.py:28-38`) |
| (추가 확인) 2.14 cuDNN SDPA head 256 | 효과 없음 | SM90/SM10.x 전용이고 우리는 SM120 |

결론: torch를 올려서 얻을 자동 가속은 근거가 없다. 측정 가치가 있는 건 2.14.1(C)뿐이고, 2.15(D)는 Python 3.11이 전제라 ROS 노드에서 쓸 수 없다(15.2).

### 15.2 프로세스·환경 구조 — 가장 큰 제약

| 프로세스 | Python 환경 | 비고 |
| --- | --- | --- |
| LLM 노드(`llm_langgraph_node.py`), STT Nemotron·Qwen3-ASR | `gemma4_env` (Python 3.10.20) | `rclpy`를 `/opt/ros/humble/...python3.10`(ROS Humble, CPython 3.10 바이너리)에서 import |
| TTS·UI | `cosyvoice3` | 별도 |
| Clova STT(`stt_node`) | `ros_env` | 별도 |

- 원인: 현재 ROS Humble 바이너리는 Python 3.10용이다.
- 결과: **Python 3.11 이상이 필요한 torch 2.15와 vLLM 0.32는 LLM 노드와 같은 프로세스에서 돌 수 없다.** vLLM은 별도 서버 프로세스로 띄우고 노드가 HTTP 등으로 호출하는 구조가 필수다(ROS를 3.11로 재빌드하지 않는 한).
- `gemma4_env`의 transformers를 올리면 STT 두 노드(`AutoModelForRNNT`, `AutoModelForMultimodalLM`)도 영향을 받는다. smoke test가 필수다.

### 15.3 이미 있는 버전 충돌·불일치

| 항목 | 상태 | 영향 | 조치 |
| --- | --- | --- | --- |
| xgrammar 0.2.5 ↔ transformers 5.13.1 | `pip check` 충돌: 0.2.5 메타데이터가 transformers<5를 요구 | 실제로는 동작함(테스트·GPU 실행 정상) | 0.2.8은 상한을 없앰(≥4.38). A에서 같이 올리고 `any_order` 회귀 확인 |
| torchaudio 2.11.0 ↔ torch 2.12.1 | 짝 불일치 | import는 성공. `src`에서 사용처 0개 | 2.12.x로 맞추거나 제거 후보. 낮은 우선순위 |
| peft 0.19.1(실행) ↔ 0.21.0(학습) | 실행 쪽이 더 오래됨 | 모르는 설정 필드 3개를 무시한다는 경고. 값이 모두 null이라 무해 | 0.21.x로 맞추기 후보 |
| bitsandbytes 0.49.2(실행) ↔ 0.50.2(학습) | 실행 쪽이 더 오래됨 | 출력 영향은 미증명 | 맞추기 후보. 회귀로 확인 |
| torch 2.12.1+cu130(실행) ↔ 2.8.0+cu128(학습) | 다름 | QLoRA 학습과 추론 사이의 수치 차이 가능성(미증명) | 학습 환경은 데이터 담당 소관이라 기록만 |
| vLLM 0.32 요구 | Python ≥3.11,<3.15, torch==2.13.0, transformers ≥5.16.1,<5.20, xgrammar==0.2.8 | `gemma4_env`와 전부 충돌 | 별도 환경 + 서버(B) |

### 15.4 실행 순서 (Codex 권장과 일치)

| 순서 | 안 | 환경 | 내용 | 성공 기준 |
| --- | --- | --- | --- | --- |
| 1 | **A** | 임시 오버레이로 시험 → `gemma4_env` | transformers 5.14.1 + 캐시 저장 직후 `clone`(view일 때만). xgrammar 0.2.8, peft 0.21.x 정렬 검토 | 17턴 시나리오 의미 동일 또는 기존 수준, VRAM 감소, STT 2개 노드 smoke 통과 |
| 2 | C | `gemma4_env` 복제본 | torch 2.14.1+cu130 성능만 비교(나머지 고정) | 같은 입력에서 지연 감소 + 의미 동일. 없으면 채택하지 않음 |
| 3 | **B** | 새 conda 환경(Python 3.12) + vLLM 0.32 서버 | k_eq_v 어댑터 변환 → Decision LoRA → 3개 → 17턴 | 캐시 ON(약 2초)보다 빠르고 의미 동일 |
| 4 | D | B와 같은 별도 환경에서만 | torch 2.15 | B가 채택된 뒤에만 검토 |

torch.compile은 LoRA 전환과 입력 길이 변화로 재컴파일 위험이 있어 별도 실험으로 분리한다.

Cause: 버전을 올리는 것만으로 빨라진다는 근거가 없고, ROS가 Python 3.10에 묶여 있다.

Effect: 같은 환경 안에서는 A(+C)만 가능하다. vLLM과 최신 torch는 "별도 서버"라는 구조 변경과 한 묶음으로 다뤄야 한다.

## 16. A 실행 — 새 환경 `soomac_llm` + 캐시 메모리 수정 (2026-11-01)

사용자 결정: 원래 `gemma4_env`는 그대로 두고, **새로 설치한 환경**으로 LLM 노드만 옮긴다. 처음엔 복제본(`gemma4_env_v2`)으로 시작했지만 사용자 지시로 새로 설치하는 방식으로 바꿨고, 복제본은 삭제했다. STT·TTS는 각자 기존 환경(`gemma4_env`, `cosyvoice3`, `ros_env`)을 계속 쓴다.

### 16.1 새 환경 구성

- 이름 `soomac_llm`, Python 3.10.20(ROS Humble `rclpy`와 호환), 5.5GB, 패키지 150개(기존 363개). `pip check` 충돌 0건(기존 1건).
- LLM 노드가 실제로 import하는 외부 라이브러리 12개만 설치했다(AST로 추출).
- 버전: torch 2.12.1+cu130(유지), **transformers 5.14.1, peft 0.21.0, bitsandbytes 0.50.2, xgrammar 0.2.8**(학습 환경과 맞추거나 버그가 수정된 버전). 나머지는 기존과 같다(accelerate 1.14.0, chromadb 1.5.9, jsonschema 4.26.0, langgraph 1.0.8, numpy 2.2.6, pillow 12.3.0, pytest 9.1.1).
- 패키지 목록: `2026-11-01_accel_evidence/soomac_llm_freeze.txt`. 기존 환경 목록: `gemma4_env_freeze_before.txt`.
- 검증: 노드 전체 모듈과 ROS(`rclpy`, `sensor_msgs`, `std_msgs`) import 성공, 단위 테스트 896개 통과.

### 16.2 캐시 메모리 수정 — 첫 시도 실패와 원인

- 첫 구현은 `tensor._base is not None`(view 여부)일 때만 clone했다. 그런데 메모리가 전혀 줄지 않았다(13.00GiB 그대로).
- 원인: 이 코드는 `torch.inference_mode()` 안에서 돈다. 그 모드에서는 view의 `_base`가 기록되지 않아(`cache_debug.py` 실측: `_base None`인데 저장소는 6,040토큰 전체 크기) 조건이 한 번도 참이 되지 않았다.
- 수정: "저장소 크기 > 실제 내용 크기"일 때만 연속 메모리로 clone한다([prefix_cache.py](../src/soomac_irc/soomac_irc/prefix_cache.py) `_build`). 저장소와 내용 크기가 같으면 아무것도 하지 않는다.
- 결과: Decision 접두 캐시 1.967 → **0.445 GiB**.

### 16.3 같은 코드·프롬프트로 기존 환경 vs 새 환경 (둘 다 캐시 ON)

그사이 데이터 담당이 프롬프트 파일 2개(`agent_prompts.py`, `prompt_examples.py`)를 바꿨기 때문에, 예전 결과가 아니라 **현재 코드로 두 환경을 연달아** 돌려 비교했다.

| 항목 | `gemma4_env` (5.13.1) | `soomac_llm` (5.14.1 + 수정) |
| --- | ---: | ---: |
| 로그 재생 14건: 캐시 적중 평균 | 1.914초 | **1.490초** |
| 로그 재생: 최대 / 캐시 생성 후 VRAM | 16.20 / 13.00 GiB | **12.69 / 9.46 GiB** |
| 17턴 시나리오: 최대 VRAM | 21.8 GB | **14.1 GB** |
| 17턴 시나리오: 전체 시간 | 103.0초 | **71.9초** (−30%) |

의미 비교(17턴, [compare_ab.txt](2026-11-01_accel_evidence/compare_ab.txt)):
- 주문·실행·commit 필드는 분리 T8~T11을 빼고 모두 같다. Decision 차이 2건은 키 순서뿐이고, Response 차이 5건은 표현뿐이다. 새 허위 발화는 없다(복합 T2의 "크림 선택" 허위 발화는 두 환경 모두에 있는 기존 오류).
- 분리 T8 "소시지만 많이 골라줘":
  - 기존 환경은 §8.3의 캐시 ON 오류를 그대로 재현했다(소시지를 dietary_rule 제한으로 등록).
  - 새 환경은 소시지 high + 게살 allergy 재기록이었고, T9 "바로 담아"에서 `commit:true`가 나왔다(소스 누락으로 실행은 막힘).
  - 이것도 거의 동점 지점의 뒤집힘이다. 새 환경이 이 오류를 "고쳤다"고 주장하지 않는다.
- 사용자 필수 시나리오의 실패(§8.2: 크림 미반영, commit 누락)는 두 환경 모두 그대로다. 데이터 담당 소관이다.

Codex 교차검증: 수치·의미 비교 찬성, **조건부 승인**. 남은 조건은 실제 ROS + 카메라(VLM) smoke test다. VLM은 합성 이미지 1건으로만 확인했다.

### 16.4 남은 단계

1. 실제 로봇 스택에서 LLM 노드를 `soomac_llm` 파이썬으로 실행하고, ROS Humble 경로를 `PYTHONPATH`에 둔다. 대화 몇 턴 + VLM 판정 1회 smoke.
2. 통과하면 LLM 노드 실행 명령을 `soomac_llm`으로 고정한다. 문제가 생기면 `gemma4_env`로 즉시 되돌린다.
3. 이어서 B(vLLM 별도 서버) 계획을 세운다.

Cause: 업그레이드 시도 중 인식하지 못한 PyTorch 동작(inference_mode의 view 정보 누락)이 한 번 있었다. 같은 조건 비교로 바로잡았다.

Effect: LLM 노드의 대기 시간이 30% 더 줄고, 최대 VRAM이 21.8GB에서 14.1GB로 내려갔다. STT·TTS 환경은 건드리지 않았다.

## 17. 전체 개선 요약과 VLM 소요 시간 (2026-11-01)

### 17.1 LLM — 원래 운영 대비 (같은 실제 로그 입력, 캐시 적중 10건)

| Agent | 원래(기존 환경, 캐시 끔) | 새 환경, 캐시 끔 | 새 환경, 캐시 켬 |
| --- | ---: | ---: | ---: |
| Decision | 4.64초 | 4.34초 | **1.04초** |
| Response task | 4.68초 | 4.33초 | **1.59초** |
| Recommendation | 5.08초 | 4.61초 | **3.09초** |
| Response general | 2.51초 | 2.34초 | **1.84초** |
| 평균 | 4.49초 | 4.16초 | **1.49초 (3.0배)** |

- D→P 한 턴 기준 약 9.3초 → 2.6초다. 원래 로그의 D→P 중앙값 9.26초와 맞는다.
- 캐시 없이 환경만 바꾸면 약 7%만 빨라진다. 새 버전이 줄인 건 decode 때 슬라이딩 층이 훑는 길이이고, 캐시가 없으면 시간 대부분이 prefill이기 때문이다.
- 속도를 쪼갠 기여(캐시 ON, 같은 입력): 5.13.1 1.85초 → 5.14.1 1.65초 → +peft/bnb/xgrammar 1.51초 → +clone 1.49초(측정 오차 ±4% 안). clone의 효과는 주로 메모리다.

### 17.2 VLM — 로그의 실제 판정 요청 재현

방법: 로그 `vlm_result` 이벤트에서 재료별로 1건씩 골라 실제 system prompt·user text·이미지 개수를 그대로 쓰고, 이미지는 640×480 합성 이미지로 대체했다(Gemma는 이미지 1장을 최대 280토큰으로 고정 변환). 출력 길이는 로그의 실제 판정 문장 토큰 수로 고정했다. Adapter는 끈다(운영과 같음). 스크립트와 결과: `2026-11-01_accel_evidence/vlm_bench.py`, `vlm_bench_*.log`.

| 항목 | 기존 환경 | 새 환경 |
| --- | ---: | ---: |
| 이미지 수 / 입력 / 출력 (평균) | 6.75장 / 2,243 / 54토큰 | 같음 |
| 전처리 | 0.05초 | 0.05초 |
| 첫 토큰까지(이미지 포함 입력 처리) | 0.72초 | 0.73초 |
| 판정 1회 전체 | **2.90초** | **2.42초** (−17%) |
| 최대 VRAM | 9.08 GiB | 9.09 GiB |

- 판정 1회는 1.75초(뚜껑·면, 37토큰)에서 3.56초(오일, 88토큰)까지다. **시간 대부분은 판정 문장 생성**이다(새 환경 약 31ms/토큰).
- 카메라 프레임 대기·ROS 전송·UI 이미지 발행 시간은 포함하지 않았다.
- 관찰: VLM decode(약 31ms/토큰, Adapter 끔)가 LLM decode(약 55~75ms/토큰, Adapter 켬)보다 훨씬 빠르다. 입력 길이 차이도 있지만, **LoRA를 합치지 않고 따로 계산하는 오버헤드가 decode 시간의 큰 부분일 가능성**이 높다(추정, 미분리 측정). vLLM의 LoRA 전용 커널이 노리는 부분이다.

## 18. B 실행 — vLLM PoC (B1~B3b) 결과와 B4 계획 (2026-11-01)

사용자 결정: 환경 이름은 `soomac_vllm_server`(나중에 다른 LLM 서버로도 쓸 수 있게 일반 이름). 근거 파일: `2026-11-01_accel_evidence/`(`b3_vllm.py`, `b3b_vllm_nf4.py`, `hf_ref.py`, `hf_bf16_ref.py`, `convert_adapters_for_vllm.py`, `b3*_rows.json`, `b3b_vllm_engine.log`).

### 18.1 B1 — 환경

- `soomac_vllm_server`: Python 3.12.16, vllm 0.32.0, torch 2.13.0, transformers 5.19.0, xgrammar 0.2.8, flashinfer 0.7.0.post1, vllm-bnb-plugin 0.0.3, bitsandbytes 0.50.2. `pip check` 충돌 0건, 8.6GB.
- 어텐션 커널은 `TRITON_ATTN`이다(head 512 때문에 FlashInfer·FlashAttention 불가, 예상대로).
- vLLM 로그가 "Gemma4UnifiedForConditionalGeneration supports adding LoRA"를 출력한다. 문서 표에는 비어 있지만 실제로 지원된다.

### 18.2 B2 — 어댑터 변환 (재학습 없음)

- 원인: 글로벌 8층(k_eq_v)에서 HF는 V = `k_proj(x)`(LoRA 포함)인데, vLLM은 기본 K만 V 칸에 복사하고 LoRA는 K 칸에만 붙인다(§4.3).
- 조치: 어댑터 3개 각각 글로벌 층의 `k_proj` lora_A/B 16개를 `v_proj` 이름으로 복제한 사본을 `models/vllm_adapters/{decision,recommendation,response}`(git 제외)에 만들었다. 원본은 변경하지 않았다. 스크립트: `convert_adapters_for_vllm.py`.

### 18.3 B3 — BF16 기반으로 비교: 엔진 차이와 정밀도 차이를 분리

입력은 HF processor가 만든 **토큰 ID를 그대로** vLLM에 넣었다(`TokensPrompt`). 따라서 차이는 채팅 템플릿이 아니라 엔진이나 정밀도에서만 생긴다. "의미 일치" = JSON을 풀어 키 순서와 무관하게 같은지 비교한 것이다.

| 비교 (14건) | 의미 일치 |
| --- | ---: |
| HF-BF16 vs vLLM-BF16 + 변환 어댑터 | 11 (cold 12) |
| HF-BF16 vs vLLM-BF16 + 원본 어댑터 | 7 |
| **HF-NF4 vs HF-BF16** | **2** |

- vLLM은 변환 어댑터를 붙이면 같은 정밀도의 HF와 거의 같은 답을 낸다. 어댑터 변환은 효과가 있다(7 → 11).
- **기반 모델을 BF16으로 바꾸면 대답이 크게 달라진다**(HF끼리도 2/14). 어댑터가 NF4 위에서 QLoRA로 학습됐기 때문이다. 예: "크림으로 선택"에 NF4는 `order: 크림`, BF16은 `confirmation: accept`. "바로 담아"에 NF4는 `commit:true`, BF16은 놓친다.
- 결론: **추론 기반은 학습 기반(NF4)과 같아야 한다.** BF16 vLLM은 VRAM도 약 30GB를 차지해 운영에 맞지 않는다.

### 18.4 B3b — vLLM + NF4 (bitsandbytes 플러그인, 이중 양자화 켬)

| 항목 | 값 |
| --- | --- |
| HF-NF4(캐시 끔)와 의미 일치, 변환 어댑터 | **12/14** (Decision 6/6, Recommendation 2/2, Response 4/6) |
| 같은 조건, 원본 어댑터 | 10/14 |
| 불일치 2건 | Response 문장 표현뿐. 그중 하나는 HF 캐시 ON에서도 나왔던 같은 대안 문장 |
| 모델 / KV 메모리 | 8.63 GiB / 6.9 GiB (21,489토큰), `gpu_memory_utilization=0.60` |

지연(같은 입력 14건):

| 조건 | 평균 | 비고 |
| --- | ---: | --- |
| 원래 HF, 캐시 없음 | 4.18~4.49초 | 기준 |
| 현재 HF + 접두 캐시(A) | 1.49초 | 캐시 적중 10건 |
| **vLLM-NF4, 실제 조건**(입력이 다르고 공통 앞부분만 캐시) | **0.53초** (6건) | Decision 0.28~0.54초, Response task 0.68~0.74초 |
| vLLM-NF4, 같은 입력 재실행(warm) | 0.51초 | 입력 전체가 캐시됨 → 낙관치(Codex 지적) |
| vLLM-NF4, Agent별 첫 호출(캐시 없음) | 1.94초 평균 | HF 캐시 없음 4.2초보다 빠름 |

관찰: 1회차에서 **Agent별 두 번째 호출도 캐시가 적중하지 않았다**(4건). 세 번째부터 적중했다. 원인은 모른다(추정: 슬라이딩 층 블록이 캐시에 등록되는 시점 문제). B4에서 실제 대화로 적중률을 다시 잰다.

Codex 교차검증: 5개 항목 모두 조건부 찬성. 지적 사항은 다음 셋이다.
- warm 0.51초는 낙관치다 → 실제 조건 0.53초를 따로 표기했다.
- "엔진 영향 없음"은 과장이다 → 남은 2건은 엔진 차이일 수 있다.
- 실행 당시 버전 기록 → `soomac_vllm_server` 버전을 위에 명시했다.

판정: **B4 조건부 진행.**

### 18.5 B4 계획 — 노드는 그대로, 모델 호출만 vLLM 서버로

```text
[soomac_vllm_server 환경, Python 3.12]             [soomac_llm 환경, Python 3.10 + ROS Humble]
 vllm serve gemma-4-12B-it (NF4)                    LLM 노드 (LangGraph·정책·ROS 그대로)
   + LoRA decision/recommendation/response   ◀─HTTP─  make_call_decision / recommendation / response
   + 자동 접두 캐시 + XGrammar                         make_call_vlm (Adapter 없이 base 호출)
   localhost:8000                                    → 모델을 직접 로드하지 않음
```

| 단계 | 내용 | 통과 기준 |
| --- | --- | --- |
| B4-1 | 서버 실행 함수(`.bashrc`에 `soomac_vllm_server`) 작성. NF4, LoRA 3개, 접두 캐시, `max_lora_rank 16`, 메모리 비율 조정 | `/health` 응답, 어댑터 3개 등록 |
| B4-2 | 노드 쪽 백엔드 스위치 `LLM_BACKEND = "hf" \| "vllm"`. vLLM 모드에서는 기존처럼 토큰 ID를 만들어 `/v1/completions`로 보낸다(어댑터 이름, 문법, stop 토큰 포함). 그래프·정책·프롬프트는 바꾸지 않는다 | HF 모드는 기존과 완전히 동일(단위 테스트) |
| B4-3 | 17턴 시나리오를 vLLM 모드로 실행 → HF(A)와 턴별 비교. 적중률·실제 조건 지연 측정 | 주문·실행·commit 필드가 A와 같음(다르면 원인 기록), 지연이 A보다 짧음 |
| B4-4 | VLM을 서버로 보내는 경로 시험(이미지 7장, Adapter 없음). vLLM의 이미지 전처리가 HF와 같은지 확인 | 판정 결과가 HF와 같음 |
| B4-5 | 메모리 조정: TTS 등 다른 노드와 같이 돌 수 있는 `gpu_memory_utilization` 결정 | OOM 없음 |

**어댑터를 다시 학습할 때 규칙**
1. 학습은 지금처럼 HF·PEFT, **NF4(QLoRA)**로 한다. 추론 기반과 같아야 한다.
2. 기반 모델은 같은 `gemma-4-12B-it`이어야 한다. LoRA 대상은 언어모델 Linear 층(현재와 같음).
3. 학습이 끝나면 `convert_adapters_for_vllm.py`를 돌려 변환본을 만든다(몇 초).
4. rank가 16보다 크면 서버의 `--max-lora-rank`를 맞춰 올린다.
5. 서버를 재시작하거나, 런타임 어댑터 로드 API로 교체한다.

## 19. B4 실행 — vLLM 서버 연결과 실제 흐름 검증 (2026-11-01)

### 19.1 만든 것

| 파일 | 역할 |
| --- | --- |
| [tools/vllm_server/start_vllm_server.sh](../tools/vllm_server/start_vllm_server.sh) | `soomac_vllm_server` 환경으로 서버 실행. NF4(bitsandbytes), LoRA 3개, 자동 접두 캐시, XGrammar, `127.0.0.1:8000`. 오프라인 변수 고정, `PYTHONPATH` 해제. `--max-num-batched-tokens 12288` |
| [vllm_client.py](../src/soomac_irc/soomac_irc/vllm_client.py) | 노드 쪽 클라이언트(표준 라이브러리만). `VllmModel`: HF 모델 자리에 그대로 넣는 대리 객체(`generate`/`set_adapter`/`disable_adapter`). 입력 토큰 ID를 그대로 보내고 XGrammar 문법을 서버 문법으로 넘긴다. `make_call_vlm_remote`: VLM을 서버 base 모델로 판정. `check_served_adapters`: 서버가 올린 변환본(`/v1/models`의 root)이 현재 `model_runtime` 어댑터에서 만들어졌는지 검사 |
| [llm_langgraph_node.py](../src/soomac_irc/soomac_irc/llm_langgraph_node.py) | `LLM_BACKEND = "hf"`(기본, 기존과 동일) / `"vllm"`. 모델 로드와 VLM 연결 2곳만 분기. Decision/Recommendation/Response 코드, `model_runtime.py`, 프롬프트, 그래프는 변경 없음 |
| [test_vllm_backend.py](../src/soomac_irc/test/test_vllm_backend.py) | 서버 없이 도는 단위 테스트 5개. 전체 918개 통과 |

### 19.2 검증 결과

| 항목 | 결과 |
| --- | --- |
| 오프라인 | 네트워크 격리(`unshare -rn`, 외부 접속 `URLError`)에서 서버 기동 132초, 어댑터 3개 등록, 요청 정상 |
| 서버 기본 조각 계산 | 기본 `max_num_batched_tokens` 2,496이면 6~8k 입력을 나눠 계산한다. 의미 일치 9/14 → 12288로 올린 뒤 10/14. 인과는 추정(Codex) |
| 17턴 시나리오(같은 코드·프롬프트, A 대비) | 전체 71.9초 → **33.5초 / 32.8초**(2회). 턴 중앙값 3.14 → **1.33초**. Decision 중앙값 1.36 → 0.57초, Response 1.93 → 0.79초. 서버 캐시 적중률 80.5% |
| 주문·실행·commit 필드 | 1회차: 분리 T11만 다름(vLLM이 말하지 않은 소스 '토마토'를 넣고 허위 응답). **2회차: 17/17 A와 동일**(T11 `clarify`) |
| VLM(실제 판정 프롬프트 8종 + 실제 밀키트 카메라 이미지 6~7장) | 판정 **8/8 동일**. 평균 2.53 → **1.56초**. 이유 문장 표현은 다름 |
| GPU | 서버 기동 직후 19.2GB, 시나리오 후 21.5GB(`gpu_memory_utilization 0.60`). A 최대 14.1GB보다 큼 |
| 노드 시작 경로 | 실제 서버로 `load_vllm_backend` → `make_call_decision` 1건 정상. 다른 어댑터 경로는 오류로 차단 확인 |

**vLLM은 실행할 때마다 결과가 완전히 같지는 않다.** HF는 같은 입력이면 17턴이 매번 같았다(§8.1). vLLM은 두 실행 사이에 거의 동점인 지점(T11)이 갈렸다. 원인은 추정(캐시 적중 여부·일괄 처리 상태에 따른 계산 경로 차이)이다. 로봇 시험에서 이 지점들을 더 관찰해야 한다.

Codex 교차검증: AGREE 2 / CONDITIONAL 3. 코드 결함 2건 지적 → 수정 완료.
1. 어댑터 확인이 로컬 파일만 봄 → 서버 `/v1/models`의 실제 경로로 대조하게 변경
2. 응답에 `token_ids`가 없으면 빈 출력 → 명확한 오류로 변경

판정: **`LLM_BACKEND` 기본값은 `hf`로 유지하고, vLLM은 로봇 시험용 선택 백엔드로 조건부 채택.**

### 19.3 사용 방법

```bash
# 터미널 1: 서버 (준비까지 약 2분 반)
/home/roma/IRC_SOOMAC3.0/tools/vllm_server/start_vllm_server.sh
# 터미널 2: llm_langgraph_node.py의 LLM_BACKEND = "vllm" 으로 바꾼 뒤 기존처럼
soomac_llm
```

- 노드는 서버가 준비될 때까지 기다린다. 서버 Adapter가 현재 학습 결과와 다르면 시작을 거부한다.
- 어댑터를 새로 학습했으면: `convert_adapters_for_vllm.py` 실행 → **서버 재시작**. 재시작하지 않고 같은 경로에 다시 변환만 하면 검사가 잡지 못한다.

### 19.4 남은 일

1. 실제 로봇 스택(STT·TTS·카메라 동시 실행)에서 `LLM_BACKEND="vllm"` smoke와 GPU 메모리 확인. 필요하면 `VLLM_GPU_MEM`을 낮춘다(이론상 최소 약 0.5).
2. vLLM 비결정성 관찰: 같은 시나리오 반복 시 갈리는 턴 기록. 필요하면 vLLM의 결정적 실행 옵션을 조사한다.
3. `.bashrc`에 `soomac_vllm_server` 함수 추가 여부(사용자 결정).

## 20. GPU 메모리 예산(27GB 상한)과 NVFP4·MTP 조사 (2026-11-01)

### 20.1 함께 도는 모델 실측 (노드와 같은 방식으로 로드 후 5회 추론, `nvidia-smi` 0.2초 샘플)

| 프로그램 | 환경 | 로드 후 | 추론 중 최대 | 1회 |
| --- | --- | ---: | ---: | ---: |
| 데스크톱(화면·브라우저) | — | 1.3~1.4GB | — | — |
| STT Qwen3-ASR 1.7B bf16 | `gemma4_env` | 4.1GB | 4.4GB | 0.48초(10초 음성) |
| TTS CosyVoice3 0.5B | `cosyvoice3` | 5.0GB | 5.3GB | 2.5초(3문장) |

VLM RAG는 저장된 벡터만 읽어 GPU 모델을 따로 올리지 않는다. 카메라·비전 노드의 GPU 사용은 측정하지 않았다.

### 20.2 vLLM 서버 설정별 실측 (STT·TTS 상주 상태에서 17턴 시나리오)

| 비율 : 동시 처리 : 최대 길이 | 서버 준비 → 시나리오 후 | GPU 전체 최대 | KV 토큰 | 17턴 | 캐시 적중 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.60 : 256 : 12288 | 18.0 → 20.4GB | 31.2GB | 19,666 | 34.1초 | 80% |
| 0.55 : 256 : 12288 | 16.4 → 18.8GB | 29.6GB | 14,829 | 32.7초 | 80% |
| 0.50 : 256 : 12288 | 기동 실패(KV 3.19 < 3.94GiB) | — | — | — | — |
| 0.50 : 2 : 10240 | 14.8 → 17.2GB | 27.9GB | 15,659 | 34.2초 | 80% |
| **0.45 : 2 : 10240 (기본값)** | 13.3 → 15.6GB | **26.6GB** | 10,717 | 37.2초 | 77% |
| 0.42 : 2 : 10240 | 기동 실패(KV 부족) | — | — | — | — |

- **단위 주의(Codex 지적)**: 위 표의 "GB"는 `nvidia-smi` MiB를 1000으로 나눈 값이다. 기본값의 최대 26,594MiB = **25.97GiB**(nvtop 표기). STT·TTS 동시 추론 증가분 542MiB를 더하면 27,136MiB = **26.5GiB**로 27GiB 상한 안이지만, 여유는 약 0.5GiB다. 카메라·비전 노드의 GPU 사용은 미측정이다.
- 남은 위험(Codex 지적): Decision 앱 쪽 입력 상한은 16,384토큰(`DECISION_INPUT_MAX_TOKENS`), 서버는 10,240이다. 10,240~16,384 입력은 HF에서는 되지만 vLLM 모드에서는 오류가 난다. 현재 관측 최대는 약 8.4k다.
- 동시 처리 수를 기본 256에서 2로 줄이면 CUDA graph·작업 공간이 줄어, 같은 비율에서 KV가 더 커진다.
- **서버는 준비된 뒤 실제 요청을 받으면서 약 2.3~2.4GB가 더 늘어난다**(원인 미확인, LoRA 계산용 임시 공간으로 추정). 예산은 이 증가분까지 포함해 잡았다.
- "GPU 전체 최대"에는 시험 하네스 클라이언트의 CUDA 사용(약 0.5GB 추정)이 포함됐고, STT·TTS 추론 증가분(+0.5GB)은 빠져 있다. 실제 로봇에서 LLM 노드(vLLM 모드)는 GPU를 쓰지 않는다.
- 결론: **기본값 0.45 / 동시 2 / 최대 10,240** → 27GB 상한 안. 속도는 HF(71.9초) 대비 약 2배.
- 주의: 최대 길이 10,240은 현재 Decision 최대 입력(약 8.4k)에 맞춘 값이다. 노드는 출력 한도를 남은 길이로 자동으로 줄이고, 입력 자체가 넘으면 명확한 오류를 낸다. 프롬프트가 늘면 `VLLM_MAX_LEN`을 올린다(메모리 증가).

### 20.3 NVFP4·MTP 지원 조사 (설치된 vLLM 0.32 코드·공식 문서·HF API 직접 확인)

| 항목 | 결과 | 근거 |
| --- | --- | --- |
| Gemma 4 12B MTP 보조 모델 | 있음 | `google/gemma-4-12B-it-assistant` 0.88GB, 공개. vLLM `config/speculative.py:1019`가 `gemma4_unified_assistant`를 `Gemma4MTPModel`로 연결 |
| **MTP(추측 디코딩) + LoRA** | **공식 미지원** | vLLM v0.32.0 `docs/features/README.md` 호환표 "SD × LoRA = ❌". 코드에서 막는 검사는 찾지 못함(돌아는 가도 결과 보장 없음) |
| 이 GPU에서 FP4 전용 연산 | 지원 | `cutlass_fp4_supported() = True`(설치된 vLLM으로 실행 확인) |
| Gemma 4 12B NVFP4 공개 모델 | 여러 개 | RedHatAI(W4A4, 10.3GB), coolthor NVFP4A16(8.25GB), AtomicChat(8.28GB), vrfai(modelopt, 11.7GB). `nvidia/` 공식판은 없음 |
| 기반 모델 버전 | 문제 없음 | `google/gemma-4-12B-it` 가중치는 6월 이후 변경 없음(10-30 커밋은 토크나이저 수정) |

판단:
- **MTP**는 Agent 3개가 모두 LoRA인 현재 구조에서 공식 미지원이라 우선순위가 가장 낮다. 어댑터를 합친 Agent별 모델 3개를 두는 방식은 VRAM이 부족하다.
- **NVFP4 + 기존 어댑터**는 대답이 크게 달라질 위험이 있다. 근거는 §18.3(어댑터가 NF4에 맞춰 학습돼 BF16으로만 바꿔도 12/14가 바뀜)에서 나온 추정이다. B3와 같은 14건 비교로 30분이면 확인할 수 있다. 크게 다르면 NVFP4 기반 재학습 여부는 데이터 담당이 판단한다. W4A4는 이미지 입력 품질 저하 보고가 있어 VLM 확인도 필요하다.

## 21. MTP·결과 고정(Batch Invariance) 실험 (2026-11-01)

### 21.1 사전 조사 요약 (Codex: 실험은 합리적, 운영은 미보장)

- `google/gemma-4-12B-it-assistant`(0.84GB, bf16, 4층)를 `/home/roma/models/gemma-4-12B-it-assistant`에 받았다. vLLM 0.32가 `gemma4_unified_assistant` → `Gemma4MTPModel`로 연결한다.
- LoRA + MTP 시작 시 충돌(이슈 #62340, v0.26.0에서 신고)의 수정 PR #47953·#47920이 **v0.32.0에 포함**돼 있다(GitHub compare). 설치 코드에서 보조 모델이 LoRA를 씌운 본 모델의 임베딩을 공유한다(`v1/spec_decode/gemma4.py:171`).
- LMCache가 12B에 필요하다고 한 `--enforce-eager`의 원인(CUDA graph 중 CPU 인덱스)은 0.32에서 GPU 텐서로 바뀌어 있다(`gemma4_mtp.py:590`). 필요 없을 것으로 추정했고, 실행으로 확인한다.
- vLLM v0.32.0 기능 호환표(`docs/features/README.md:44`)는 **SD × LoRA = ❌**다. 코드에서 막지는 않는다. 그래서 **MTP 켬/끔 출력 동일성**으로 결과가 조용히 틀리지 않는지 검증한다.
- 보조 모델이 bf16이어도 정답에는 영향이 없다(본 모델이 모든 토큰을 검증). 적중률(속도)에만 영향을 준다. 보조 모델은 원본 bf16에 맞춰 학습돼 NF4 + LoRA에서는 적중률이 낮을 수 있다.
- `VLLM_BATCH_INVARIANT=1`: 배치나 순서가 달라도 같은 수치를 내도록 하는 베타 옵션이다. 정답률을 올리는 기능이 아니라 **재현성** 기능이다(틀린 답이 매번 똑같이 나올 수도 있다).

### 21.2 계획

| 설정 | MTP | Batch Invariant | 목적 |
| --- | --- | --- | --- |
| A | 끔 | 끔 | 기준 |
| B | 끔 | 켬 | 출력 흔들림(§19 T11 '토마토')이 MTP와 별개로 사라지는지 |
| C | 켬(k=2) | 끔 | MTP 단독 속도·품질 |
| D | 켬(k=2) | 켬 | 결합 |

- 공통 조건: 서버 스크립트 그대로(NF4, LoRA 3개, 최대 길이 10,240, 동시 처리 2). **메모리 비율은 네 설정 모두 0.55로 고정**한다. MTP 공간 확보와 공정 비교를 위해서이고, STT·TTS는 띄우지 않는다. 27GiB 운영 적합성은 통과한 설정만 따로 잰다.
- 측정
  - 로그 입력 14건 × 2회(1회차 실제 조건, 2회차 warm)
  - 17턴 시나리오 × 2회(같은 서버에서 연속 실행)
  - Agent별 지연, MTP 적중률(서버 metrics), KV 공간, 서버 GPU 메모리
- 판정은 문장 유사도가 아니라 **필드 값**으로 한다.
  - 14건: HF-NF4 기준 의미 일치, 설정 A와의 토큰 완전 일치
  - 17턴: 주문·실행·commit·pending, 같은 설정 2회 간 일치, 설정 A와의 일치
- 실패 규칙: 메모리 부족이나 시작 실패가 나도 비율을 억지로 올리지 않는다. 로그의 KV 공간과 원인만 기록한다. 기존 서버 기본값과 어댑터는 그대로 둔다.

### 21.3 결과 (근거: `2026-11-01_accel_evidence/mtp/`, 시나리오 원본 `training/.../accel_prefix_cache_20261101/mtp_*`)

| 설정 | 시작 | 호출 1회(warm) | 실제 조건(앞부분 캐시) | 17턴(1회차) | 같은 서버 2회 반복 일치 | A와 필드 일치 | HF-NF4와 의미(warm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A 기준 | ✅ | 0.513초 | 0.521초 | 33.2초 | 16/17 | — | 11/14 |
| B 결과 고정 | ✅ | 0.648초(+26%) | 0.623초 | 35.0초(+5%) | **17/17** | 13/17 | 13/14 |
| C MTP(k=2) | ✅(2차) | 0.483초(−6%) | 0.488초 | 32.2초(−3%) | 13/17 | 17/17 | 11/14 |
| D MTP + 결과 고정 | ✅(2차) | 0.535초 | 0.527초 | 32.7초 | **17/17** | 16/17 | 11/14 |

**MTP**
- 1차 시도는 C·D 모두 시작에 실패했다. 오류: `Gemma4MTP does not support BitsAndBytes quantization`. 원인: 보조 모델이 본 모델의 bitsandbytes 로드 방식을 물려받았다(`model_loader/utils.py:46-56`). 서버 스크립트의 MTP 옵션에 `"draft_load_config": {"load_format": "auto"}`를 넣어 **보조 모델만 bf16 원본으로** 올리니 해결됐다.
- `enforce_eager=False`, CUDA graph 정상 캡처. LMCache가 말한 `--enforce-eager`는 0.32에서 필요 없다.
- LoRA 3개 + 문법 강제와 함께 동작했다. 추측 적중률은 **75.8%**(1418/1872, k=2)로 공개 사례(85%)보다 낮다. NF4 + LoRA 때문으로 추정한다.
- 속도 이득은 작다. 호출당 −6%, 17턴 −3%이고, Response(general)는 오히려 느려졌다(0.426 → 0.478초).
- 비용: 모델 메모리 8.63 → 9.42GiB, 같은 비율(0.55)에서 KV 6.59 → 4.68GiB.
- 출력이 완전히 같지는 않다. A와 토큰 완전 일치 24/28, 같은 서버 2회 반복 13/17(T8 한 지점이 뒤집혀 T9~T11로 연쇄). 결과 고정(D)과 함께 쓰면 17/17이 된다.

**결과 고정(`VLLM_BATCH_INVARIANT=1`)**
- 같은 서버에서 17턴을 2번 돌렸을 때 A는 16/17, B는 17/17로 같았다. 실행마다 답이 흔들리는 문제를 줄인다.
- 비용: 호출당 +26%, 17턴 +5%. 메모리는 0.55 기준으로 거의 같다(KV 6.59 vs 6.62GiB). 0.45에서는 재지 않았다.
- 정확도 기능은 아니다. 실패 지점이 바뀌었다.
  - A: 분리 T8 "소시지만 많이"에서 소시지를 식단 제한으로 등록
  - B: T8은 맞지만 T9 "바로 담아"에서 **말하지 않은 '토마토' 소스를 넣고 commit**
  - 실패 체크는 A 45, B 43으로 B가 조금 적다(Codex 정정)

**데이터 담당에게 넘기는 관찰**: 소스가 비어 있는 상태에서 "바로 담아"를 받으면, 실행 경로에 따라 모델이 소스('토마토')를 지어내 commit한다(§19 vLLM 1회차, §21 B T9). 속도 설정과 무관한 모델 쪽 위험이다.

### 21.4 결론과 다음 단계

| 항목 | 판단 |
| --- | --- |
| MTP | **지금은 채택하지 않는다.** 기술적으로는 동작하지만, 이득(3~6%)에 비해 VRAM이 약 2GiB 더 들고 실행마다 출력이 흔들린다. 27GiB 상한 안에서 돌아가는지는 미측정(KV 부족 추정) |
| 결과 고정 | **선택 옵션.** 로봇 시험·디버깅 단계에서 재현성이 필요하면 켠다(17턴 +5%). 0.45 설정에서의 기동·메모리는 아직 확인하지 않았다 |
| 기본 서버 설정 | 변경 없음(0.45, MTP·결과 고정 꺼짐) |

Codex 교차검증: X1~X3 찬성, X4~X6 조건부. "결과 고정이 정확도를 안 높인다"는 표현이 단정적이라는 지적, 0.45와 27GiB 관련 내용이 추정이라는 지적을 반영했다.

## 22. 결과 고정 운영 설정 확인 + NVFP4 비교 (2026-11-01)

### 22.1 계획

**① 결과 고정을 운영 조건에서 확인**
- 조건: 서버 기본값(0.45 / 동시 2 / 최대 10,240) + `VLLM_BATCH_INVARIANT=1`, STT·TTS 상주
- 측정: 기동 여부, KV 공간, GPU 전체 최대(27GiB 상한), 17턴 × 2회 필드 일치, 17턴 시간
- 통과 기준: 기동 성공, 27GiB 이하, 반복 17/17 일치 → 로봇 시험 단계용 옵션으로 권장

**② NVFP4 + 기존(변환) 어댑터**
- 후보: `RedHatAI/gemma-4-12B-it-NVFP4`(가중치·연산 모두 FP4, W4A4)와 가중치만 FP4인 W4A16 1종. 다운로드는 각 8~10GB이며 ①을 도는 동안 받는다.
- 서버 스크립트에 `VLLM_QUANT=auto`(체크포인트의 양자화 설정을 그대로 사용) 옵션을 추가한다. 기본값은 bitsandbytes 그대로다.
- 측정: 14건(HF-NF4 기준 의미 일치, Decision 필드), 지연, 메모리, VLM 실제 이미지 8종 판정(HF와 비교)
- 판단: Decision·Recommendation 필드가 HF-NF4와 크게 다르면 기존 어댑터로는 불가 → 재학습 판단은 데이터 담당. 비슷하면 17턴·메모리 시험으로 진행
- 실패 규칙: 기동 실패나 OOM이 나면 원인만 기록하고 비율을 억지로 올리지 않는다. 기존 설정과 어댑터는 그대로 둔다.

### 22.2 ① 결과 — 결과 고정을 운영 설정에서 확인: 통과

| 항목 | 값 |
| --- | --- |
| 조건 | 0.45 / 동시 2 / 최대 10,240 + `VLLM_BATCH_INVARIANT=1`, STT·TTS 상주 |
| 기동 | 성공, KV 10,927토큰 |
| GPU 전체 최대 | 26,463MiB = **25.8GiB**. STT·TTS 동시 추론(+542MiB)을 더해도 약 26.4GiB → 27GiB 안 |
| 17턴 | 36.0초(결과 고정 없는 0.45: 37.2초와 비슷) |
| 같은 서버 17턴 2회 필드 일치 | **17/17** |
| 분리 T9 "바로 담아" | `commit`만 출력(토마토 지어내기 없음) |

- 주의: 같은 결과 고정이라도 메모리 비율 0.55(§21 B)와 0.45는 T9~T11 답이 달랐다(14/17 일치). 결과 고정은 **같은 설정 안에서** 반복을 고정하는 것이지, 설정이 바뀌어도 같은 답을 보장하지는 않는다. 원인은 KV 크기에 따른 캐시 밀어내기 경로 차이로 추정한다.
- 결론: 로봇 시험 단계에서는 `VLLM_BATCH_INVARIANT=1 soomac_vllm_server`로 켜는 것을 권장한다(비용 17턴 약 +5% 이내). 기본값을 바꿀지는 사용자가 결정한다.
- **사용자 결정(2026-11-01): 서버 기본값으로 켬.** `start_vllm_server.sh`에서 `VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}`. 끄려면 `VLLM_BATCH_INVARIANT=0 soomac_vllm_server`. NVFP4 시험은 기준(A, 끔)과 비교하기 위해 명시적으로 끄고 돌린다.
- 근거: `2026-11-01_accel_evidence/mtp/gpu_sweep_bi.out`, 시나리오 `training/.../accel_prefix_cache_20261101/bi_045/`

### 22.3 현재까지 요약 (2026-11-01, NVFP4 시험 전)

**속도 단계별 (17턴 시나리오)**

| 단계 | 방식 | 17턴 | Decision 1회 | 최대 GPU | 상태 |
| --- | --- | ---: | ---: | ---: | --- |
| 시작 전 | HF, 캐시 없음 | 210.8초 | 4.64초 | 13.5GB | 기존 |
| A | HF + 접두 캐시 + `soomac_llm` | 71.9초 | 1.04초 | 14.1GB | 현재 기본(`LLM_BACKEND="hf"`) |
| B | vLLM 서버(0.45 / 동시 2 / 10,240) | 36~37초 | 약 0.4~0.5초 | 25.8GiB(STT·TTS 포함 전체) | 선택(`"vllm"`), 로봇 시험 대기 |

**vLLM 추가 옵션**

| 옵션 | 기동 | 속도 | 같은 입력 2회 반복 일치 | 메모리 | 판단 |
| --- | --- | ---: | ---: | --- | --- |
| 없음(기준) | ✅ | — | 16/17 | 기준 | — |
| 결과 고정(`VLLM_BATCH_INVARIANT=1`) | ✅ | 17턴 약 +5% 이내 | 17/17 | 변화 없음, 27GiB 안 | **서버 기본값으로 켬**(사용자 결정) |
| MTP(k=2) | ✅(보조 모델 로더 수정 후) | −3~6% | 13/17 | 약 +2GiB | 미채택 |
| MTP + 결과 고정 | ✅ | 기준과 비슷 | 17/17 | 약 +2GiB | 미채택(메모리) |
| NVFP4 | 다운로드 중 | — | — | — | 시험 예정(§22.4) |

**데이터 담당 전달 — 모델 쪽 위험**

| 상황 | 현상 | 근거 |
| --- | --- | --- |
| 소스가 빈 상태에서 "바로 담아" | 실행 경로에 따라 말하지 않은 '토마토' 소스를 지어내 commit | §19.2 vLLM 1회차, §21.3 B T9 |
| "소시지만 많이 골라줘" | 가끔 소시지를 식단 제한(dietary_rule)으로 등록 | §8.3, §21.3 A T8 |
| "크림으로 선택할게. 바로 담아." | 크림이 주문에 안 들어감(모든 설정 공통) | §8.2 |

### 22.4 NVFP4를 내려받는 이유

GPU(SM120)는 NVFP4 연산을 지원한다(`cutlass_fp4_supported=True`). 하지만 **모델 파일이 NVFP4 형식이어야** 쓸 수 있다.

| | 현재(NF4) | NVFP4 |
| --- | --- | --- |
| 디스크의 모델 | 원본 BF16(23GB) | 미리 FP4로 변환된 체크포인트(8~10GB) |
| 4bit 변환 시점 | 서버를 켤 때마다 bitsandbytes가 즉석 변환 | 미리 변환한 파일을 그대로 읽음 |
| 미리 변환이 필요한 이유 | NF4는 고정 규칙이라 즉석 변환 가능 | 값 16개 묶음마다 배율을 정해야 하고, 연산까지 FP4(W4A4)로 하려면 실제 데이터로 활성값을 보정(calibration)해야 한다. vLLM 0.32는 이를 기동 시 즉석으로 하지 않는다 |

- 방법은 공개 변환본을 받는 것(시험용, 현재 진행)과 llm-compressor로 원본을 직접 변환하는 것(보정 데이터와 GPU 시간 필요, 시험 결과가 좋을 때 검토) 두 가지다.
- 받는 것: `RedHatAI/gemma-4-12B-it-NVFP4`(W4A4, 비전·오디오 입력층과 lm_head는 제외) → `/home/roma/models/nvfp4/redhat_w4a4`, `coolthor/gemma-4-12B-it-NVFP4A16`(W4A16) → `/home/roma/models/nvfp4/coolthor_w4a16`.
- 다운로드 메모: HF의 xet 전송이 1.03GB에서 1분 넘게 멈춰 프로세스를 종료하고 `HF_HUB_DISABLE_XET=1`(일반 HTTP)로 이어받는 중이다. 속도는 약 3~4MB/s.
- 예상 위험: 어댑터는 NF4 위에서 QLoRA로 학습됐다. BF16으로만 바꿔도 14건 중 12건이 달라졌으므로(§18.3), NVFP4에서도 크게 달라질 가능성이 높다(추정). 이번 시험은 그 정도를 숫자로 확인하는 단계다.
- 서버 스크립트에 `VLLM_QUANT=auto` 옵션을 추가했다(기본값 bitsandbytes 유지). `MODEL_PATH`를 NVFP4 체크포인트로 주면 체크포인트의 양자화 설정을 그대로 쓴다.

## 23. NVFP4용 LoRA 재학습 경로와 어댑터 켜고 끄기 (2026-11-01)

### 23.1 NVFP4가 빠르면 NVFP4 전용 LoRA를 만들면 되나 → 가능. 단, 재학습은 지금 어댑터의 품질이 떨어질 때만 필요하다

| | 현재(NF4) | NVFP4 전용 |
| --- | --- | --- |
| 학습 도구 | HF + bitsandbytes QLoRA(표준, 현재 학습 도커) | **NVIDIA ModelOpt NVFP4 QLoRA(공식 예제, 1순위)**: 압축된 FP4 가중치를 그대로 두고(`mtq.compress()`) LoRA만 학습. 대안: NVFP4 가중치를 bf16으로 풀어 그 위에서 학습 |
| W4A16(가중치만 FP4) | — | 학습·추론을 맞추기 쉽지만, 속도 목적이면 FP4 연산을 쓰지 못해 이점이 작을 수 있다 |
| W4A4(연산도 FP4) | — | **속도 목적의 핵심 후보.** ModelOpt QLoRA/QAT 공식 예제가 W4A4 레시피라 활성값 양자화 오차까지 학습에 반영할 수 있다 |
| 학습 GPU | NF4라 작다 | 단정 불가. 공식 QAT 예제(Qwen3-8B)는 80GB × 2 이상, QLoRA는 이보다 적다. Gemma 4 12B·32GB에서 되는지는 미검증 |

- 원리: 어댑터는 학습할 때 본 기반 모델에 맞춰진다(§18.3: 기반을 BF16으로만 바꿔도 12/14가 달라짐). 그래서 NVFP4에서도 품질이 떨어질 가능성이 있지만, **실제로 떨어지는지는 §22 시험으로 확인한다.** 유지되면 지금 어댑터를 그대로 쓴다.
- LoRA 가중치 자체는 NVFP4로 양자화하지 않는다. bf16 그대로 둔다.
- 재학습은 데이터 담당 소관이다. 속도 담당은 NVFP4의 속도·메모리·현재 어댑터 적합도를 숫자로 제공한다.

### 23.2 어댑터 켜고 끄기

| 질문 | 답 |
| --- | --- |
| 요청마다 어댑터 고르기 | 된다. 지금도 Decision/Recommendation/Response 요청마다 서버가 어댑터를 바꾼다 |
| 어댑터 끄기 | 된다. VLM 판정은 어댑터 없이 기반(`gemma4-base`)으로 호출한다 |
| 같은 서버에 NF4용 어댑터와 NVFP4용 어댑터 같이 두기 | **등록은 된다.** 다만 서버 하나는 기반 모델 하나만 올리므로, 그 기반에서 학습되지 않은 어댑터는 품질이 보장되지 않는다. NF4 기반과 NVFP4 기반을 동시에 쓰려면 서버가 두 개 필요하고, GPU 메모리가 두 배라 27GiB 상한에서 불가능하다 |

결론: 선택지는 "NF4 + 현재 어댑터(검증 완료)", "NVFP4 + 현재 어댑터(품질이 유지될 때)", "NVFP4 + 재학습한 어댑터 3개(품질이 떨어질 때, ModelOpt QLoRA 1순위)" 셋이다.

### 23.3 외부 의견(GPT) 팩트체크

| 주장 | 판정 | 근거 |
| --- | --- | --- |
| NVIDIA ModelOpt에 NVFP4 QLoRA 학습 → 기반·LoRA 분리 내보내기 → vLLM 서빙 예제가 있다 | **사실** | `Model-Optimizer/examples/llm_qat/README.md` "QLoRA (Real Quantization)": quantize(`--compress True`) → `train.py --config qlora_nvfp4.yaml` → `export.py` → `vllm serve .../base_model --enable-lora --lora-modules adapter=...`. 대상은 Qwen3-8B이고, "QLoRA export is not currently supported with FSDP2" |
| 그 예제의 기본 레시피 | 주의 | `general/ptq/nvfp4_default-kv_fp8`은 W4A4 + **FP8 KV 캐시**다. FP8 가중치 제외 결정과는 별개지만 KV도 FP8이 되므로 따로 확인이 필요하다 |
| vLLM 0.32가 NVFP4를 공식 지원한다 | **사실** | v0.32.0 `docs/features/quantization/modelopt.md`: NVFP4 / W4A16_NVFP4(`modelopt_fp4`) |
| 네이티브 FP4 커널이 없으면 Marlin(W4A16)으로 대체되어 기대한 가속이 줄 수 있다 | **사실** | 같은 문서 note: 대체 시 경고 로그. `--linear-backend`로 강제 가능. 우리 GPU는 `cutlass_fp4_supported=True`라 네이티브 경로가 예상되며, 실험 로그로 확인한다 |
| NVFP4 + LoRA에서 커널 충돌 사례(GLM-5.2) | **사실, 범위 주의** | 이슈 #48862(열림, vLLM 0.25.1, GLM-5.2-NVFP4, 첫 요청에서 LoRA Triton 커널 illegal memory access), 수정 PR #49058(열림, "NVFP4/EP padded buffers"). GLM-5.2는 MoE이고, 수정 대상이 전문가 병렬(EP) 패딩 버퍼라서 **dense인 Gemma 4 12B에는 해당하지 않을 가능성이 높다(추정)**. 그래서 기동뿐 아니라 실제 LoRA 요청 처리까지 확인한다 |
| LoRA를 NVFP4로 양자화할 필요는 없다 | **사실** | LoRA는 bf16으로 유지 |
| 우리 조합(Gemma 4 Unified + RTX PRO 4500 + vLLM 0.32 + LoRA 3개)이 공식 검증됐다 | **아님** | 공식 예제는 Qwen3-8B. 우리 조합은 §22 시험으로 확인 |
| 우리 공개 후보는 ModelOpt 형식이다 | **아님** | 다운로드 중인 RedHatAI·coolthor는 **compressed-tensors**(llm-compressor) 형식이다. ModelOpt 형식은 `vrfai/gemma-4-12B-it-nvfp4`(11.7GB). 둘 다 vLLM이 지원한다 |

순서는 GPT 제안과 같다.
1. NF4 + 현재 LoRA 기준선(완료)
2. NVFP4 + 같은 LoRA(변환본) 14건 비교와 속도(§22, 진행 중)
3. 크게 다르면 ModelOpt NVFP4 QLoRA 재학습 검토(데이터 담당, Gemma 4 적용은 미검증)

### 23.4 외부 의견(GPT) 2차 지적 반영

| 지적 | 판정 | 반영 |
| --- | --- | --- |
| "NVFP4용 표준 QLoRA 도구가 없다"는 틀림 | 맞음 | 채팅 첫 설명의 오류였다. §23.1에 ModelOpt를 1순위로 명시 |
| "반드시 BF16으로 풀어 학습"은 틀림 | 맞음 | ModelOpt QLoRA는 압축된 FP4 가중치를 유지한다(`--compress True`, `mtq.compress()`) |
| "30~40GB 이상"은 단정 불가 | 맞음 | BF16으로 푸는 방식 기준의 추정이었다. "미검증"으로 수정 |
| "반드시 재학습"은 아님 | 맞음 | 지금 어댑터의 품질이 유지되면 그대로 쓴다. 판단은 §22 결과로 |
| 속도 목적이면 W4A4가 핵심, 공식 QLoRA도 W4A4 | 맞음 | §23.1 표 수정 |
| 한 서버에 NF4용·NVFP4용 어댑터 등록 가능 | 맞음 | 등록은 가능, 품질이 문제라고 §23.2 수정 |
| NVIDIA 벤치(W4A16은 BF16보다 느린 형태 12개 중 10개, W4A4는 12개 중 9개에서 빠름) | 미확인 | 원문을 찾지 못했다. 우리 GPU 실측(§22)으로 대체 |
| "기존 Unsloth NVFP4 체크포인트와 ModelOpt 포맷 호환성 확인" | GPT 착오 | 이 프로젝트는 Unsloth NVFP4를 쓰지 않는다. 시험 대상은 RedHatAI·coolthor(compressed-tensors)다 |

### 23.5 시험 대상 확정 — RedHatAI W4A4만 (2026-11-01)

| 모델 | 만든 곳 | 다운로드 수 | KV 캐시 | 결정 |
| --- | --- | ---: | --- | --- |
| `RedHatAI/gemma-4-12B-it-NVFP4`(W4A4) | Red Hat AI(HF 인증 조직, llm-compressor 개발팀·vLLM 주요 기여사) | 4,442 | `kv_cache_scheme: None` → bf16 유지 | **시험** |
| `coolthor/gemma-4-12B-it-NVFP4A16`(W4A16) | 개인 사용자 | 135 | bf16 | **제외**(다운로드 중단·삭제) |

- 공식 여부: Google 공식 4bit는 `gemma-4-12B-it-qat-w4a16-ct`(INT4 QAT, NVFP4 아님)이고, NVIDIA 공식 Gemma 4 12B NVFP4는 없다(`nvidia/` 저장소 없음).
- 제외 이유: 속도 목적에서는 FP4 연산을 쓰는 W4A4가 핵심 후보이고, coolthor는 출처 신뢰도가 낮다. 다운로드 속도(약 1.7MB/s)로 약 80분이 더 걸린다.
- FP8 KV 확인(외부 의견 반영): NVIDIA ModelOpt 예제 기본 레시피는 KV까지 FP8(`nvfp4_default-kv_fp8`)이지만, 시험하는 RedHatAI 체크포인트는 KV를 FP8로 바꾸지 않는다. 앞서 제외한 FP8은 **가중치 연산** 기준이었다. 재학습을 하게 되면 "NVFP4 W4A4 + bf16 KV"를 기준으로 한다.
- 외부 의견 정정 수용: "W4A16은 BF16보다 12개 중 10개 형태에서 느림" 수치는 출처가 확인되지 않아 판단 근거에서 뺐다. 판단은 우리 GPU의 17턴 실측으로 한다.
