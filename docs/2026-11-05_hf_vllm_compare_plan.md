# 최신 D/Q/P HF·vLLM 옵션 비교 계획

사용자 승인: 최신 프롬프트와 세 adapter를 GPU에 적재해 HF/vLLM을 여러 옵션으로 다시 비교한다. 운영 source/환경/adapter/기존 TTS·VLM은 그대로 두고 별도 프로세스만 사용한다. 계획 작성 후 실행한다. 예상30~60분.

1. [x] 조건 확정: HF NF4/BF16 캐시OFF/ON, vLLM16K/.58 prefixOFF/ON(invariantON), vLLM16K/.58 prefixON/invariantOFF, vLLM12K/.50 prefixON/invariantON의6조건. HF는 기존soomac_llm, 서버는soomac_vllm_server를 사용한다. 순수 라이브러리 버전 통일 실험이 아닌 실제 배포 경로 비교이며 dtype/quantization/adapter/prompt/token identity를 기록한다.
2. [x] 봉인·도구 검증: 현행 source/원본PEFT/변환adapter SHA와 변환 텐서 동일성, tokenization·grammar·출력상한을 확인한다. 기존36턴에서 D5/Q2/P5 고정입력을 골라 모든조건에서 같은12개를3회 반복한다. 이전 출력은 비교 reference이며 의미 정답으로 승격하지 않는다. 같은요청 반복의 warm cache와 새질문의 공유접두 효과를 구분한다.
3. [x] 실제GPU 비교: 각 조건 새 프로세스에서 Qwen STT/기존HF TTS를 동일 방식으로 상주시킨다. 냉시작 첫12개→warm반복2회, 캐시활성조건의 공통접두 준비, 새callable로36턴 closed-loop를 실행한다. HF cache는 episode 간 유지하되 replay 전용cache는 제거해 memory 중복을 막는다. 전체서버 시작/로드·준비시간은 요청지연과 별도다. vLLM cache는 replay 이후 시작워밍업으로 준비하며 cache상태를 기록한다. cold phase에서도 선행12개간공유접두 적중은가능하므로 모든요청cache0으로 부르지 않는다.
4. [x] 품질·속도 분석: adapter별 latency 중앙값/P95·출력토큰수·stop/길이종료·JSON·동일입력반복변동, 36턴상태/수량/실행계획/route·history/입력삭제 여부, NVML 전체peak/RAM을 보고한다. 오류턴도 실제잘못된state를이어가며 중단하지 않는다. 다른답변길이의요청시간차이를 순수decode가속으로 해석하지 않는다. 물리ROS·VLM·학습·adapter변환/병합·원본환경설치는하지않는다.
5. [x] 종료·보고: 소유scope/port/GPU반환 및 원본보존을 확인하고 옵션표와실제출력/실패자료를worklog에기록한다. 운영설정자동교체는하지않는다. 이번 LLM 시간은 STT/TTS 상주조건의 LLM측정이며 실제음성재생/로봇동작까지의시연시간이아니다.

## 중단 및 해석 기준

GPU 여유<1GiB 또는RAM여유<6GiB가4개연속50ms표본이면 시험소유scope만중단한다. 각scope RAM40GiB/Swap0 및CPU2thread를사용한다. 다른프로세스는중단하지않고관측한다. HF OOM/비정상캐시/길이초과는실패로그보존후해당조건을미검증/실패로표시하며임의로prompt/history/출력상한을줄이지않는다. 최대16384/12288이외 NF4/LoRA를변경하지않는다. MTP/NVFP4는별도모델/정밀도/호환성변경이므로이번엔포함하지않는다.

같은입력/서버설정에서도출력변동이있었던기록을고려해속도순위만으로정확도승격하지않는다. 각조건3회replay와단회closed-loop의기술통계이다. 12K에서는staticbudget와전송token한도를모두검사하고동일입력을자르지못하면실패로남긴다.

## 실행 전 교차검수 반영

일반설계Codex조건부3건 반영: frozen입력token+출력상한은12K에도모두들어감을확인했다. closed-loop추가history삭제는별도실패로표시하고전송/원본history를남긴다. 시나리오events는같아도생성history가달라지면동일prompt성능비교가아니며replay만짝지어비교한다. invariantOFF도평가graph/fixture만실행하며ROS·실물부작용없다. 원본656tensor×3와변환본추가16tensor×3의실제값동일성을CPU에서확인했다.

## 정밀도 대조 추가 (기존6조건 실행 중 발견)

기존HF PEFT로드는1968개LoRA파라미터tensor를FP32로승격한다. vLLM0.32의LoRAConfig는auto를model_config.dtype(BF16)로설정한다. 따라서단순NF4/BF16라벨만으로전체동일dtype라고말할수없다. 원래6조건은배포경로비교로유지하고,7번째`hf_cache_lora_bf16`은시험프로세스에서만LoRA1968tensor를BF16으로맞춘다. 원본weight/환경/source는그대로다. 이조건도12입력3반복+36턴을실행해cast효과와vLLM비교를분리한다. 다른라이브러리/커널차이까지없어지는것은아니다. 원래6조건controller는이미로드된코드로진행하고추가조건은별도순차controller로실행한다.

## 18:38 재개 기록

HF 캐시OFF는 replay36회/시나리오18턴 뒤 GPU여유<1GiB 안전기준에 따라 중단했고, HF 캐시ON은 replay36회+시나리오36턴을 완료했다. 첫 vLLM캐시OFF는 replay6회에서 기록이 끊겼으며 종료 원인은UNKNOWN이다. 18:37 호스트 확인에서 controller/소유scope/GPU프로세스가 모두 없었다. 해당부분기록은 보존하고 `vllm_16k_nocache_r2`로 전체조건을 다시 실행한다. 나머지 미실행4조건과 함께 순차 실행한다. 재개시 source/tokenizer/input 해시는 초기값과 일치했다. 상세 `RESUME_AUDIT.json`.

## TTS 초기화의 외부 다운로드 차단

18:48 HF BF16 대조 준비 중 기존TTS helper의 wetext가 ModelScope 최신캐시 확인/다운로드를 수행했고 일부FST 다운로드 및403재시도가관측됐다. HF_HUB_OFFLINE은ModelScope에적용되지않았다. 원본모델/환경/소스수정은없지만보조캐시는변경됐으므로완전한파일불변을주장하지않는다. 18:50경소유scope만정리하고부분기록보존했다. 두남은조건은시험wrapper `hold_tts_offline.py`로wetext의snapshot조회만기존로컬디렉터리로고정한다. 원래helper의모든합성은text_frontend=False이며frontend.py:134~135에서정규화함수호출을건너뛴다. TTS모델·dtype·합성입력은그대로다. 새결과는`vllm_16k_nocache_r2`와`hf_cache_lora_bf16_r2`에저장한다.

## 종료

7조건 측정 처리완료: fixed replay252회 모두JSON/stop정상, closed-loop6조건36/36완주·HF캐시OFF18턴안전중단. 추가부분시도9회는비교통계제외. 최종source/adapter/환경목록보존및시험GPU반환확인완료. 조건부교차검수범위와예외는[최종보고](2026-11-05_hf_vllm_compare_review.md)에명시했다. 운영설정교체없음.
