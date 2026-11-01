# 로컬 학습 보존 중단·Unsloth/attention 시험과 원격 적용 확인

[작업 기록](2026-11-01_training_package_build_worklog.md) · [현재 학습 계획](2026-11-01_speed_first_training_plan.md) · [Unsloth Gemma4](https://unsloth.ai/docs/models/gemma-4/train)
→ **사용자 승인으로 로컬 Response→Recommendation만 보존 중단하고 가속을 비교한다. 원격 Decision은 변경하지 않는다.**

## 최신 우선 목표 — 품질 검증 후 최종 실행 구성 확정

사용자 최신 지시로 기존 속도 우선 목표를 대체한다. **3개 어댑터의 올바른 학습·미학습 실제 기능이 최우선이며, 11월4일 step 완료는 품질 합격을 대신하지 않는다.** 본 학습 시작/중단·원격 교체·checkpoint 삭제/덮어쓰기·기존 환경 변경은 별도 승인이 필요하다. 폐기용 정확성·성능·저장재개 실험은 추가 재승인 없이 GPU 단일 프로세스로 진행한다. 도구의 기존 Goal 표시는 paused이며 objective 변경/재개 API가 없어 새 goal 생성이 거절됐다. 완료로 조작하지 않고 이 최신 지시를 실제 작업 기준으로 삼는다.

1. **입력/학습 의미:** 최신 sealed r5 데이터에서 D/Q/P 각각 길이가 가장 짧은 행·가장 긴 행·history가 많은 멀티턴 행을 ID로 사전 고정한다. 서로 중복되면 다음 독립 행을 택한다. 원본 input/history/labels/tokenizer/template/max_length를 변경하지 않는다. 동일 초기 LoRA·native CE·마스킹·gradient accumulation16·optimizer/clip/LR 의미를 HF/Unsloth에 대조한다. 진단 dropout0과 실제 dropout0.05를 구분한다. 대표 행별 loss/logits/gradient norm·relativeL2·cosine을 기록하며 finite만으로 합격하지 않는다.
2. **판정 사전 고정:** 입력·labels·target 위치·trainable 대상은 exact 일치가 필수다. native CE는 원래1e-3 loss 관문을 완화하지 않는다. gradient 비교의 기존 엄격 관문을 우회하지 않으며 A/A 변동과 A/B 차이를 함께 본다. 큰 차이가 있으면 numerical-equivalence 미통과로 남기고 training-quality를 기능 검증과 구별한다. toy CPU normalization 시험은 실제 모델 정확성의 대체물이 아니다. baseline의 Gemma4Unified `accepts_loss_kwargs=False`와 Trainer의 microbatch mean/GA divisor가 새 실행기와 같음을 실제 실행 정보로 확인한다.
3. **저장/재개:** 새로운 폐기용 Unsloth 출력에서 committed full-state 저장→의도적 실패→새 프로세스 재개를 검증한다. adapter/optimizer/scheduler/RNG·step의 native 복원과 최종 표준 PEFT 재로딩을 각각 확인한다. 기존 HF checkpoint를 Unsloth로 이어도 된다고 가정하지 않는다. 6epoch production 정책이나 정상 checkpoint는 변경하지 않는다.
4. **누출 없는 기능 평가:** 기존 train/validation 및 보강 시나리오를 학습에 사용한 사실을 인정한다. 사용자 원문 시연은 필수 회귀이지 미학습 holdout이라고 부르지 않는다. 별도의 새 시나리오 그룹·고유 발화/상태를 동결하고 train/validation의 발화 exact match·입력 exact match·동일 대화 그룹을 검사한다. D: commit/변경량 보존/제약·pending 구별. Q: 취향·제약에 맞는 제안/근거 없는 새 양 normal. P: 실제 변경·양·시작/완료 사실성·불필요한 추천/실행 질문 없음. 안전·시연 필수 케이스는 전부 통과, 나머지 작업별 성공률은95% 이상으로 사전 고정한다. 사람 의미 검수와 평가기 oracle의 양성/음성 대조를 포함한다. 일반화의 완전 무누출을 exact 검사만으로 주장하지 않는다.
5. **실행/ETA:** 추가 Attention 최적화는 correctness 미통과로 production 제외한다. 기존 SmartGC batch1 nativeCE의 warmup2+측정3 update를 신규 측정한다. cold값을 warm이나 평균 production ETA로 혼동하지 않는다. 원격 Decision은 기존 작업을 유지하고 이미지 버전·RAM/GPU 예산·복원 호환·동일 원격 실측을 별도 판단한다. 현재 모델 학습 품질 미합격/본 학습 미시작과, 실행기 기술적 준비를 분리해 보고한다.

진행 중: warm 측정 첫 명령은 도구가 full 모드에서 측정3회를 요구해 GPU 시작 전에 거절됐다. 정책을 바꾸지 않고 기존 도구 요구대로3회로 정정했다. 실패 로그와 새 실행 로그를 구별해 보존한다. 측정은 약23~26분 예상이며, 그동안 CPU에서 대표 행·기능 평가 기준·원격 패키지를 확인한다.

Cause: 기존의 한 행 eval loss 일치·유한 gradient·epoch 완료는 학습 의미와 실제 대화 기능을 보증하지 않는다.
Effect: 정확성·복원·기능·속도 관문을 분리하고 통과한 범위만 최종 실행 후보에 포함한다. production은 승인 전 실행하지 않는다.

### 품질 검수 실제 진행 기록

`tools/training_acceleration/quality_gate.py`와 CPU 시험을 추가했다. 대표 행은 D8608/8020/8602, Q4201/3747/4129, P8430/1507/7678토큰이며 각 장문·단문·history 다수행이다. sealed encoder·labels·history를 그대로 사용하고 새 attention mask를 자의로 만들지 않는다. encoder가 반환하는 입력은 전부 유효 토큰이며 실제 학습 collator와 같은1 마스크를 보존해 padding8 collator에 전달하고, 최종 padded mask hash까지 A/B 대조한다. 수치 비교는9행+각 agent 장문 A/A 반복3행으로 총12개씩이며, 실제 모델 GPU 비교 전이다. 준비본은 `/tmp/soomac-training-accel-20261101/quality_prepared_v2/QUALITY_PREPARE.json`이다.

기존 holdout을 모든 agent의 train/validation과 사용자 최신 발화·과거 user 발화까지 엄격 대조했을 때 D286/286, Q0/13, P5/20만 남았다. **같은 짧은 발화를 다른 상태에서 쓰는 것 자체를 오염으로 단정하지 않는다.** 다만 독립적인 새 평가라는 근거로는 채택하지 않았다. 원본 행을 삭제하거나 training split을 바꾸지 않았다. 입력 exact/발화 정규화/명시적 대화 그룹 검사는 의미론적 근접 중복 검사를 대신하지 않는다.

별도 `heldout_cases.py`로 D/Q/P8개씩 새 합성 기능 평가 후보24건을 작성했다. `new_holdout_v1`은 보존한다. Codex(gpt-6-sol/high) 의미 검수는 D06의 지연/거절 모호성, P02의 버섯만/채소 전체 실행 대상 모호성, P06의 이미 고른 양/이번 턴 변경량 불일치를 지적했다. D06은 실행 확인 질문 취소를 명시하고, P02는 양파 포함 채소 전체 실행을 명시하고, P06은 이번 턴 변경을 비웠다. 수정본은 `/tmp/soomac-training-accel-20261101/new_holdout_v2/NEW_FUNCTIONAL_CASES.json`이며 입력·정규화 발화·그룹 중복0과 target schema 통과를 확인했다. 독립 의미 재검수/실제 adapter 추론은 아직이다.

`functional_eval.py`는 freeze hash·전체 case ID 완전성·스키마·기존 의미 평가기를 확인하고 Q의 새 normal 기본량 oracle을 strict 비교한다. 고정 Response 문장과 비교하지 않는다. 사람이 보는 사실성·반복·필요 없는 추천/질문은 여전히 별도 필수이며, 자동 통과로 최종 품질 합격을 만들 수 없다. CPU 양성/음성 대조5건, 품질 helper5건이 통과했다. 이는 실제 모델 평가 점수가 아니다. `adapter_reload.py`는 새 프로세스에서 표준 HF/PEFT로 smoke 최종 adapter를 읽고 tensor exact 일치와 forward를 검사하도록 작성했으며 아직 GPU 미실행이다.

최초 quality 도구 Codex 검수에서 읽은 동시 편집 전 버전의 FP64 함수 미사용/zip 완전성 지적은 수정했다. 이후 실제 source의 `_chunked_diff`, 성공 상태+12행+hash 확인을 root가 재확인했다. 실제 padded mask hash·모든 텍스트/그룹 키 대조·과거 승인 문구 정정도 반영했다. 두 번째 검수는 진단 probe 실행에 조건부 동의하며, loss 통과를 gradient/기능/최종 품질 통과로 확대하지 말 것을 지적했다. loss·gradient·A/A 결과 집계도 분리했다. 검수 로그는 `/tmp/soomac_quality_gate_review.out`, `/tmp/soomac_quality_final_design_review.out`이다.

## 과거 승인과 단계 — 아래 기록은 최신 production 실행 권한이 아니다

과거 사용자는 **최종 Unsloth 기반 학습 시작까지 진행해 ETA를 줄이는 Goal**과 멀티에이전트 병렬 작업을 명시했다. 환경/벤치마크/원격 호환성을 각 기존 Codex sol/high 작업자에게 분담했다. GPU 시험은 root만 순차 실행했다. 이전 미완료 Goal이 paused로 남아 최초 새 objective 등록은 거절됐다. 이후 당시 “paused 없애라” 메시지 뒤 get_goal에서 goal=null을 확인했고, 새 목표를 정식 등록하여 active를 반환받았다(createdAt1793527317). 이는 역사적 기록이며 이후 기다림으로 paused가 됐다. **현재 작업 권한은 위 최신 품질 우선 절만 따른다. 본 학습은 별도 승인 전 시작하지 않는다.**

2026-11-01 KST. 사용자는 첨부 분석의 학습 유지 권고를 읽은 뒤에도 “로컬 중단하고, 데이터·정상 체크포인트 보존한 채 가속 시험 진행”을 선택했다. 첨부는 `/home/roma/.codex/attachments/aca1517e-ff2a-42b0-8e1f-5c4c615bd5d6/Pasted text.txt`를 끝까지 읽었다. 설정 오류 미확인·현재430.4nonpad tok/s·SDPA/FP32/reentrant 가설과 이미 적용된 scored logits 절약을 구분한다. 가설을 결함이나 측정된 가속으로 보고하지 않는다.

1. 정상 checkpoint와 정확한 로컬 프로세스 확인: 완료. checkpoint5 필수7파일 SHA 검증 exit0.
2. 로컬 launcher/trainer 프로세스 그룹 정상 종료·GPU 반환 확인: 완료. 정확한 PGID914967에 SIGINT, 두 PID 종료·compute process0·잔여 VRAM1193MiB 확인. 마지막 실행로그는6step까지 왔으나 검증 복귀 지점은5step이다. 기존 pipeline_state의 running은 SIGINT로 최종 갱신되지 않은 잔류 표시이며 실제 생존을 뜻하지 않는다.
3. 격리 환경 Unsloth 설치/정확한 unified12B 호환 확인, baseline attention 실측: 미완료.
4. 같은 입력/label/LoRA/유효배치의 제한된 비교·저장/재개·PEFT 재로딩: 미완료.
5. 실제 원격 예산·GPU·CUDA/Docker 호환 대조와 권고: 미완료. 원격 중단·새 TAR 배포·본 학습 재시작은 이 시험 범위에서 자동 수행하지 않는다.

## 보존 대상과 중단 대상

보존: 최신 `compound_semantics_20261101_r2/dataset`과 SHA, 봉인 build05 release/TAR, 모든 정상 checkpoint/adapter/log, 기존 `gemma4_env`와 운영 환경, VLM/runtime/history/prompt/정답/LoRA dropout0.05.

정확한 중단 대상은 PID914967의 r5 local_operator launcher와 자식 PID914979의 Response trainer다. 확인된 PGID/SID는914967이며 다른 프로세스가 섞이지 않았는지 종료 직전 다시 확인한다. 학습 중인 미저장 update는 취소될 수 있고, 마지막 검증 checkpoint5가 복귀 기준이다. Q 자동 시작을 막기 위해 부모·자식 모두 종료한다. 파일은 삭제하지 않는다.

체크포인트: `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_local_speed_20261101_build05/response/attempt-6b570c7c5d8a49e8b9a64bbe79b439e6/response/checkpoint-5`. CPU 해시 검증은 실제7개 state와 설정/step을 검사했으며 GPU native 복원 시험은 아직 아니다.

## 유한 시험 기준

**최신 사용자 우선순위:** prefix-sharing 추가 연구·측정·구현은 보류한다. 담당 작업자를 interrupt했고 기존 문서는 보존한다. 현재 우선 범위는 Unsloth 학습 실행기·실제 GPU 비교·로컬 순차 학습 및 원격 Decision 배포 패키지 준비다. 원격 기존 작업 중단·production 전환은 별도 승인 전까지 금지한다.

- 기존 환경을 변경하지 않고 별도 환경/폐기 출력에서 실행한다. baseline과 대안이 GPU를 동시에 쓰지 않는다. GPU lock과 실제 compute process 반환을 확인한다.
- 동일 pinned base/processor와 사전 인코딩 input_ids/attention_mask/labels·history를 사용한다. dropout·rank·alpha·epoch·학습률을 가속 명목으로 변경하지 않는다. prompt 전체 attention을 보존한다.
- 먼저 실제 attention operator·dtype·시간을 측정하고 Unsloth를 우선 비교한다. 모든 대안을 한꺼번에 설치/패치하지 않는다. max입력과 혼합길이·같은유효배치를 구분하고 warmup 뒤 처리량·메모리·loss/grad를 비교한다.
- 정상 저장·재개·기존 PEFT 로딩을 확인한다. 프레임워크 간 optimizer 상태의 무검증 재사용은 금지한다. 현재 checkpoint와 시험 결과를 합격 조작하지 않는다.
- 준비/비교60~120분을 작업 예산으로 삼되 설치/컴파일이 지연되면 상태를 보고한다. 같은 실패를3번 반복하지 않고 가정을 재평가한다. 무제한 OOM 재시도·환경 덮어쓰기·원격 작업 변경·자동commit/push를 하지 않는다.

원인: 초기 진행량에 비해 남은 학습 시간이 길어 가속 가능성을 제한된 비용으로 확인할 필요가 있다.
결과: 복귀 가능한 상태를 남기고 실제 효과로 전환 여부를 판단한다. 시험 승인은 새 본 학습 재시작이나 데이터 외부 업로드 승인을 대신하지 않는다.
