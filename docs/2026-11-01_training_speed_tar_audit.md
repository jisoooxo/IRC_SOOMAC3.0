# build04 TAR 학습 속도 원인 점검

[실제 TAR 제작 기록](2026-11-01_decision_build04_processor_fix.md) · [학습 작업 기록](2026-11-01_training_package_build_worklog.md)
→ RAM 부족으로 단정하지 않는다. 실제 이미지의 설정, 원격 진행 수치, 아직 측정하지 못한 병목을 구분한다.

## 1. 범위와 실제 확인 자료

2026-11-01 KST. 사용자 요청은 현재 원격 Decision 학습이 약80시간으로 표시되는 원인과 TAR 설정 점검이다. **읽기 전용 진단**이며 학습 중단·재시작, 새 GPU 시험, 설정·데이터·프롬프트·adapter 변경, TAR 재빌드, ROS, commit/push를 하지 않았다. 문서와 `/tmp` 진단 자료만 추가한다.

- 실제 파일: `/home/roma/Downloads/soomac-decision-r5-compound-semantics-20261101-build04.tar`, 6,998,478,336bytes. Docker save 메타데이터와 상위 레이어의 실제 파일을 읽었다. 단순히 현재 소스를 이미지 내용이라고 가정하지 않았다.
- 이미지의 `common/train_agent.py`, `core.py`, `budget.py`, `profiling.py`, `verification.py`, `loss.py`, `lab_progress.py`, `remote/run_training.sh`는 현재 source와 봉인 bundle 모두 byte 동일했다. 내장 Decision system과 processor config도 봉인 bundle과 동일했다. RELEASE SHA는 `f709f5c69264b0030f4f039016b29bc3de46bcac0027a518b8a99bff45372def`다. 이번 점검은 전체 TAR SHA·모든 layer payload 재검증이 아니다.
- 원격 작업 ID: `8b07d54d930b4a5992de1b9075bd1f87`. 서버 GET `/api/jobs`, `/api/snapshot?id=...`, `/api/overview`로 실제 상태를 읽었다.
- 증거: `/tmp/soomac_tar_speed_audit.py`, `/tmp/soomac_tar_speed_audit_evidence.jsonl`, `/tmp/soomac_lab_readonly_snapshot.json`, `/tmp/soomac_lab_readonly_snapshot_later.json`, `/tmp/soomac_lab_readonly_overview.json`.

규격 원문은 Downloads의 `TRAINING_IMAGE_SPEC.ko_v2_9 (2).md`다. §1은 서버 장치를 RTX PRO 6000 Blackwell Max-Q로 명시하고, §4는 공유 슬롯이 소프트웨어 메모리 예약이며 MIG/연산 독점이 아님을 명시한다. 웹 설정의 CPU8/RAM16GiB/shm4GiB는 예시이지 모델 속도 보증값이 아니다.

## 2. 원격에서 실제 관측한 값

| 항목 | 실제 확인 | 해석의 경계 |
| --- | --- | --- |
| GPU | RTX PRO 6000 Blackwell Max-Q Workstation Edition, 전체97,887MiB | 사용자가 앞서 말한 A6000과 다르다. 최종 하드웨어 판단은 이 조회값을 따른다 |
| 서버 작업 GPU 예산 | 50,246,713,344bytes = 약46.80GiB | 실제 PyTorch 상한과 같지 않다 |
| 자원 요청 | CPU8, RAM16g, shm4g, shared | 요청값만으로 병목을 확정하지 않는다 |
| 진행 | 6/1638step, 96sequences, 777,906tokens, 학습시간1056.1856초 | 평균176.03초/optimizer-step, 약8,103tokens/sequence, 유효 배치16 |
| 재조회 | 같은 attempt의 7/1638step, 112sequences, 학습시간1231.9188초 | 6→7도 약175.73초다. 2step 순간의 숫자만 본 것은 아니다 |
| 메모리·사용률 | GPU약34.5GiB·100%, 서버 RAM총62.4GiB 중 약7.9GiB 사용 | **둘 다 호스트/장치 전체 수치다. 컨테이너 단독 지표가 아니다** |
| 상태 | 컨테이너 running, oom_killed=false, 공개 보드상 실행 GPU 작업1개 | PyTorch profile 내부 OOM, 스왑, 보드 밖 GPU 프로세스, 전력·열 제한 여부까지 증명하지 않는다 |

원인: frontend `app.js:326-345`가 GPU 장치 수치와 `system.memory_total - system.memory_available`을 표시한다.

결과: 이전 답변의 「작업이 RAM7.9GiB를 사용한다」는 해석은 정정한다. 현재 자료에는 RAM16GiB 상한이 지연을 일으켰다는 증거가 없지만, 컨테이너 압박·스왑까지 검증한 것은 아니다.

## 3. TAR 안에서 확인한 성능 관련 설정

### 3.1 웹보다 작은 메모리 예산을 별도로 고정한다

`remote/run_training.sh:70-73`은 `--budget-bytes 44000000000 --batch auto`를 실제 명령에 넣는다. `SOOMAC_GPU_BUDGET_BYTES` 환경 변수만 바꿔도 이 명령 인자는 변하지 않는다.

`common/budget.py:11-22,48-54`는 요청값·실제 VRAM·서버 bytes·서버 fraction의 최솟값에서 별도2GiB를 뺀다. 실제 서버 조회값을 이 함수에 CPU로 대입했다.

```text
서버 허용 예산                 50,246,713,344bytes ≈ 46.80GiB
이미지 자체 전체 프로세스 예산  44,000,000,000bytes ≈ 40.98GiB
이미지 PyTorch allocator 상한  41,852,516,352bytes ≈ 38.98GiB
```

원인: 이전 사용자 요청44GB와 allocator 외2GiB 여유를 보존한 정책이다. 우발적 계산 오류라고 확인된 것은 아니다.

결과: 서버 허용량 전부를 쓰지 않으며, 같은 TAR로 UI만 단독 GPU로 바꾸어도 내부44GB 한도는 남는다. **이 때문에 원격에서 어떤 배치가 탈락했는지는 원격 PROFILE이 없어 미확정**이다. 상한 변경만으로 배치2/4 성공이나 특정 배수의 가속을 보장하지 않는다.

### 3.2 배치 자동 탐색은 실제로 연결돼 있지만 장시간 최적화 시험은 아니다

`train_agent.py:160-179,239-247`은 배치1→2→4→8을 유한 탐색하며 첫 OOM 뒤 큰 후보를 시험하지 않는다. 선택 후 `accumulation=16//batch`를 적용한다. 이전 r3 wrapper의 「배치1 강제·탐색 미연결」 결함을 이 TAR 원인으로 재사용하면 틀리다.

`profiling.py:21-42,109-119`는 최장/넓은 정답 범위의2~3개 probe를 사용하고 마지막 reference case의 `batch/seconds[-1]`로 비교한다. 가장 빠른 후보의5% 이내에서 큰 배치를 고른다. 메모리 안전성에 유용한 유한 시험이지만 반복 정상상태의 처리량·평가·저장까지 최적화했다고 볼 수 없다. 현재 원격 microbatch는 아직 조회하지 못했다. 관측된 유효 배치16으로 microbatch를 역추정하지 않는다.

### 3.3 짧은 사용자 발화와 달리 실제 입력은 약8천 토큰이다

봉인된 최신 train Decision은4360행, 인코딩 길이 중앙값8069/p958264/최대8608, 1epoch 합35,348,919tokens다. 이 값은 기존 고정 `validation/TOKEN_MASK_ROWS.jsonl` 집계이며 원격 관측 평균약8103tokens와도 규모가 맞는다. 고정 processor를 CPU로 사용한 빈 사용자입력 `{}`의 시스템·대화 양식 길이는7933tokens였다. 이는 모든 행에서 정확히7933개의 공통 prefix가 적중한다는 측정이 아니라 고정 설명의 규모를 보는 진단이다.

`train_agent.py:85-96`, `core.py:55-81`은6epoch이며, Decision 총 입력 반복량은 약2.12억tokens다. `core.py:180-240`은 전체 지침·입력·정답을 인코딩하고 history를 자르지 않는다. `loss.py:4-21`은 불필요한 출력 logits를 이미 줄이지만 transformer 내부의 전체 입력 계산을 제거하지 않는다.

원인: 학습 예시의 대부분을 긴 고정 지침·few-shot이 차지하고, 이를6번 반복한다.

결과: 행수만으로 학습량이 작다고 판단할 수 없다. 26,160개 예시를 약0.09sequences/s로 처리하면 약80.7시간이므로 표시 수치는 현재 처리량과 산술적으로 맞는다. **이 산술이 현재 처리량의 최적성을 증명하지는 않는다.** 지침을 정리하더라도 사용자 요구인 멀티턴·수량 보존·복합 commit·제약·few-shot을 임의 삭제하지 않는다.

### 3.4 메모리를 아끼는 대신 계산하는 설정과 초기 점검이 있다

`core.py:510-585`는 NF4 가중치/BF16 계산, SDPA, gradient checkpointing, paged AdamW8bit이다. 명시적 CPU 모델 offload는 없지만 optimizer paging 여부까지 배제하지 않는다. SDPA라는 이름만으로 FlashAttention이 꺼졌다고 단정하지 않는다. 실제 attention kernel과 처리량은 미측정이다.

`core.py:673-695,810-820`은 첫10 optimizer-step에서 학습 파라미터 gradient의 finite/nonzero를 검사하는 hook을 둔 뒤 제거한다. `training_step`의 token 집계는 `.item()`으로 host 값을 읽고, optimizer 경계에서 CUDA 동기화도 한다. 이들은 실제 추가 작업이지만 **시간 비중은 아직 측정하지 않았다**. 첫10step 이후 시간을 별도로 비교해야 초기 검사 비용을 전체 학습 속도로 일반화하지 않는다.

### 3.5 평가·저장은 현재 느린 training-step의 직접 설명이 아니다

평가/저장은50step마다 및 epoch 경계마다 수행된다. `core.py:634-640,725-761`의 timer는 optimizer-step 구간이며 평가·저장 구간을 포함하지 않는다. 따라서 평가/저장은 이후 실제 완료 시각을 더 늦출 수 있지만, 아직6~7step인 현재의176초/step을 설명하는 원인으로 세지 않는다. 반복 heartbeat도10초마다 상태를 다시 출력할 뿐 재학습이나 모델 호출 반복이 아니다.

## 4. 아직 필요한 증거와 중단 전 조건

1. 원격 `/checkpoints/PROFILE.json`, `INPUT_IDENTITY.json` 또는 초기 `PROFILE_RESULT` 전체: 실제 microbatch·누적·OOM·profile 시간 확인. GET snapshot은 최근 로그만 제공했다. 상세 조회 API는 웹 접근 검사를 요구했고 우회하지 않았다. 브라우저 스킬 확인 결과 연결된 browser도 없어 전체 초기 로그는 아직 확보하지 못했다.
2. 같은 attempt의10step 검사 제거 이후 정상상태 학습 속도: 현재까지의176초/step이 유지되는지 확인. 자동으로 중단·재시작하지 않는다.
3. 다음 별도 승인 시험이 필요하다면 attention kernel·gradient checkpointing·microbatch/상한을 한 항목씩 계측한다. RAM만 늘리거나 모든 설정을 한꺼번에 바꿔 원인을 잃지 않는다. 추론 prefix KV cache는 가중치가 바뀌는 이 학습의 재사용 가속 수단으로 넣지 않는다.
4. 중단한다면50step 저장 주기와 실제 `.lab-monitor/resume.json`을 확인한다. 지금 화면의5개 파일/약5.9KiB는 학습 가중치·optimizer 저장 완료의 증거가 아니다. 동일 설정 full-state resume와 설정 변경 재학습은 구분한다.

**현재 결론:** CPU로 잘못 학습하거나 배치를 무조건1로 고정한 TAR라고 확인되지는 않았다. RAM 상한을 원인으로 확인하지도 못했다. 긴 지침·6epoch·계산 재수행 정책은 실제 비용이며, 실제 서버보다 낮은 내부 VRAM 한도와 짧은 batch 선택 시험은 성능 재검토 대상이다. 특정 하나가80시간의 지배적 병목이라는 결론은 원격 PROFILE 및 정상상태/커널 계측 전에는 보류한다.

독립 Codex CLI `gpt-6-sol/high` 교차검수는 **조건부**다(`/tmp/soomac_tar_speed_deep_review.out`). root는 지적 근거의 실제 `remote/run_training.sh:70-73`, `budget.py:11-22,48-54`, `profiling.py:36-42,113-119`, `core.py:673-695,715-720,810-820`와 frontend의 host 집계 코드를 직접 대조했다. RAM을 단일 원인으로 단정하거나 재시작을 권할 근거가 없다는 제한을 유지한다. 이 검수는 실제 원격 PROFILE이나 kernel 계측을 대신하지 않는다.

16:59 KST 후속 GET 조회에서도 같은 작업이 running,8/1638step,128sequences,누적학습1410.4045초였다. 7→8은 약178.49초다. 이 시점은 아직10step 검사 제거 전이다. 학습 중단/설정 변경은 없으며, 원격 병목 확정과10step 이후 비교는 미완료다.

## 5. 후속 사용자 기준 — 로컬·원격 모두 속도 우선

사용자가 「gradient checkpointing 할 필요 없는데」「메모리 안 아껴도 되는 상황」「확실하게 빠르게 학습」「우리 로컬도 마찬가지」라고 우선순위를 명시했다. **메모리를 적게 쓰는 것 자체를 목표로 삼지 않고, 실제 장치·서버 허용량 안에서 OOM을 피하면서 처리량을 높이는 것**을 후속 비교 기준으로 둔다. 이 문장은 현재 학습을 중단하거나 새 GPU 시험을 실행했다는 뜻이 아니다.

1. 상한 완화는 우선 검토한다. allocator 상한은 GPU 물리 용량·서버 허용 범위·드라이버/화면/비-PyTorch 여유를 구분해 잡으며 RAM 증설과 혼동하지 않는다. 상한 숫자만 높여서는 현재 batch/계산량이 자동으로 바뀌지 않는다.
2. gradient checkpointing OFF를 비교 후보에 포함한다. `prepare_model_for_kbit_training`과 `TrainingArguments` 양쪽을 함께 확인해야 실제로 꺼진다. 이 두 enable 경로가 있다는 사실만으로 계산이 두 번 중복된다고 주장하지 않는다. OFF는 학습 복구용 checkpoint 저장을 끄는 것과 별개다.
3. 로컬 최신 PROFILE은 checkpointing ON에서도 batch2가 allocator26.87GiB 한도로 OOM이었다. 물리 여유3.89GiB만으로 OFF가32GB에 들어간다고 보장하지 않는다. 원격도 장치 전체95.6GiB가 아니라 작업 허용46.8GiB가 기준이다.
4. 비교 채택은 ON/OFF와 microbatch 조합의 실제 sequences/s·최장 입력 역전파·optimizer·메모리 peak로 한다. OFF가 들어가더라도 ON의 더 큰 batch보다 빠른지 확인한다. 유효 배치16을 유지하더라도 미세 수치·loss 가중치까지 완전히 동일하다고 보장하지 않는다. 현재 봉인 설정과 다른 실행의 재개는 기존 identity 검증을 임의로 우회하지 않는다.

실제 상한 변경·checkpointing OFF 시험·학습 재시작·새 TAR 제작은 아직 하지 않았다. 속도 우선 기준만 기록했다.
