# Decision build05 속도 우선 학습용 TAR

[학습용 TAR](/home/roma/Downloads/soomac-decision-r5-compound-semantics-20261101-build05-speed.tar) · [실제 완료 영수증](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_package_test_20261101_build05_speed/DECISION_READY.json) · [계획](2026-11-01_speed_first_training_plan.md) · [작업 기록](2026-11-01_training_package_build_worklog.md)
→ **2026-11-01 18:05 KST: 새 TAR를 서버에 업로드해 본 학습을 시작할 수 있다. 원격 업로드·실행은 사용자가 담당한다.**

## 파일과 검증 결과

파일은 `soomac-decision-r5-compound-semantics-20261101-build05-speed.tar`, 크기는 **6,998,482,944바이트(약6.52GiB)**이다. SHA-256은 다음과 같다.

```text
493acef5df21cf613eeee8be68da8535244980a062a31d961397fccf1d5b1298
```

이미지 `soomac/decision-r5:20261101-build05-speed`의 ID는 `sha256:422b9b440f92e04ae441e56418bac6aa5b2c769bdc658fb1cf4c9c0596ea9de4`다. 실제 Docker load가 exit0이고 로딩 뒤 ID가 일치했다. 기본 명령은 `train`, 사용자 UID는 `1000:1000`, 규격은2.9다. 레이어57개·Docker/OCI 설정 연결·실제 내용 해시·전체 TAR SHA 검증이 통과했다. 기존 build04 및 이전 TAR는 삭제하지 않았다.

## 무엇을 바꿨는가

1. 서버가 준 실제 VRAM 예산을 사용하며 장치·서버 bytes/fraction 한도 중 최솟값으로 다시 제한한다. 기존 이미지의44GB 고정 요청은 기본값에서 해제했다. 원격 allocator 밖 여유분은1GiB다.
2. 기울기 체크포인팅(gradient checkpointing) OFF/ON에서 microbatch1·2·4·8을 유한하게 비교한다. OOM이면 같은 모드의 더 큰 후보를 중단한다. 유효 배치16을 실제로 누적한 optimizer update 처리량 중 가장 빠른 정상 후보를 선택한다.
3. 본 학습 첫5step에 full-state를 저장한다. 이후50step·epoch 경계·마지막 저장과 strict native resume는 유지한다. OOM이라고 batch·설정을 몰래 바꿔 같은 출력에 재시작하지 않는다.

데이터는 최신 `compound_semantics_20261101_r2/dataset`, MANIFEST SHA `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa` 그대로다. history·prompt·few-shot·정답·6epoch·학습률·LoRA 구조·NF4/BF16은 바꾸지 않았다. 기존 외부 모델 template 차이 때문에 중단되던 문제를 막는 봉인 processor도 유지했다.

원인: 고정된 메모리 요청과 동일한 checkpointing 설정만으로는 실제 장치의 빠른 후보를 선택하지 못했다.
결과: 원격 첫 실행에서 실제 허용량으로 후보를 다시 측정한다. **설정 변경과 검증 완료가 원격 속도 향상 보장은 아니다.**

## GPU 시험이 증명한 범위

로컬32,000,000,000bytes·여유1GiB에서 Decision 폐기 profile은 OFF1 OOM / ON1 PASS / ON2 OOM으로 종료했다.4·8은 해당 모드 OOM 뒤 중단되어 실측하지 않았다. 정상 후보의 최장 입력16개 누적 update 한 표본은325.0245초였으며, 원격이나 P/Q의 속도·전체 학습 ETA로 환산하지 않는다.

실제 optimizer10step의 저장5→의도적 실패→재개10→의도적 실패→마지막 step native 복원은 통과했다. adapter·optimizer·scheduler·RNG의 실제 로드 후 해시가 모두 일치하고 원래 checkpoint도 그대로다. 복원 smoke는 batch1/accumulation1이므로 유효 배치16의 장기 복원이나6epoch 완주를 시험했다고 하지 않는다.

실제 로그 Codex CLI 교차검수는 **TAR 내보내기 진행에 조건부 찬성**이었다. 원격 속도·원격 예산 준수 실측·6epoch 완주·새 모델의 시나리오 합격은 아직 미검증이다. CLI 실행 헤더는 `model: gpt-6-sol`, `reasoning effort: high`이며 검수 모델의 자기 식별 문구 대신 이 실행 기록을 따른다.

## 로컬 후속 작업

TAR 완성 뒤에만 로컬 Response→Recommendation을 순차 실행한다. 최신 로컬 후보는 전체 process32GB/allocator 밖512MiB/GC auto이며 원격1GiB와 구분한다. 검수된 실행기·두 설정은 `runs/r5_package_test_20261101_build05_speed/local_operator/`에 읽기 전용으로 별도 보존한다. source 수정으로 장시간 학습의 재개 식별자가 바뀌지 않게 한다. 두 Agent 각각의 GPU 실측·optimizer 진행·첫 full-state 저장 확인 전에는 전체 Goal을 complete로 표시하지 않는다.

## 용어 정리

- 기울기 체크포인팅(gradient checkpointing): 활성값 일부를 역전파에서 재계산해 메모리를 줄이는 선택이다.
- 전체 상태 체크포인트(full-state checkpoint): adapter와 optimizer·scheduler·난수 상태를 함께 저장한 복원 지점이다.
- 유효 배치(effective batch): optimizer update 한 번에 누적하는 예시 수이며 여기서는16이다.

(내가 의역해보자면, **서버에서 바로 학습시킬 상자는 완성됐고**, 실제 서버에서 몇 시간 걸릴지는 첫 속도 측정으로 확인해야 한다.)
