# build03 원격 모델 메타데이터 불일치

> **후속 상태:** 사용자가 기존 TAR 기반 수정을 승인해 build03 레이어를 보존한 build04를 제작했다. 동일 공식 메타데이터 입력으로 새 preflight와 실제 GPU 프로파일·step5 복원을 통과했고, 최종 저장/복원·TAR 마감 중이다. [현재 수정 기록](2026-11-01_decision_build04_processor_fix.md)을 우선한다. 아래 ‘미구현’은 원인 조사 당시 상태로 보존한 것이다.

[제작 worklog](2026-11-01_training_package_build_worklog.md) · [기존 build03 검증](2026-11-01_decision_build03_assets_fix.md)
→ 로컬 GPU 저장·복원 성공은 원격 기본 HF 입력 일치의 증거가 아니었다. build03 원격 학습은 현재 실패 상태다.

## 1. 실제 오류와 영향

사용자 제공 원격 로그는 `phase=loading step=0/?`에서 `train_agent.py:218 → checkpointing.py:110 → ValueError: base model drift: chat_template.jinja`이다.

이 호출은 release 검증 중 중단됐다. 모델 설정·프로파일·Trainer 실행은 이후이므로 해당 attempt의 optimizer update는 0회다. 다른 과거 attempt나 원격 체크포인트 전체를 확인한 것은 아니다. 원격 실제 파일 SHA와 선택한 캐시 identity는 로그에 없으므로 추측하지 않는다.

## 2. 직접 확인한 제작 계약 모순

build03 `RELEASE.json`과 최신 dataset MANIFEST는 로컬 `/home/roma/Desktop/sLLM/gemma-4-12B-it`의 파일을 pin한다. 반면 `remote/lab-inputs.json`의 기본 입력은 공식 `google/gemma-4-12B-it` commit `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`이다. root가 해당 공식 commit의 실제 소형 파일 5개와 LFS weight hash 메타데이터를 읽었다. 가중치는 다운로드하지 않았다.

| 파일 | build03/local SHA256 | 공식 고정 commit SHA256 | 결과 |
| --- | --- | --- | --- |
| chat_template.jinja | `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0` | `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` | 불일치 |
| tokenizer_config.json | `09e6222fd7049dae603d0f61a8b10af5a813f22dca10e6cdc6918f76d7661104` | `a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9` | 불일치 |
| config.json | `478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9` | 동일 | 일치 |
| processor_config.json | `6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9` | 동일 | 일치 |
| tokenizer.json | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` | 동일 | 일치 |
| model.safetensors | `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d` | 공식 LFS SHA가 동일 | 가중치 메타데이터 일치 |

공식 template은 18,683bytes, 로컬은 17,466bytes다. 주석뿐 아니라 턴 마감·thinking·tool-call 처리 로직도 다르다. 공식 tokenizer_config에는 `response_template` 필드가 추가돼 있다. 이것이 현재 모든 SFT의 token을 바꾼다고 단정하지 않는다. 반대로 바이트 차이를 무시하거나 기대 SHA만 바꾸고 기존 인코딩 증거가 유지된다고 주장할 수도 없다.

공식 근거: [고정 commit template](https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/chat_template.jinja), [고정 commit tokenizer config](https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/tokenizer_config.json).

원인: 제작 시험은 로컬 pin에 맞는 모델로 수행됐지만, 선언한 외부 기본 HF commit과 로컬 pin을 실제 대조하지 않았다.

결과: build03의 assets·TAR load·로컬 복원 증거는 유지하되, 원격 기본 입력 학습 준비 완료라는 주장은 철회한다. 같은 입력으로 단순 재시도하지 않는다.

## 3. 수정 후보 — 아직 미구현

1. 무코드 대안: 로컬 6개 파일을 그대로 별도 모델 아카이브로 올려 서버 캐시를 바꾼다. 가중치만 23,919,549,408bytes 전송이 필요해 자동 실행하지 않는다. 전체 6개 SHA가 맞아야 한다.
2. 전송량이 작은 수정안: **새 이미지**에 기존 인코딩과 같은 processor/tokenizer 파일들을 봉인하고, 외부의 동일 weight·model config는 계속 검증한다. 원 후보·build03·옛 영수증은 수정하지 않는다. 이 포장 오류 해결에 데이터 재작성·재학습이 필요한 것은 아니다.
3. `PROCESSOR_DIR` 환경 변수만 바꾸는 것은 불충분하다. `common/train_agent.py:58,67`이 이를 모델 경로로 덮어쓰며 profile·smoke·실학습·completion 검증을 모두 맞춰야 한다. 해시 검사 삭제·expected SHA만 교체·원격 파일 덮어쓰기는 하지 않는다.
4. 부정 검수: 외부 template이 달라도 실제 serializer는 봉인된 원본을 쓰는지, 외부 weight/config 변조·봉인 processor 누락/변조를 거부하는지, 원본 token/mask가 같은지, 복원에도 같은 processor identity를 쓰는지 검사한다. CPU 통과를 GPU/서버 성공으로 보고하지 않는다.

Codex CLI `gpt-6-sol/high` 읽기 전용 대조는 exit0이다. 로그 `/tmp/soomac_build03_template_review.out`. 원격 파일 중 최초 불일치만 확정됨, expected SHA 교체 금지, 정확한 전체 모델 업로드는 조건부 가능, 새 봉인 processor 분리는 가능하나 ENV만 바꿔서는 불가능, 해당 호출 update0에 동의했다. root는 관련 실제 줄을 직접 열어 확인했다. CLI 검토 후 추가로 확인한 공식 commit hash 비교는 root 증거이며 리뷰어가 직접 재현했다고 쓰지 않는다. 별도 TAR 에이전트도 검수 중이다.

새 이미지/TAR 제작·원격 변경은 아직 하지 않았다.

별도 `remote_template_failure` 에이전트의 읽기 전용 검수도 완료했다. 새 이미지에 processor 5파일을 봉인하고 외부 weight/config 검증을 유지하는 안은 조건부 찬성이다. `verify_completion`도 release를 재검증하므로 끝까지 같은 processor 선택이 필요하고, `remote/build_export.py`가 그 실제 파일을 context로 복사·검증해야 한다고 지적했다. root가 `train_agent.py:58,67,149`, `core.py:462,520`, `checkpointing.py:180` 경로를 직접 대조했다. 원격 실제 모델 전체 SHA와 수정 이미지의 실제 실행은 여전히 미확인이다.

## 4. 병렬 로컬 작업 — 사용자 최신 변경

사용자는 NVFP4 대기를 철회하고 **Response 의미 추가 검수 → 잘못된 행 국소 제외/교정 → Response → Recommendation 순차 학습**으로 변경했다. 실제 다른 GPU 작업과의 동시 점유는 금지다. NVFP4를 완료했다고 기록하거나 옛 OOM marker를 PASS로 바꾸지 않는다.

Response 두 검수자가 정렬된 실제 `records/response/*.jsonl` 전반/후반을 나눠 읽기 전용 검수한다. root가 실제 행·state·기존 판정을 대조해 채택한다. 수정이 있으면 새 로컬 후보에 반영하며, 불변 Decision TAR의 데이터와 조용히 섞지 않는다. 아직 본 학습을 시작하지 않았다.
