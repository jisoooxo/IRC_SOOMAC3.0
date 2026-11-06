"""Render observed results only; SUMMARY is regenerated separately."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105'
s=json.loads((BASE/'SUMMARY.json').read_text())
names={'hf_nocache':'HF 접두 캐시 OFF','hf_cache':'HF 접두 캐시 ON (기존 FP32 LoRA)','hf_cache_lora_bf16_r2':'HF 접두 캐시 ON (BF16 LoRA 대조)','vllm_16k_nocache_r2':'vLLM 16K/.58 접두 OFF·invariant ON','vllm_16k_current':'vLLM 16K/.58 접두 ON·invariant ON (현행)','vllm_16k_bi_off':'vLLM 16K/.58 접두 ON·invariant OFF','vllm_12k':'vLLM 12K/.50 접두 ON·invariant ON'}
def fmt(x):return '미완료' if x is None else f'{x:.2f}'
def esc(x):return str(x).replace('|','\\|').replace('\n','<br>')
out=['# 최신 프롬프트·D/Q/P HF/vLLM 실측 비교 — 2026-11-05',
'', '근거: [원시 결과·SUMMARY.json](../training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105/SUMMARY.json) · [계획](2026-11-05_hf_vllm_compare_plan.md) · [실제 답변 전체](../training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105/OUTPUTS.md)。',
'', '## 동일 입력 반복 속도', '',
'D5·Q2·P5의 12개 입력을 조건별 3회 실행했다. 아래는 2·3회차의 역할별 응답 완료 시간 중앙값이다(D10/Q4/P10 표본). 첫 토큰 시간(TTFT)은 측정하지 않았다. HF는 시스템 접두를, vLLM은 적중 가능한 전체 입력 블록을 재사용하므로 캐시 정책 자체도 비교에 포함한다. 이 숫자를 새 질문마다 보장되는 지연으로 해석하지 않는다.', '',
'| 조건 | D 초 | 추천 초 | Response 초 | warm 24회 합계 초 | 출력 JSON/종료 |', '|---|---:|---:|---:|---:|---|']
for label,name in names.items():
 x=s['conditions'].get(label)
 if not x:continue
 r=x['replay'];vals=[r['roles'][role].get('warm_repeated',{}).get('latency_s',{}).get('median') for role in ['decision','recommendation','response']]
 out.append('| '+name+' | '+' | '.join(fmt(v) for v in vals)+' | '+fmt((r['warm_all'] or {}).get('sum'))+f" | {r['valid_json']}/{r['total']} JSON, {r['stopped']}/{r['total']} stop |")
out+=['','## 실제 5개 시나리오·36턴 및 VRAM','','| 조건 | 상태·계약 통과 | 전체 LLM 시간 초 | GPU peak GiB | 최소 여유 GiB |','|---|---:|---:|---:|---:|']
for label,name in names.items():
 x=s['conditions'].get(label)
 if not x:continue
 r=x['scenario'];g=x.get('gpu',{})
 out.append(f"| {name} | {r['contract_passed']}/{r['turns']} 관측 / 예정36 | {fmt(r.get('wall_s'))} | {fmt(g.get('peak_gib'))} | {fmt(g.get('min_free_gib'))} |")
out+=['', 'Qwen STT와 기존 HF TTS를 각 조건에 함께 적재하고 각각 저장 음성·문장으로 1회 추론 후 상주시켰다. 위 시간은 **LLM graph 처리 시간**이며 STT 입력·TTS 재생·실물 로봇 동작 시간은 제외한다. VLM 이미지 추론도 이번에는 실행하지 않았다. 따라서 실제 전체 시연 시간이나 VLM 동시 부하 안전성을 새로 보장하는 수치가 아니다.',
'', '시나리오의 발화·기대 상태는 같지만 실제 생성된 답변·history가 달라질 수 있다. HF는 replay 캐시를 제거하고 새 callable로 시작했고 vLLM은 replay 캐시와 공통 접두 워밍업을 유지했다. 이 표의 전체 시간을 동일 계산량의 엔진 속도비로 해석하지 않는다. 순서가 고정된 단회 closed-loop이며 시나리오 계약 통과와 자유 발화 전체 정확도는 다르다.',
'', '## 첫 순회 및 출력 동일성', '',
'첫 순회도 앞선 요청의 공유 접두가 적중할 수 있으므로 모든 요청이 완전 cold라는 뜻은 아니다. P95 등 상세 통계와 출력 토큰 수는 SUMMARY.json에 보존했다.', '',
'| 조건 | D 첫 순회 중앙값 초 | 추천 첫 순회 중앙값 초 | Response 첫 순회 중앙값 초 | 3회 사이 JSON이 바뀐 입력 ID |', '|---|---:|---:|---:|---|']
for label,name in names.items():
 x=s['conditions'].get(label)
 if not x:continue
 vals=[x['replay']['roles'][r].get('first_sweep',{}).get('latency_s',{}).get('median') for r in ['decision','recommendation','response']]
 out.append('| '+name+' | '+' | '.join(fmt(v) for v in vals)+' | '+str(x['replay']['repeat_json_variation'])+' |')
out+=['', '| 비교 기준 → 대상 | 같은 입력 warm 표본 | JSON 내용 일치 | 시간 합계 비율(기준/대상) |', '|---|---:|---:|---:|']
for a,b in [('hf_nocache','vllm_16k_current'),('hf_cache','vllm_16k_current'),('hf_cache_lora_bf16_r2','vllm_16k_current'),('vllm_16k_current','vllm_16k_bi_off'),('vllm_16k_current','vllm_12k')]:
 x=s['comparisons'].get(a+'__'+b)
 if x:out.append(f"| {names[a]} → {names[b]} | {x['n']} | {x['json_equal']}/{x['n']} | {x['sum_speed_ratio']:.3f} |")
out+=['', '현행 HF 캐시ON과 vLLM의 warm 출력 차이는 동일 Response 입력ID56의 “보통 양으로 선택했어요”와 “보통 양으로 골랐어요” 표현이다(각2회). 면종류·수량·크림추천은 같다. JSON 내용 일치는 의미 정답 검사가 아니다. 출력 길이·정밀도·커널·버전·캐시 정책 차이가 남아 있으므로 합계 비율은 해당 배포 조건의 관측값이다. 소수 표본 P95는 서비스 SLA 추정이 아니다.', '', '## 확인된 실패와 실제 답변', '']
for label,name in names.items():
 x=s['conditions'].get(label)
 if not x:continue
 if x['failure']:out.append(f"- {name}: {x['failure']['error']}. 완료된 replay와 부분 시나리오만 결과로 사용한다.")
 for r in x['scenario']['failed_turns']:
  out.append(f"- {name}, {r['episode']} turn {r['turn']}: 사용자 “{r['user']}”, 답변 “{r['reply']}”. 기대 plan `{r['expected_plans']}`, 실제 `{r['actual_plans']}`. 체크 상세는 SUMMARY.json.")
out+=['', 'HF 캐시OFF는 GPU 여유<1GiB가 4개 연속 표본에서 관측돼 시험 소유 scope를 중단했다. OOM으로 단정하지 않는다. 첫 vLLM 캐시OFF 시도는 6개 replay 이후 기록이 끊겼고 종료 원인은 UNKNOWN이다. 부분 기록을 보존했으며 새 `_r2` 디렉터리 결과만 완주 조건으로 쓴다.', '', '## 동일성·측정 범위', '',
'모델은 로컬 Gemma4 12B, adapter는 D epoch6/step1638, Q epoch6/step234, P best/step150이다. 세 adapter를 병합하지 않고 활성 adapter를 바꿔 실행했다. 원본656개 텐서×3와 vLLM 변환본의 원본 대응 값, 추가16개 tied K/V 복사 텐서×3를 CPU에서 대조했다. 프롬프트·tokenizer·입력 token ID·grammar·greedy·출력 한도를 고정했다. 실제 adapter 경로·SHA는 BEFORE.json에 있다.',
'', '기존 HF는 NF4/BF16 본체에 PEFT가 LoRA1968개 텐서를 FP32로 올린다. vLLM은 LoRA auto가 모델 BF16을 따른다. HF BF16 대조는 시험 프로세스 내 LoRA dtype만 바꿨으며 원본 체크포인트를 수정하지 않았다. HF torch2.12.1+cu130/transformers5.14.1, vLLM torch2.13.0/transformers5.19.0/vLLM0.32.0으로 라이브러리도 다르다. 따라서 완전히 같은 연산 구현을 비교한 실험은 아니다.',
'', 'NVML50ms 표본의 장치 전체 사용량이다. RustDesk 등 외부 점유도 포함하며 첫 HF 캐시OFF와 이후 조건의 외부 부하가 같지는 않다. CUDA allocator 단독 peak나 순간 최대치 전부를 포착했다는 뜻은 아니다. 각 조건의 실제 최대 표본 간격은 SUMMARY.json에 남겼다.',
'', '보존 검증은 source40개·모델JSON·입력·원본/변환adapter 파일 SHA와 환경별 배포 이름/버전 목록을 비교한다. base weight는 크기·mtime 비교이며 전체 SHA 검사가 아니다. 패키지 설치 파일 전부의 binary hash 검사도 아니다. 최종 검증·GPU 반환은 AFTER.json을 근거로 별도 판정한다.',
'', '## 재개 및 보존 예외', '', '첫 HF BF16 대조 준비에서 wetext의 ModelScope 다운로드/HTTP403 재시도가 관측돼 소유 scope를 정리했다. 보조 FST 캐시 일부는 실제 갱신됐다. 원래 HF 오프라인 변수는 ModelScope를 막지 못했다. 두 마지막 재시험은 시험 프로세스에서 wetext snapshot 조회만 기존 로컬 디렉터리로 고정했다. 원래 helper는 text_frontend=False로 합성하며 frontend.py:134~135에서 정규화를 건너뛴다. 운영 helper·모델 설정은 바꾸지 않았다. 첫 BF16의 부분 replay3회와 첫 vLLM OFF의6회는 최종 조건 통계에서 제외한다. 외부 보조 캐시 불변을 주장하지 않는다.', '', '## 최종 판정', '', '**전체 조건 종료·교차검수 후 기입한다.**']
(ROOT/'docs/2026-11-05_hf_vllm_compare_generated.md').write_text('\n'.join(out)+'\n')
outputs=['# 조건별 실제 사용자 발화·응답', '', '상태/계약 체크의 상세 및 모델 입력·raw 출력은 각 TURNS.jsonl에 있다. 아래 답변은 graph 최종 출력이며 실행 턴의 Python 고정 안내를 포함한다. 모든 행을 모델의 자유 생성 문장으로 해석하지 않는다.']
for label,name in names.items():
 p=BASE/label/'TURNS.jsonl'
 if not p.exists():continue
 outputs+=['','## '+name,'','| 대화/턴 | 사용자 | 최종 답변 | 계약 |','|---|---|---|---|']
 for l in p.read_text().splitlines():
  r=json.loads(l);outputs.append(f"| {r['episode']}/{r['turn']} | {esc(r['user_text'])} | {esc(r['reply'])} | {'PASS' if r['contract_passed'] else 'FAIL'} |")
(BASE/'OUTPUTS.md').write_text('\n'.join(outputs)+'\n')
print(ROOT/'docs/2026-11-05_hf_vllm_compare_generated.md')
