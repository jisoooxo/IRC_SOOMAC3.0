#!/usr/bin/env python3
"""Build a self-contained, read-only view of the recorded conversation.

Only the archived measurement and optional supplement are read. No model, ROS,
network, or training code is imported or run.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MEASUREMENTS = HERE / "2026-11-01_scenario_measurements.json"
SUPPLEMENT = HERE / "2026-11-01_scenario_supplement_view.json"
OUTPUT = HERE / "2026-11-01_conversation_view.html"


def inline_json(value: object) -> str:
    """Keep JSON inside a script element without allowing an end tag."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


HTML = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>IRC 대화 시나리오 · 실제 17턴 관측</title>
  <style>
    :root{font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;color:#172628;background:#f3f7f6;font-size:15px}
    *{box-sizing:border-box} body{margin:0} button{font:inherit;cursor:pointer} a{color:#146a68}
    .shell{max-width:1550px;margin:auto;padding:22px clamp(14px,2vw,30px) 70px}
    .top{background:#123c3c;color:#fff;border-radius:18px;padding:26px 30px;box-shadow:0 10px 26px #153b3b1c}
    .eyebrow{text-transform:uppercase;letter-spacing:.11em;color:#add5cd;font-size:.74rem;font-weight:800}
    h1{font-size:clamp(1.55rem,3vw,2.45rem);line-height:1.2;margin:.4rem 0 .7rem}
    h2{font-size:1.15rem;margin:0 0 12px} h3{font-size:1rem;margin:0 0 10px}
    p{line-height:1.55;margin:.35rem 0} .top p{max-width:82ch;color:#d9ece7}
    .stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:16px 0}
    .stat,.notice,.panel,.sidebar,.detail,.all-card{background:white;border:1px solid #d8e4e0;border-radius:14px}
    .stat{padding:14px 16px}.stat strong{display:block;font-size:1.5rem;line-height:1.15}.stat span{color:#4e6262;font-size:.83rem}
    .notice{padding:14px 18px;border-left:5px solid #cd7854;margin-bottom:16px}
    .notice strong{color:#874028} .notice p{margin:.15rem 0}
    .toolbar{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0 12px}
    .segmented{display:flex;gap:6px;background:#e3eeeb;padding:5px;border-radius:11px;flex-wrap:wrap}
    .segmented button,.mini-button{border:0;background:transparent;color:#315353;border-radius:8px;padding:9px 12px;font-weight:700}
    .segmented button[aria-pressed="true"],.mini-button[aria-pressed="true"]{background:#fff;color:#0e5552;box-shadow:0 2px 7px #153b3b20}
    .layout{display:grid;grid-template-columns:minmax(250px,300px) minmax(0,1fr);gap:16px;align-items:start}
    .sidebar{position:sticky;top:12px;padding:16px;max-height:calc(100vh - 24px);overflow:auto}
    .sidebar h2{margin-bottom:5px}.sidebar .hint,.muted{color:#637474;font-size:.85rem}
    .turn-list{display:grid;gap:7px;margin:12px 0 0}
    .turn-button{width:100%;text-align:left;background:#f7faf9;border:1px solid #dce7e3;border-radius:10px;padding:10px 12px;color:#203838}
    .turn-button:hover,.turn-button:focus-visible{border-color:#27847c;outline:2px solid transparent}
    .turn-button[aria-current="true"]{background:#e0f0ea;border-color:#237770;box-shadow:inset 3px 0 #237770}
    .turn-button b{display:block;font-size:.9rem}.turn-button span{display:block;color:#5b6e6b;font-size:.8rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:3px}
    .detail{padding:clamp(16px,2vw,26px);min-width:0}
    .detail-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;border-bottom:1px solid #dbe7e3;padding-bottom:15px;margin-bottom:16px}
    .detail-head h2{font-size:1.35rem;margin:3px 0}.meta{color:#58706a;font-size:.86rem}
    .chips{display:flex;gap:6px;flex-wrap:wrap}.chip{display:inline-block;border-radius:20px;padding:4px 9px;background:#e9f3f0;color:#195c55;font-size:.77rem;font-weight:700}
    .chip.warn{background:#fff0e7;color:#985030}.chip.neutral{background:#edf0f1;color:#51666b}
    .dialogue{display:grid;gap:10px;margin-bottom:20px}.bubble{border-radius:13px;padding:13px 16px;max-width:92%}
    .bubble.user{background:#e5efed;justify-self:start}.bubble.assistant{background:#e9f0ff;justify-self:end}
    .bubble label{display:block;font-size:.74rem;font-weight:800;color:#52716f;margin-bottom:5px}.bubble.assistant label{color:#51688e}
    .bubble p{font-size:1.02rem;line-height:1.55;margin:0}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.panel{padding:16px;min-width:0}
    .panel.actual{border-top:4px solid #26827b}.panel.expected{border-top:4px solid #b69150;background:#fffdf8}
    .field{display:grid;grid-template-columns:minmax(115px,42%) 1fr;gap:9px;padding:7px 0;border-top:1px solid #eef2f0;line-height:1.35;overflow-wrap:anywhere}
    .field:first-of-type{border-top:0}.field .field-name{font-size:.83rem;color:#5c6d6c}.field .field-value{margin:0;font-weight:650}
    .diff{width:100%;border-collapse:collapse;font-size:.88rem}.diff th{text-align:left;color:#586e69;font-size:.77rem;background:#f0f5f3}.diff th,.diff td{padding:8px;border-bottom:1px solid #e8efed;vertical-align:top;overflow-wrap:anywhere}
    .diff tr.changed td{background:#ebf7f1}.diff tr.mismatch td:last-child{color:#9a4e2c}
    .call{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.call > div{border:1px solid #e0e9e6;border-radius:10px;padding:10px;background:#f8fbfa}
    .call b{display:block}.call small{display:block;color:#657772;margin-top:4px}.call .no{color:#985030}
    .subhead{font-size:.78rem;color:#68807a;font-weight:800;letter-spacing:.04em;text-transform:uppercase;margin:16px 0 7px}
    .raw-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:14px}
    details{border:1px solid #dce7e3;border-radius:10px;background:#f9fbfa;min-width:0} summary{cursor:pointer;font-weight:700;padding:10px 13px;color:#255d59}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:330px;overflow:auto;margin:0;padding:0 13px 12px;font-size:.76rem;line-height:1.48;color:#2a3c3c}
    .all-episodes{display:grid;gap:22px}.all-episodes h2{margin-bottom:4px}.all-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
    .all-card{padding:15px}.all-card header{display:flex;justify-content:space-between;gap:8px;align-items:baseline}.all-card h3{margin:0}
    .all-card .line{margin:10px 0;border-left:3px solid #bed7d2;padding-left:10px}.all-card .line.reply{border-color:#b6c7e7}
    .all-card small{font-weight:800;color:#5d7771}.all-card p{margin:3px 0}.all-card button{border:1px solid #bdd9d3;background:white;color:#1d6961;border-radius:8px;padding:6px 9px;font-weight:700}
    .bottom{margin-top:22px;display:grid;gap:12px}.bottom .panel{background:#fcfefd}.source-list{line-height:1.75;padding-left:20px;margin:.3rem 0}
    .supplement{border-left:5px solid #8b929a}.supplement .tag{display:inline-block;background:#e9eced;color:#4b6064;border-radius:6px;padding:3px 7px;font-size:.75rem;font-weight:800}
    .supp-list{display:grid;gap:9px;margin-top:12px}.supp-item{border:1px solid #dce5e4;border-radius:10px;padding:11px 13px;background:#f7faf9}.supp-item .label{font-size:.75rem;color:#657975;font-weight:800}.supp-item p{margin:5px 0}.supp-item .target{border-left:3px solid #9fb5b0;padding-left:9px}.supp-item .target pre{padding:5px 0;max-height:none}.supp-item details{margin-top:8px}
    [hidden]{display:none!important}
    @media(max-width:950px){.layout{grid-template-columns:1fr}.sidebar{position:static;max-height:none}.turn-list{grid-template-columns:repeat(3,minmax(0,1fr))}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:650px){.shell{padding:12px 10px 45px}.top{padding:20px}.turn-list{grid-template-columns:repeat(2,minmax(0,1fr))}.grid,.raw-grid,.all-grid{grid-template-columns:1fr}.call{grid-template-columns:1fr}.bubble{max-width:100%}.diff{font-size:.78rem}}
  </style>
</head>
<body>
<script id="measurement-data" type="application/json">__MEASUREMENTS__</script>
<script id="supplement-data" type="application/json">__SUPPLEMENT__</script>
<div class="shell">
  <header class="top">
    <div class="eyebrow">2026-11-01 · 로컬 관측 뷰</div>
    <h1>주문 대화 17턴: 실제 상태를 따라 보기</h1>
    <p>복합 발화 6턴과 분리 대조 11턴을 각각 한 세션의 누적 대화로 표시한다. 기대 주문은 비교용 오라클이며 실제 모델 상태에 주입되지 않았다.</p>
  </header>
  <section class="stats" aria-label="전체 실제 관측">
    <div class="stat"><strong>6 + 11</strong><span>실제 입력·응답 관측</span></div>
    <div class="stat"><strong>D 17 · Q 0 · P 17</strong><span>실제 모델 호출</span></div>
    <div class="stat"><strong>0 / 17</strong><span>execute=true</span></div>
    <div class="stat"><strong>noodle · 소스 없음</strong><span>17턴 모두 실제 section · 확정 소스 없음</span></div>
  </section>
  <section class="notice" aria-label="해석 경계">
    <strong>관측 완료는 시나리오 통과가 아니다.</strong>
    <p>크림을 추천한 문장은 사용자 요청에 맞지만, Q는 호출되지 않았고 후보·pending·확정 소스는 없었다. C2와 S2의 “크림으로 선택했어요”는 실제 주문과 다르다. 생성 계획과 실제 ROS 실행도 0건이다.</p>
  </section>
  <div class="toolbar">
    <div class="segmented" role="group" aria-label="표시 방식">
      <button id="mode-focus" type="button" aria-pressed="true">턴 선택</button>
      <button id="mode-all" type="button" aria-pressed="false">모든 대화 보기</button>
    </div>
    <span class="muted">원본 기록 그대로 · 미확인 실행/완료 없음 · file:// 사용 가능</span>
  </div>
  <main>
    <div id="focus-view" class="layout">
      <nav class="sidebar" aria-label="턴 선택">
        <h2>턴 선택</h2><div class="hint">복합과 분리 대조는 서로 다른 세션이다.</div>
        <div class="segmented" role="group" aria-label="에피소드" style="margin-top:12px">
          <button id="episode-compound" type="button" aria-pressed="true">복합 6턴</button>
          <button id="episode-split_control" type="button" aria-pressed="false">분리 11턴</button>
        </div>
        <div id="turn-list" class="turn-list"></div>
      </nav>
      <article id="turn-detail" class="detail" aria-live="polite"></article>
    </div>
    <div id="all-view" hidden></div>
  </main>
  <section class="bottom" aria-label="출처와 보강 자료">
    <div class="panel">
      <h2>출처와 판정 범위</h2>
      <ul class="source-list">
        <li><a href="2026-11-01_scenario_measurements.json">17턴 계측 JSON</a>: 화면의 발화, 답변, 전후 상태, D/P raw, 정책, 계획의 직접 출처.</li>
        <li><a href="2026-11-01_llm_scenario_regression.md">시나리오 회귀 기록</a> · <a href="2026-11-01_scenario_output_review.md">출력 독립 검수</a>: 의미 판정과 시험 경계.</li>
        <li>그래프 시간은 모델 로드·STT·TTS·물리 이동을 제외한다. 실행 0건으로 물리 완료 fixture도 적용되지 않았으며, 실제 ROS/VLM 검증은 없다.</li>
      </ul>
      <details><summary>계측 메타데이터 원본 JSON</summary><pre id="metadata-raw"></pre></details>
    </div>
    <div class="panel supplement">
      <h2>추가 보강 자료 <span class="tag">비활성 · 미학습</span></h2>
      <p>이 자료는 위 17턴의 실제 관측·판정에 섞지 않는다. 정답을 강제 주입한 오라클 리플레이이며 실제 모델 추론이나 ROS 성공이 아니다. <a href="2026-11-01_scenario_supplement_provenance.md">보강 데이터 출처</a></p>
      <p id="supplement-status"></p>
      <div id="supplement-list" class="supp-list"></div>
      <details id="supplement-details" hidden><summary>보강 자료 원본 JSON</summary><pre id="supplement-raw"></pre></details>
    </div>
  </section>
</div>
<script>
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('measurement-data').textContent);
  const supplement = JSON.parse(document.getElementById('supplement-data').textContent);
  const turns = data.turns;
  const state = {episode:'compound', turn:1, mode:'focus'};
  const title = {compound:'복합 발화', split_control:'분리 대조'};
  const section = {noodle:'면', veggie:'채소', meat:'육류', extra:'추가 토핑'};
  const orderLabels = {sauce:'소스', noodle_type:'면 종류', noodle_portion:'면 양', toppings:'재료', restrictions:'제한'};
  const valueLabels = {low:'적게 (low)',normal:'보통 (normal)',high:'많이 (high)',none:'제외 (none)'};
  const $ = id => document.getElementById(id);
  const fmt = value => value === null ? '없음' : value === undefined ? '기록 없음' : typeof value === 'boolean' ? (value ? '예' : '아니요') : valueLabels[value] || String(value);
  const json = value => JSON.stringify(value, null, 2);
  const escape = value => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const shown = value => escape(fmt(value));
  const pretty = value => escape(json(value));
  const byEpisode = ep => turns.filter(t => t.episode === ep);
  const current = () => turns.find(t => t.episode === state.episode && t.turn === state.turn);
  const context = t => t.response_context[0] || {};
  const raw = (t,stage) => t.raw_calls.find(c => c.stage === stage);
  const rawObject = call => { if (!call) return null; try { return JSON.parse(call.raw); } catch { return call.raw; } };
  const field = (name,value) => `<div class="field"><span class="field-name">${escape(name)}</span><span class="field-value">${shown(value)}</span></div>`;
  const fieldHtml = rows => rows.map(([name,value]) => field(name,value)).join('');
  const flattenOrder = order => {
    const result = {'소스':order.sauce,'면 종류':order.noodle_type,'면 양':order.noodle_portion};
    for (const [key,value] of Object.entries(order.toppings || {})) result[`재료 · ${key}`] = value;
    result['제한'] = order.restrictions || [];
    return result;
  };
  const valueText = value => Array.isArray(value) ? (value.length ? value.map(x => typeof x === 'object' ? json(x) : String(x)).join(', ') : '없음') : fmt(value);
  const diffHtml = t => {
    const before = flattenOrder(t.before_order), after = flattenOrder(t.after_order), expected = flattenOrder(t.expected_order);
    const keys = [...new Set([...Object.keys(before),...Object.keys(after),...Object.keys(expected)])];
    return `<table class="diff"><thead><tr><th>주문 필드</th><th>이전</th><th>실제 이후</th><th>기대</th></tr></thead><tbody>${keys.map(key => {
      const a=valueText(before[key]), b=valueText(after[key]), e=valueText(expected[key]);
      return `<tr class="${a!==b?'changed':''} ${b!==e?'mismatch':''}"><td>${escape(key)}</td><td>${escape(a)}</td><td>${escape(b)}</td><td>${escape(e)}</td></tr>`;
    }).join('')}</tbody></table>`;
  };
  const suppDiffHtml = item => {
    const before=flattenOrder(item.before_state.order), after=flattenOrder(item.after_state.order);
    const keys=[...new Set([...Object.keys(before),...Object.keys(after)])];
    return `<table class="diff"><thead><tr><th>오라클 주문 필드</th><th>이전</th><th>리플레이 이후</th></tr></thead><tbody>${keys.map(key=>{
      const a=valueText(before[key]),b=valueText(after[key]);
      return `<tr class="${a!==b?'changed':''}"><td>${escape(key)}</td><td>${escape(a)}</td><td>${escape(b)}</td></tr>`;
    }).join('')}</tbody></table>`;
  };
  const plans = list => list.length ? list.map(x => `${x.class} × ${x.repeat_count}`).join(', ') : '없음 (0건)';
  const pendingBrief = pending => {
    if (!pending) return '없음';
    const candidate=pending.candidate && pending.candidate.sauce ? ` · 후보 소스 ${pending.candidate.sauce}` : '';
    const targets=Array.isArray(pending.targets) && pending.targets.length ? ` · 대상 ${pending.targets.join(', ')}` : '';
    return `${pending.type || '종류 기록 없음'}${candidate}${targets}`;
  };
  const callHtml = (t,stage,label) => {
    const c=raw(t,stage);
    return `<div><b class="${c?'':'no'}">${label}: ${c?'호출됨':'호출 없음'}</b><small>${c ? '원본 호출·토큰·시간은 아래 JSON에 있음' : '이 턴 raw_calls에 호출 기록 없음'}</small></div>`;
  };
  const details = (label,value) => `<details><summary>${escape(label)}</summary><pre>${pretty(value)}</pre></details>`;
  const notes = t => {
    if (t.tag === 'initial') return '크림 제안 문장은 실제 P 답변이다. 구조화된 Q 후보·추천 pending은 없으며, 크림은 확정 주문이 아니다.';
    if (t.tag === 'cream_select' || (t.episode === 'compound' && t.tag === 'cream_execute')) return 'P가 크림 선택을 말했으나 실제 after_order.sauce는 없음이다. 응답과 저장 상태가 다르다.';
    if (t.decision_final.commit && !t.policy.execute) return 'D는 commit=true를 냈지만 정책이 실행을 허용하지 않았다. 계획과 물리 완료는 없다.';
    if (t.tag === 'allergy_sausage_execute' || t.tag === 'cheese_execute') return '즉시 실행 요청이 있었으나 D raw에 commit이 없었다. 실제 실행도 없다.';
    return '실제 답변과 주문 상태를 함께 확인한다. “담을까요?”는 구조화된 실행 pending을 뜻하지 않는다.';
  };
  const renderList = () => {
    $('turn-list').innerHTML = byEpisode(state.episode).map(t => `<button class="turn-button" type="button" data-turn="${t.turn}" aria-current="${t.turn===state.turn}"><b>${state.episode==='compound'?'C':'S'}${t.turn} · ${escape(t.tag)}</b><span>${escape(t.user_text)}</span></button>`).join('');
    document.querySelectorAll('.turn-button').forEach(btn => btn.addEventListener('click', () => {state.turn=Number(btn.dataset.turn); render();}));
    ['compound','split_control'].forEach(ep => $(`episode-${ep}`).setAttribute('aria-pressed',String(state.episode===ep)));
  };
  const renderDetail = () => {
    const t=current(), ctx=context(t), d=rawObject(raw(t,'decision'));
    const actualSection=section[t.actual_section] || t.actual_section;
    const expectedSection=section[t.expected_section] || t.expected_section;
    $('turn-detail').innerHTML = `
      <div class="detail-head"><div><div class="eyebrow" style="color:#477c73">${escape(title[t.episode])} · ${t.episode==='compound'?'C':'S'}${t.turn}</div><h2>${escape(t.tag)}</h2><div class="meta">예상 단계 ${escape(expectedSection)} / 실제 ${escape(actualSection)}</div></div>
      <div class="chips"><span class="chip ${t.expected_section!==t.actual_section?'warn':''}">실제 ${escape(actualSection)} 단계</span><span class="chip neutral">execute ${t.policy.execute?'true':'false'}</span></div></div>
      <div class="dialogue"><div class="bubble user"><label>사용자 실제 발화</label><p>${escape(t.user_text)}</p></div><div class="bubble assistant"><label>P 실제 답변</label><p>${escape(t.reply)}</p></div></div>
      <div class="grid">
        <section class="panel actual"><h3>실제 관측</h3><div class="call">${callHtml(t,'decision','D')}${callHtml(t,'recommendation','Q')}${callHtml(t,'response','P')}</div>
          <div class="subhead">D raw 출력 → 정규화된 최종 상태</div>
          ${fieldHtml([['D raw commit',d && typeof d==='object' && Object.hasOwn(d,'commit') ? d.commit : '출력 안 함'],['D final commit',t.decision_final.commit],['D final 추천 action',t.decision_final.recommendation.action==='none'?'추천 요청 없음 (none)':t.decision_final.recommendation.action],['D final 소스 patch',t.decision_final.order_patch.sauce],['실제 소스',t.after_order.sauce]])}
          <div class="subhead">실행 연결</div>
          ${fieldHtml([['pending 이전',t.before_pending===null?'없음':json(t.before_pending)],['pending 이후',t.after_pending===null?'없음':json(t.after_pending)],['next_prompt',ctx.next_prompt===null?'없음':ctx.next_prompt===undefined?'기록 없음':json(ctx.next_prompt)],['policy',`${t.policy.status} · ${t.policy.reason}`],['execute',t.policy.execute],['P 입력 execution_authorized',ctx.execution_authorized],['실제 생성 plans',plans(t.actual_plans)]])}
        </section>
        <section class="panel expected"><h3>예상 기준 · 실제 상태와 분리</h3><p class="muted">이쪽은 사용자 요청에서 만든 오라클이다. 모델 입력이나 주문 상태를 고친 값이 아니다.</p>
          ${fieldHtml([['예상 단계',expectedSection],['예상 주문 소스',t.expected_order.sauce],['예상 plans',plans(t.expected_plans)],['예상/실제 계획 일치',t.checks.plans_match],['예상/실제 주문 일치',t.checks.order_matches],...(Object.hasOwn(t.checks,'cream_is_unconfirmed_candidate')?[['크림 후보 연결',t.checks.cream_is_unconfirmed_candidate]]:[])])}
          <p class="muted">주문 일치는 확정 주문 필드 비교만 뜻한다. 추천 후보 저장이나 Q 호출 성공을 뜻하지 않는다.</p>
          <div class="subhead">검수 문서의 해석</div><p>${escape(notes(t))}</p>
        </section>
      </div>
      <div class="subhead">주문 필드 전후 · 기대값 비교 <span class="muted">(초록: 실제 변경 / 갈색: 기대와 불일치)</span></div>
      <div class="panel" style="overflow-x:auto">${diffHtml(t)}</div>
      <div class="raw-grid">
        ${details('D raw JSON',rawObject(raw(t,'decision')))}
        ${details('D final JSON',t.decision_final)}
        ${details('P raw JSON',rawObject(raw(t,'response')))}
        ${details('P 입력 핵심 사실 JSON',ctx)}
        ${details('policy / checks JSON',{policy:t.policy,checks:t.checks})}
        ${details('이 턴 전체 계측 JSON',t)}
      </div>`;
  };
  const renderAll = () => {
    $('all-view').innerHTML = `<div class="all-episodes">${['compound','split_control'].map(ep => `<section><h2>${title[ep]} · ${byEpisode(ep).length}턴</h2><p class="muted">한 에피소드 안에서 실제 after state가 다음 턴으로 이어졌다. 답변은 P raw와 발행 reply를 기준으로 한다.</p><div class="all-grid">${byEpisode(ep).map(t => `<article class="all-card"><header><h3>${ep==='compound'?'C':'S'}${t.turn} · ${escape(t.tag)}</h3><button type="button" data-episode="${ep}" data-turn="${t.turn}">상세 보기</button></header><div class="line"><small>사용자</small><p>${escape(t.user_text)}</p></div><div class="line reply"><small>P 실제 답변</small><p>${escape(t.reply)}</p></div><div class="chips"><span class="chip neutral">D ${raw(t,'decision')?'1':'0'} · Q ${raw(t,'recommendation')?'1':'0'} · P ${raw(t,'response')?'1':'0'}</span><span class="chip ${t.decision_final.commit?'':'neutral'}">commit ${t.decision_final.commit}</span><span class="chip neutral">execute ${t.policy.execute}</span><span class="chip ${t.expected_section!==t.actual_section?'warn':'neutral'}">실제 ${escape(section[t.actual_section])}</span></div></article>`).join('')}</div></section>`).join('')}</div>`;
    $('all-view').querySelectorAll('button[data-turn]').forEach(btn => btn.addEventListener('click', () => {state.episode=btn.dataset.episode;state.turn=Number(btn.dataset.turn);state.mode='focus';render();window.scrollTo({top:0,behavior:'smooth'});}));
  };
  const render = () => {
    const focus=state.mode==='focus'; $('focus-view').hidden=!focus; $('all-view').hidden=focus;
    $('mode-focus').setAttribute('aria-pressed',String(focus)); $('mode-all').setAttribute('aria-pressed',String(!focus));
    if (focus) {renderList();renderDetail();} else renderAll();
  };
  $('mode-focus').addEventListener('click',()=>{state.mode='focus';render();});
  $('mode-all').addEventListener('click',()=>{state.mode='all';render();});
  ['compound','split_control'].forEach(ep => $(`episode-${ep}`).addEventListener('click',()=>{state.episode=ep;state.turn=1;render();}));
  $('metadata-raw').textContent=json(Object.fromEntries(Object.entries(data).filter(([key])=>key!=='turns')));
  if (supplement === null) $('supplement-status').textContent='대기 중: 빌드 시 보강 JSON 파일이 없었다. 보강 자료는 이 HTML에 포함되지 않았다.';
  else {
    const summary=supplement.status_summary || {}, examples=supplement.examples || [];
    $('supplement-status').textContent=`${examples.length}개 선택 턴 · ${summary.selected_agent_targets ?? '기록 없음'}개 타깃. 학습 실행 ${summary.training_run===false?'없음':'기록 확인 필요'}, 로더 연결 ${summary.loader_connected===false?'없음':'기록 확인 필요'}. 아래는 모델 추론 결과가 아닌 오라클 강제 리플레이 타깃이다.`;
    $('supplement-list').innerHTML=examples.map((item,index)=>{
      const after=item.after_state || {}, choice=item.recommendation && item.recommendation.proposal ? item.recommendation.proposal : null;
      const dAction=item.decision.recommendation && item.decision.recommendation.action;
      const dCommit=Object.hasOwn(item.decision,'commit') ? fmt(item.decision.commit) : '출력 안 함';
      return `<article class="supp-item"><div class="label">오라클 리플레이 ${index+1} · ${escape(item.turn_id || 'ID 없음')}</div><p><b>사용자 발화</b> ${escape(item.user_text || '')}</p><div class="target"><small>D 정답 target</small><pre>${pretty(item.decision)}</pre></div><div class="target"><small>Q 정답 target</small><p>${item.recommendation ? `후보 소스 ${choice ? shown(choice.sauce) : '없음'} · ${escape(json(item.recommendation))}` : '해당 턴 Q 미호출'}</p></div><div class="target"><small>P 정답 target</small><p>${escape(item.response && item.response.reply || '해당 턴 P 미호출')}</p></div><div class="chips"><span class="chip neutral">D 추천 action ${dAction ? escape(dAction) : '요청 안 함'}</span><span class="chip neutral">D raw commit ${escape(dCommit)}</span><span class="chip neutral">해석 commit ${shown(item.commit)}</span><span class="chip neutral">policy execute ${shown(item.policy.execute)}</span></div><div class="field"><span class="field-name">pending 이전</span><span class="field-value">${escape(pendingBrief(item.before_state.pending))}</span></div><div class="field"><span class="field-name">pending 이후</span><span class="field-value">${escape(pendingBrief(after.pending))}</span></div><div style="overflow-x:auto;margin-top:10px">${suppDiffHtml(item)}</div>${details('pending 전후 · 오라클 원본',{before:item.before_state.pending,after:item.after_state.pending})}${details('오라클 리플레이 전체 JSON',item)}</article>`;
    }).join('');
    $('supplement-details').hidden=false; $('supplement-raw').textContent=json(supplement);
  }
  render();
})();
</script>
</body>
</html>
'''


def main() -> None:
    measurements = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    supplement = json.loads(SUPPLEMENT.read_text(encoding="utf-8")) if SUPPLEMENT.exists() else None
    if len(measurements["turns"]) != 17:
        raise ValueError("Expected exactly 17 recorded turns")
    page = HTML.replace("__MEASUREMENTS__", inline_json(measurements)).replace(
        "__SUPPLEMENT__", inline_json(supplement)
    )
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Built {OUTPUT} ({len(measurements['turns'])} turns; supplement={'included' if supplement is not None else 'pending'})")


if __name__ == "__main__":
    main()
