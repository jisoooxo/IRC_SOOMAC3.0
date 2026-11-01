"""check_after_training.sh가 만든 17턴 결과 3개를 비교해 보고서를 출력한다.

1. C 동일성: hf vs hf-no-budget. 턴마다 모델 입력(messages)·원문 출력·응답·주문 상태가 완전히 같아야 한다.
   하나라도 다르면 종료 코드 1(입력 길이 처리가 평소 입력을 바꿨다는 뜻).
2. vLLM vs HF: 턴별 필드 일치 수(정보용, 실패로 보지 않는다. 계산 순서 차이로 일부 갈릴 수 있다).
3. 실행별 하네스 통과 턴 수, 17턴 총 시간, GPU 최대 사용량.
4. RAM·OOM 안전선: 단계별 시스템 RAM 최대·최소 여유, 단계 프로세스별 최대 vs 상한.
"""
import argparse
import json
from pathlib import Path

VOLATILE = {"latency_ms", "prefix_cache"}  # 실행마다 달라도 되는 값
CORE = ("order", "pending", "execute")  # 로봇 동작을 정하는 필드
GPU_CAP_MIB = 27 * 1024
# 이 PC는 swap이 없어 여유 RAM이 0이 되면 바로 OOM이다. 로봇 스택(ROS·카메라·UI)이 더 올라갈 몫을 남겨 둔다.
RAM_SAFE_FREE_GIB = 8
RAM_WARN_FREE_GIB = 4
STAGE_NAMES = {"convert": "어댑터 변환", "hf": "HF 17턴(C 적용)", "hf_no_budget": "HF 17턴(C 이전)",
               "stack_hold": "STT·TTS 올리기", "vllm_start": "vLLM 서버 시작", "vllm": "vLLM 17턴(+STT·TTS)"}


def load_run(path: Path) -> dict:
    exit_info = json.loads((path / "EXIT.json").read_text(encoding="utf-8"))
    if exit_info.get("exit_code") != 0:
        raise SystemExit(f"{path.name} 실행 실패: {exit_info.get('error')}")
    rows = {}
    for line in (path / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["episode"], row["turn"])
        if key in rows:
            raise SystemExit(f"{path.name} 결과에 같은 턴이 두 번 있음: {key}")
        rows[key] = row
    # 하네스가 계획한 턴을 다 돌았는지 확인한다(중간에 멈춘 결과를 완주로 오인하지 않게).
    result = json.loads((path / "RESULT.json").read_text(encoding="utf-8"))
    planned = sum(episode["planned_turns"] for episode in result["episodes"])
    if len(rows) != planned:
        raise SystemExit(f"{path.name} 결과 턴 수 {len(rows)} != 계획 {planned}")
    return rows


def stable(value):
    # 시간·캐시 상태만 빼고 행 전체(입력 messages·원문 출력·주문·정책·검사 결과)를 그대로 비교한다.
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def adapter_hashes(path: Path) -> dict:
    identity = json.loads((path / "IDENTITY.json").read_text(encoding="utf-8"))
    return {name: info["weight_sha256"] for name, info in identity["adapters"].items()}


def fields(row: dict) -> dict:
    return {
        "order": row["after"]["order"],
        "pending": row["after"].get("pending"),
        "execute": row["policy"].get("execute"),
        "decision_raw": [e.get("raw") for e in row["traces"].get("decision", [])],
        "reply": row["reply"],
    }


def trimmed_calls(rows: dict) -> int:
    return sum(1 for row in rows.values() for events in row["traces"].values() for e in events
               if e.get("history_trimmed_messages"))


def summary_line(name: str, rows: dict) -> str:
    passed = sum(1 for row in rows.values() if row.get("passed"))
    total_s = sum(row["latency_ms"] for row in rows.values()) / 1000
    return f"{name:13s} 하네스 통과(엄격 기준) {passed}/{len(rows)}턴, 17턴 총 {total_s:.1f}초, 대화 자르기 {trimmed_calls(rows)}회"


def gpu_peak(path: Path) -> str:
    if not path.is_file():
        return "기록 없음"
    values = [int(v) for v in path.read_text().split() if v.strip().isdigit()]
    if not values:
        return "기록 없음"
    peak = max(values)
    return f"{peak} MiB ({peak / 1024:.1f} GiB){' ⚠ 27GiB 상한 초과' if peak > GPU_CAP_MIB else ''}"


def gib(text: str) -> float:
    return float(text.rstrip("Gg"))


def ram_report(path: Path, ram_cap_gib: float, hold_cap_gib: float, total_cap_gib: float) -> None:
    if not path.is_file():
        print("RAM 기록 없음")
        return
    total = None
    stages: dict = {}
    scopes: dict = {}  # 이름 -> [전체 최대, anon 최대]
    slice_peak = None
    hold_seconds = [0, 0]  # vLLM 17턴 동안 [기록된 초, STT·TTS가 둘 다 살아 있던 초]
    for line in path.read_text().splitlines():
        if line.startswith("# MemTotal_MiB"):
            total = int(line.split()[-1])
            continue
        parts = line.split()
        if len(parts) < 4 or not parts[2].isdigit():
            continue
        stage, used, avail = parts[1], int(parts[2]), int(parts[3])
        peak = stages.setdefault(stage, {"used": 0, "avail": 10**9})
        peak["used"], peak["avail"] = max(peak["used"], used), min(peak["avail"], avail)
        if stage == "vllm":
            names = {item.partition("=")[0] for item in parts[4:]}
            hold_seconds[0] += 1
            hold_seconds[1] += {"stt", "tts"} <= names
        for item in parts[4:]:
            name, _, value = item.partition("=")
            if name == "total":
                slice_peak = max(slice_peak or 0, int(value))
                continue
            current, _, anon = value.partition("/")
            best = scopes.setdefault(name, [0, 0])
            best[0], best[1] = max(best[0], int(current)), max(best[1], int(anon or 0))
    measured = {stage: peak for stage, peak in stages.items() if stage in STAGE_NAMES}
    if not measured:
        print("RAM 기록 없음")
        return
    print(f"전체 RAM {total / 1024:.1f} GiB, swap 없음(여유 0 = 즉시 OOM). 1초 간격이라 1초보다 짧은 순간 최대는 빠질 수 있다")
    for stage, peak in measured.items():
        print(f"  {STAGE_NAMES[stage]:22s} 시스템 사용 최대 {peak['used'] / 1024:5.1f} GiB, 최소 여유 {peak['avail'] / 1024:5.1f} GiB")
    if slice_peak is None:
        print("  ⚠ 검사 묶음 합계 기록 없음(cgroup 경로 확인)")
    else:
        print(f"  검사 묶음 합계 최대 {slice_peak / 1024:.1f} / {total_cap_gib:.0f} GiB (파일 캐시 포함)")
    if "stt" in scopes or "tts" in scopes:
        seconds, alive = hold_seconds
        flag = "" if seconds and alive == seconds else " ⚠ 도중에 STT·TTS가 꺼졌음(이 구간은 로봇 스택 조건이 아님)"
        print(f"  vLLM 17턴 동안 STT·TTS 상주 확인 {alive}/{seconds}초{flag}")
    print("  단계별 최대(전체 = 파일 캐시 포함, anon = 실제로 잡은 메모리):")
    for name, (current, anon) in scopes.items():
        cap = hold_cap_gib if name in {"stt", "tts"} else ram_cap_gib
        ratio = current / 1024 / cap
        flag = " ⚠ 상한 90% 이상" if ratio >= 0.9 else ""
        print(f"    {name:11s} 전체 {current / 1024:5.1f} / {cap:.0f} GiB ({ratio:.0%}), anon {anon / 1024:5.1f} GiB{flag}")
    worst = min(peak["avail"] for peak in measured.values()) / 1024
    if worst >= RAM_SAFE_FREE_GIB:
        verdict = f"통과 — 최소 여유 {worst:.1f} GiB ≥ {RAM_SAFE_FREE_GIB} GiB"
    elif worst >= RAM_WARN_FREE_GIB:
        verdict = f"⚠ 주의 — 최소 여유 {worst:.1f} GiB. ROS·카메라·UI까지 켜면 부족할 수 있음"
    else:
        verdict = f"⚠ 위험 — 최소 여유 {worst:.1f} GiB < {RAM_WARN_FREE_GIB} GiB. 로봇 스택을 더 올릴 여유가 거의 없음"
    print(f"OOM 안전선 판정: {verdict}")
    print("  (측정 조건: STT·TTS는 상주만 하고 동시 추론은 하지 않음. ROS·카메라·UI 노드는 포함 안 됨. 운영 후보 판단용이며 안전 인증은 아님)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, help="check_after_training.sh 결과 폴더")
    parser.add_argument("--ram-cap", default="40G", help="단계별 RAM 상한(check_after_training.sh의 CHECK_RAM_CAP)")
    parser.add_argument("--hold-cap", default="16G", help="STT·TTS 대역 RAM 상한(CHECK_HOLD_RAM_CAP)")
    parser.add_argument("--total-cap", default="44G", help="검사 묶음 전체 RAM 상한(CHECK_TOTAL_RAM_CAP)")
    args = parser.parse_args()

    hf = load_run(args.out / "hf")
    hf_old = load_run(args.out / "hf_no_budget")
    vllm = load_run(args.out / "vllm") if (args.out / "vllm").is_dir() else None
    keys = sorted(hf, key=lambda k: (k[0] != "compound", k[1]))

    print("== 1. C 동일성 (HF: C 적용 vs C 이전) ==")
    if adapter_hashes(args.out / "hf") != adapter_hashes(args.out / "hf_no_budget"):
        raise SystemExit("두 HF 실행의 어댑터가 다름(IDENTITY.json). 비교 무효")
    if set(hf) != set(hf_old):
        raise SystemExit(f"두 HF 실행의 턴 목록이 다름: {sorted(set(hf) ^ set(hf_old))}")
    differ = [k for k in keys if stable(hf[k]) != stable(hf_old[k])]
    print(f"입력·출력 완전 일치 {len(keys) - len(differ)}/{len(keys)}턴" + (" → 통과" if not differ else " → ⚠ 실패"))
    for key in differ:
        print(f"  다름: {key}")
    if differ:
        print("  (이전 측정에서 HF는 같은 입력이면 17턴이 매번 같았다. 다르면 turns.jsonl 원문을 비교해 원인을 확인)")

    if vllm is not None:
        print("\n== 2. vLLM vs HF (정보용) ==")
        same = core_same = 0
        for key in keys:
            a_core = {k: fields(hf[key])[k] for k in CORE}
            if key in vllm and {k: fields(vllm[key])[k] for k in CORE} == a_core:
                core_same += 1
            if key in vllm and fields(vllm[key]) == fields(hf[key]):
                same += 1
                continue
            a, b = fields(hf[key]), fields(vllm.get(key) or {"after": {"order": None}, "policy": {}, "traces": {}, "reply": None})
            changed = [name for name in a if a[name] != b[name]]
            print(f"  {key} [{hf[key]['tag']}] 다른 항목: {changed}")
            if "reply" in changed:
                print(f"    HF  : {a['reply']}\n    vLLM: {b['reply']}")
        print(f"주문·pending·실행 일치 {core_same}/{len(keys)}턴, Decision 원문·응답 문장까지 일치 {same}/{len(keys)}턴")

    print("\n== 3. 실행 요약 ==")
    print(summary_line("hf", hf))
    print(summary_line("hf_no_budget", hf_old))
    if vllm is not None:
        print(summary_line("vllm", vllm))
    print(f"GPU 최대(HF 실행 중): {gpu_peak(args.out / 'gpu_hf.log')}")
    with_hold = (args.out / "stt.log").is_file() and (args.out / "tts.log").is_file()
    print(f"GPU 최대(vLLM{' + STT·TTS 상주' if with_hold else ' 단독'}): {gpu_peak(args.out / 'gpu_vllm.log')}"
          " (1초 간격, 짧은 순간 최대는 빠질 수 있음)")

    print("\n== 4. RAM·OOM 안전선 ==")
    if not with_hold:
        STAGE_NAMES["vllm"] = "vLLM 17턴(단독)"
    ram_report(args.out / "ram.log", gib(args.ram_cap), gib(args.hold_cap), gib(args.total_cap))
    raise SystemExit(1 if differ else 0)


if __name__ == "__main__":
    main()
