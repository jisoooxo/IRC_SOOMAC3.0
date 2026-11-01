#!/usr/bin/env bash
# 어댑터 학습 뒤(또는 GPU가 빈 때) 한 번에 확인한다. GPU를 쓰므로 다른 GPU 작업이 없을 때만 실행한다.
#
#   bash tools/vllm_server/check_after_training.sh --preflight   # 실행 조건만 확인(GPU 안 씀)
#   bash tools/vllm_server/check_after_training.sh               # 전체 실행(약 15~20분)
#
# 순서: 실행 조건 확인 → vLLM용 어댑터 변환 → HF 17턴(C 적용) → HF 17턴(C 이전)
#       → STT·TTS 대역 상주 + vLLM 서버 + 17턴(로봇 스택과 비슷한 조건) → 비교 보고서(RAM·GPU 안전선 포함)
#
# 메모리 보호(2026-11-01 OOM 사고 대책). 이 PC는 swap이 없어 RAM 여유가 0이 되면 바로 OOM이다.
#   - 모든 단계를 묶음(soomac_check.slice) 하나에 넣고 묶음 전체에 상한을 건다(동시에 떠 있는 서버·STT·TTS·클라이언트 합계).
#   - 단계마다 상한도 따로 건다. 넘치면 그 단계만 종료(137)되고 데스크톱은 보호된다.
#   - 1초마다 ram.log에 시스템 사용·여유, 묶음 합계, 단계별 사용(전체/anon)을 기록한다.
# 어댑터 경로는 src/soomac_irc/soomac_irc/model_runtime.py가 가리키는 것을 쓴다.
set -euo pipefail

ROOT=/home/roma/IRC_SOOMAC3.0
TOOLS=$ROOT/tools/vllm_server
LLM_PY=/home/roma/miniconda3/envs/soomac_llm/bin/python
ENVS=/home/roma/miniconda3/envs
OUT=${CHECK_OUT:-$ROOT/training/learning_first_v43/final/goals/GL10/accel_prefix_cache_20261101/after_training_$(date +%Y%m%d_%H%M%S)}
TOTAL_RAM_CAP=${CHECK_TOTAL_RAM_CAP:-44G}  # 묶음 전체 상한(전체 62.6GiB 중 데스크톱 몫을 남긴다)
RAM_CAP=${CHECK_RAM_CAP:-40G}              # 단계별 상한
HOLD_RAM_CAP=${CHECK_HOLD_RAM_CAP:-16G}    # STT·TTS 대역 상한
MIN_FREE_RAM_GB=${CHECK_MIN_FREE_RAM_GB:-48}  # 시작 조건: 여유 RAM이 묶음 상한 + 4GB 이상
WITH_STT_TTS=${CHECK_WITH_STT_TTS:-1}      # 1이면 vLLM 단계에 STT·TTS 모델을 함께 올린다(실제 로봇 스택 조건)
SLICE=soomac_check.slice
SLICE_CG=/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/$SLICE

export MAX_JOBS=2                       # 커널 즉석 컴파일 동시 작업 수 제한
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH

log() { echo "[$(date +%H:%M:%S)] $*"; }
# 단계 실행: 묶음 안의 scope + 단계 상한. 이름은 ram.log·보고서에 그대로 나온다.
capped() {
    local name=$1 cap=$2
    systemd-run --user --scope --quiet --slice="$SLICE" --unit="soomac-check-$name-$$" \
        -p MemoryMax="$cap" -p MemorySwapMax=0 "${@:3}"
}
fail() {
    # $1 = 종료 코드, $2 = 단계 설명, $3 = 로그
    if [ "$1" = 137 ]; then
        log "$2 실패: 강제 종료(137). 메모리 상한에 걸렸을 가능성이 큼 → ram.log 마지막 줄·$3 확인"
    else
        log "$2 실패(종료 코드 $1) → $3"
    fi
    exit 1
}
stage() { echo "$1" > "$OUT/.stage"; log "$2"; }

preflight() {
    local ok=1 apps free_gb running
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
    if [ -n "$apps" ]; then
        echo "✗ GPU 사용 중(학습·다른 시험이 도는지 확인):"; echo "$apps" | sed 's/^/    /'; ok=0
    else
        echo "✓ GPU 비어 있음"
    fi
    if curl -s -o /dev/null --max-time 2 http://127.0.0.1:8000/health; then
        echo "✗ 포트 8000에 서버가 이미 떠 있음(vLLM 서버를 끄고 다시 실행)"; ok=0
    else
        echo "✓ 포트 8000 비어 있음"
    fi
    free_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    if [ "$free_gb" -lt "$MIN_FREE_RAM_GB" ]; then
        echo "✗ 사용 가능 RAM ${free_gb}GB < ${MIN_FREE_RAM_GB}GB(묶음 상한 $TOTAL_RAM_CAP + 여유)"; ok=0
    else
        echo "✓ 사용 가능 RAM ${free_gb}GB"
    fi
    if systemd-run --user --scope --quiet -p MemoryMax=64M true 2>/dev/null; then
        echo "✓ 메모리 상한 사용 가능"
    else
        echo "✗ systemd-run 메모리 상한을 쓸 수 없음"; ok=0
    fi
    if systemctl --user list-units --no-legend "soomac-check-*" 2>/dev/null | grep -q .; then
        echo "✗ 이전 검사의 단계가 아직 남아 있음: systemctl --user stop $SLICE"; ok=0
    fi
    running=$(docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null || true)
    if [ -n "$running" ]; then
        echo "  참고: 실행 중인 docker 컨테이너:"; echo "$running" | sed 's/^/    /'
    fi
    [ "$ok" = 1 ]
}

ram_monitor() {
    # 1초마다: 시각 단계 시스템사용MiB 여유MiB total=묶음MiB [단계=전체MiB/anonMiB ...]
    # 전체(memory.current)에는 파일 캐시가 들어 있다. anon은 프로세스가 실제로 잡은 메모리다.
    set +e
    awk '/MemTotal/ {printf "# MemTotal_MiB %d\n", $2/1024}' /proc/meminfo
    while true; do
        local line scope name current anon
        line="$(date +%s) $(cat "$OUT/.stage" 2>/dev/null || echo -) $(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "%d %d", (t-a)/1024, a/1024}' /proc/meminfo)"
        current=$(cat "$SLICE_CG/memory.current" 2>/dev/null) && line+=" total=$((current / 1048576))"
        for scope in "$SLICE_CG"/soomac-check-*-$$.scope; do
            name=${scope##*/soomac-check-}; name=${name%-$$.scope}
            current=$(cat "$scope/memory.current" 2>/dev/null) || continue
            anon=$(awk '/^anon /{print $2; exit}' "$scope/memory.stat" 2>/dev/null)
            line+=" $name=$((current / 1048576))/$(( ${anon:-0} / 1048576 ))"
        done
        echo "$line"
        sleep 1
    done
}

gpu_monitor_start() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$1" & GPU_MON=$!; }
gpu_monitor_stop() { kill "${GPU_MON:-}" 2>/dev/null || true; wait "${GPU_MON:-}" 2>/dev/null || true; }
ram_monitor_stop() { kill "${RAM_MON:-}" 2>/dev/null || true; wait "${RAM_MON:-}" 2>/dev/null || true; }
stop_all() { systemctl --user stop "$SLICE" 2>/dev/null || true; }  # 묶음 안의 서버·STT·TTS·남은 단계까지 모두 종료
trap 'gpu_monitor_stop; stop_all; ram_monitor_stop' EXIT

wait_until() {
    # $1 = 확인 명령, $2 = 기다릴 프로세스 PID, $3 = 최대 초, $4 = 설명, $5 = 로그
    local waited=0 rc
    until eval "$1"; do
        if ! kill -0 "$2" 2>/dev/null; then
            rc=0; wait "$2" || rc=$?
            fail "$rc" "$4(시작 중 종료)" "$5"
        fi
        [ "$waited" -lt "$3" ] || { log "$4: $3초 안에 준비되지 않음 → $5"; exit 1; }
        sleep 2; waited=$((waited + 2))
    done
}

start_hold() {
    # STT·TTS 대역을 띄우고 HOLD(모델 상주 + 1회 추론 끝)까지 기다린다.
    local name=$1 env=$2
    capped "$name" "$HOLD_RAM_CAP" env -u PYTHONPATH "$ENVS/$env/bin/python" -I "$TOOLS/stack_hold_$name.py" > "$OUT/$name.log" 2>&1 &
    wait_until "grep -q HOLD '$OUT/$name.log' 2>/dev/null" $! 300 "$name 대역" "$OUT/$name.log"
}

log "실행 조건 확인"
if ! preflight; then
    log "조건 미충족 → 중단(아무것도 바꾸지 않음)"
    trap - EXIT
    exit 2
fi
if [ "${1:-}" = "--preflight" ]; then
    log "조건 확인만 요청됨 → 종료"
    trap - EXIT
    exit 0
fi

mkdir -p "$OUT"
log "결과 폴더: $OUT"
systemctl --user set-property --runtime "$SLICE" MemoryMax="$TOTAL_RAM_CAP" MemorySwapMax=0
log "묶음 전체 RAM 상한 $TOTAL_RAM_CAP, 단계별 $RAM_CAP, STT·TTS 각 $HOLD_RAM_CAP"
echo start > "$OUT/.stage"
ram_monitor > "$OUT/ram.log" &
RAM_MON=$!
cd "$ROOT"

stage convert "1/4 vLLM용 어댑터 변환"
capped convert "$RAM_CAP" "$LLM_PY" "$TOOLS/convert_adapters_for_vllm.py" > "$OUT/convert.log" 2>&1 \
    || fail "$?" "변환" "$OUT/convert.log"
grep -v Warning "$OUT/convert.log" || true

gpu_monitor_start "$OUT/gpu_hf.log"
stage hf "2/4 HF 17턴 (C 적용)"
capped hf "$RAM_CAP" "$LLM_PY" "$TOOLS/run_scenario.py" hf --measure-all --out "$OUT/hf" > "$OUT/hf.log" 2>&1 \
    || fail "$?" "HF 17턴" "$OUT/hf.log"
stage hf_no_budget "3/4 HF 17턴 (C 이전 동작)"
capped hfold "$RAM_CAP" "$LLM_PY" "$TOOLS/run_scenario.py" hf-no-budget --measure-all --out "$OUT/hf_no_budget" > "$OUT/hf_no_budget.log" 2>&1 \
    || fail "$?" "HF 17턴(C 이전)" "$OUT/hf_no_budget.log"
gpu_monitor_stop

gpu_monitor_start "$OUT/gpu_vllm.log"
if [ "$WITH_STT_TTS" = 1 ]; then
    stage stack_hold "4/4 STT·TTS 대역 올리기(로봇 스택 조건)"
    start_hold stt gemma4_env
    start_hold tts cosyvoice3
fi
stage vllm_start "4/4 vLLM 서버 시작(첫 시작은 2~4분)"
capped vllm "$RAM_CAP" bash "$TOOLS/start_vllm_server.sh" > "$OUT/server.log" 2>&1 &
wait_until "curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8000/health" $! 900 "vLLM 서버" "$OUT/server.log"
stage vllm "vLLM 17턴"
capped vllmclient "$RAM_CAP" "$LLM_PY" "$TOOLS/run_scenario.py" vllm --measure-all --out "$OUT/vllm" > "$OUT/vllm.log" 2>&1 \
    || fail "$?" "vLLM 17턴" "$OUT/vllm.log"
stop_all
gpu_monitor_stop
echo done > "$OUT/.stage"
sleep 2
ram_monitor_stop

log "비교"
set +e
"$LLM_PY" "$TOOLS/compare_scenario_runs.py" "$OUT" --ram-cap "$RAM_CAP" --hold-cap "$HOLD_RAM_CAP" \
    --total-cap "$TOTAL_RAM_CAP" | tee "$OUT/REPORT.txt"
status=${PIPESTATUS[0]}
set -e
log "끝. 보고서: $OUT/REPORT.txt (종료 코드 $status: 0=C 동일성 통과)"
exit "$status"
