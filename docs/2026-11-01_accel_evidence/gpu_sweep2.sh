#!/usr/bin/env bash
# STT·TTS를 먼저 GPU에 올린 상태에서 vLLM 서버 메모리 비율별로 17턴 시나리오를 돌리고 GPU 사용량을 기록한다.
set -u
S=/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad
ENVS=/home/roma/miniconda3/envs
cd /tmp
for util in "$@"; do
  env -u PYTHONPATH $ENVS/gemma4_env/bin/python -I $S/scripts/stt_hold.py > $S/sw_stt_$util.log 2>&1 & STT=$!
  env -u PYTHONPATH $ENVS/cosyvoice3/bin/python -I $S/scripts/tts_hold.py > $S/sw_tts_$util.log 2>&1 & TTS=$!
  until grep -q HOLD $S/sw_stt_$util.log && grep -q HOLD $S/sw_tts_$util.log; do sleep 2; done
  holders=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

  VLLM_GPU_MEM=$util /home/roma/IRC_SOOMAC3.0/tools/vllm_server/start_vllm_server.sh > $S/sw_server_$util.log 2>&1 & SRV=$!
  ( while kill -0 $SRV 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > $S/sw_total_$util.samples & MON=$!
  ready=0
  while kill -0 $SRV 2>/dev/null; do
    if curl -sf http://127.0.0.1:8000/health > /dev/null; then ready=1; break; fi
    sleep 2
  done
  if [ $ready -ne 1 ]; then
    echo "util=$util SERVER FAILED: $(grep -E 'ValueError|OutOfMemory|rror:' $S/sw_server_$util.log | tail -1 | cut -c1-220)"
    kill $MON $STT $TTS 2>/dev/null; wait $STT $TTS 2>/dev/null; sleep 3; continue
  fi
  after_ready=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

  (cd /home/roma/IRC_SOOMAC3.0 && PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH $ENVS/soomac_llm/bin/python \
      $S/run_scenario_vllm.py --measure-all --out $S/sw_scn_$util > $S/sw_scn_$util.log 2>&1); rc=$?
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr '\n' ';')
  hits=$(curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:prefix_cache_(hits|queries)_total" | awk '{print $2}' | tr '\n' ' ')
  kvtok=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" $S/sw_server_$util.log | tail -1)
  kill $SRV; wait $SRV 2>/dev/null; kill $MON 2>/dev/null
  peak=$(sort -n $S/sw_total_$util.samples | tail -1)
  tot=$($ENVS/soomac_llm/bin/python -c "import json;rows=[json.loads(l) for l in open('$S/sw_scn_$util/turns.jsonl')];print(round(sum(r['latency_ms'] for r in rows)/1000,1))" 2>/dev/null)
  echo "util=$util rc=$rc 17turn_s=$tot | GPU MiB: STT+TTS+desktop=$holders, +server_ready=$after_ready, peak=$peak | $kvtok | hits/queries=$hits | apps=$apps"
  kill $STT $TTS; wait $STT $TTS 2>/dev/null; sleep 3
done
echo ALLDONE
