#!/usr/bin/env bash
# usage: exp_mtp.sh NAME SPEC_K(0=off) BATCH_INVARIANT(0/1)
set -u
S=/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad
ENVS=/home/roma/miniconda3/envs
name=$1; k=$2; bi=$3; O=$S/mtp_$name; mkdir -p $O
envs="VLLM_GPU_MEM=0.55 VLLM_BATCH_INVARIANT=$bi"
[ "$k" != "0" ] && envs="$envs VLLM_SPEC_MODEL=/home/roma/models/gemma-4-12B-it-assistant VLLM_SPEC_K=$k"
env $envs /home/roma/IRC_SOOMAC3.0/tools/vllm_server/start_vllm_server.sh > $O/server.log 2>&1 & SRV=$!
ready=0; while kill -0 $SRV 2>/dev/null; do curl -sf http://127.0.0.1:8000/health >/dev/null && { ready=1; break; }; sleep 2; done
if [ $ready -ne 1 ]; then echo "$name SERVER FAILED: $(grep -E 'ValueError|RuntimeError|AssertionError|NotImplementedError|OutOfMemory|rror:' $O/server.log | grep -v 'Engine core initialization failed' | tail -2 | cut -c1-260 | tr '\n' ' ')"; exit 0; fi
kv=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" $O/server.log | tail -1); mem_ready=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
cd /tmp; env -u PYTHONPATH $ENVS/soomac_llm/bin/python -I $S/scripts/replay14_http.py $O/replay14.json > $O/replay14.log 2>&1
cd /home/roma/IRC_SOOMAC3.0
for i in 1 2; do PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH $ENVS/soomac_llm/bin/python $S/run_scenario_vllm.py --measure-all --out $O/scn$i > $O/scn$i.log 2>&1; done
curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:(spec_decode_num_(accepted|draft)_tokens_total|spec_decode_num_drafts_total|prefix_cache_(hits|queries)_total)" > $O/metrics.txt
mem_after=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
kill $SRV; wait $SRV 2>/dev/null
eager=$(grep -c -i "enforce.eager\|eager mode" $O/server.log)
echo "$name ok | $kv | server MiB ready=$mem_ready after=$mem_after | $(tr '\n' ' ' < $O/replay14.log)| eager_mentions=$eager"
