/home/roma/IRC_SOOMAC3.0/tools/vllm_server/start_vllm_server.sh > /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/b4_scn_server.log 2>&1 &
SERVER=$!
cd /tmp
env -u PYTHONPATH /home/roma/miniconda3/envs/soomac_llm/bin/python -c "import sys; sys.path.insert(0,'/home/roma/IRC_SOOMAC3.0/src/soomac_irc'); from soomac_irc import vllm_client; vllm_client.wait_until_ready(900); print('server ready')"
curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:prefix_cache_(hits|queries)_total" > /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/b4_metrics_before.txt
nvidia-smi --query-gpu=memory.used --format=csv,noheader > /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/b4_gpu_mem.txt
cd /home/roma/IRC_SOOMAC3.0 && PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH /home/roma/miniconda3/envs/soomac_llm/bin/python /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/run_scenario_vllm.py --measure-all --out /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/b4_scn_vllm > /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/b4_scn_vllm.log 2>&1
echo "scenario exit=$?"
curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:prefix_cache_(hits|queries)_total" > /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/b4_metrics_after.txt
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/b4_gpu_mem.txt
kill $SERVER; wait $SERVER 2>/dev/null
echo ALLDONE
