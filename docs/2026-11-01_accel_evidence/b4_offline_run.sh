ip link set lo up
/home/roma/IRC_SOOMAC3.0/tools/vllm_server/start_vllm_server.sh > /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/b4_server.log 2>&1 &
SERVER=$!
env -u PYTHONPATH /home/roma/miniconda3/envs/soomac_llm/bin/python -I /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/scripts/b4_offline_smoke.py
kill $SERVER; wait $SERVER 2>/dev/null
