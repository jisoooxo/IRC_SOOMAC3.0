set -u
cd /tmp
for env in gemma4_env soomac_llm; do
  PY=/home/roma/miniconda3/envs/$env/bin/python
  PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH $PY -I /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/scripts/bench_with_tf.py - /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/ab_rows_$env.json > /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/ab_bench_$env.log 2>&1
  echo "BENCH $env exit=$? $(grep -E 'SUMMARY|peak' /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/ab_bench_$env.log | tr '\n' ' ')"
done
for env in gemma4_env soomac_llm; do
  PY=/home/roma/miniconda3/envs/$env/bin/python
  cd /home/roma/IRC_SOOMAC3.0
  PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH $PY /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/run_scenario_cache.py on --measure-all --out /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/ab_scn_$env > /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/ab_scn_$env.log 2>&1
  echo "SCENARIO $env exit=$? $(cat /tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/ab_scn_$env/EXIT.json 2>/dev/null | tr -d '\n ' | cut -c1-160)"
done
echo ALLDONE
