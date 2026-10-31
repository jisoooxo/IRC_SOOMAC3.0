# usage: mem_probe.sh <name> <python> <script>
name=$1; py=$2; script=$3; out=$4
cd /tmp; env -u PYTHONPATH $py -I $script > $out/$name.log 2>&1 &
pid=$!
: > $out/$name.samples
while kill -0 $pid 2>/dev/null; do
  m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p{print $2}')
  echo "$(date +%s.%N) ${m:-0} $(grep -c 'LOADED' $out/$name.log)" >> $out/$name.samples; sleep 0.2
done
awk '{if($3==1 && !idle){idle=$2} if($2>max)max=$2} END{printf "'$name' idle_after_load=%d MiB peak=%d MiB\n", idle, max}' $out/$name.samples
