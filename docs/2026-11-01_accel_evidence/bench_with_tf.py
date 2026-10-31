"""Run bench_prefix_cache.main() with an alternate transformers prepended (argv: tf_path, out_json)."""
import sys, json, importlib.util
tf_path, out_json = sys.argv[1], sys.argv[2]
if tf_path != "-":
    sys.path.insert(0, tf_path)
import transformers
print("transformers", transformers.__version__, flush=True)
spec = importlib.util.spec_from_file_location("bench", "/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/bench_prefix_cache.py")
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
bench.MAX_PER_KEY = {"decision": 6, "recommendation": 2, "response/task": 4, "response/general": 2}
_orig_dump = json.dump
def dump(obj, fp, *a, **k):
    if getattr(fp, "name", "").endswith("bench_rows.json"):
        fp = open(out_json, "w")
    return _orig_dump(obj, fp, *a, **k)
bench.json.dump = dump
bench.main()
