from pathlib import Path

script = Path("/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/scripts/eval_decision_adapter.py")
source = script.read_text(encoding="utf-8")
old = "compile_json_schema(schema)"
new = "compile_json_schema(schema, any_order=True)"
if source.count(old) != 1:
    raise RuntimeError(f"expected exactly one grammar call, found {source.count(old)}")
source = source.replace(old, new)
exec(compile(source, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
