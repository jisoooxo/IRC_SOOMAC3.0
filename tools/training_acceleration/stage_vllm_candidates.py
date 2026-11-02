#!/usr/bin/env python3
"""Stage three original PEFT adapters for isolated vLLM evaluation only.

This reuses the production K→V converter but never writes its default adapter
root. The output directory must not already exist. It does not start a server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/vllm_server"))
import convert_adapters_for_vllm as converter  # noqa: E402

AGENTS = ("decision", "recommendation", "response")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(output: Path, sources: dict[str, Path], model_config: Path) -> dict:
    output = output.resolve()
    production_root = converter.OUT.resolve()
    if (output.exists() or output in (Path("/"), Path("/tmp"), ROOT)
            or output == production_root or production_root in output.parents):
        raise ValueError("a new, non-production staging directory is required")
    if set(sources) != set(AGENTS):
        raise ValueError("exactly decision/recommendation/response sources are required")
    source_identity = {}
    for name in AGENTS:
        source = sources[name].resolve(strict=True)
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("staging path overlaps source adapter")
        files = ("adapter_config.json", "adapter_model.safetensors")
        if not all((source / filename).is_file() for filename in files):
            raise ValueError(f"incomplete original PEFT adapter: {name}: {source}")
        if (source / "CONVERSION.json").exists():
            raise ValueError(f"converted adapter cannot be used as source: {name}")
        source_identity[name] = {"path": str(source),
                                 "file_sha256": {filename: sha(source / filename) for filename in files}}
    config = json.loads(model_config.read_text(encoding="utf-8"))["text_config"]
    global_layers = {index for index, kind in enumerate(config["layer_types"])
                     if kind == "full_attention"}
    if not global_layers:
        raise ValueError("base model has no global attention layers")
    output.mkdir(parents=True, exist_ok=False)
    old_out = converter.OUT
    try:
        converter.OUT = output
        converted = {}
        for name in AGENTS:
            staged = converter.build_staging(name, Path(source_identity[name]["path"]), global_layers)
            final = output / name
            staged.rename(final)
            metadata = json.loads((final / "CONVERSION.json").read_text(encoding="utf-8"))
            if (metadata["source_sha256"] != source_identity[name]["file_sha256"]["adapter_model.safetensors"]
                    or metadata["added_v_proj_tensors"] != 2 * len(global_layers)
                    or metadata["global_layers"] != sorted(global_layers)
                    or sha(final / "adapter_config.json") !=
                    source_identity[name]["file_sha256"]["adapter_config.json"]):
                raise ValueError(f"converted adapter identity mismatch: {name}")
            converted[name] = {"path": str(final),
                               "file_sha256": {filename: sha(final / filename) for filename in
                                               ("adapter_config.json", "adapter_model.safetensors", "CONVERSION.json")}}
        for name in AGENTS:
            for filename, original_sha in source_identity[name]["file_sha256"].items():
                if sha(Path(source_identity[name]["path"]) / filename) != original_sha:
                    raise ValueError(f"source adapter changed during staging: {name}: {filename}")
        receipt = {"status": "ISOLATED_VLLM_STAGE_NOT_MODEL_LOAD",
                   "base_config_sha256": sha(model_config),
                   "global_layers": sorted(global_layers),
                   "sources": source_identity, "converted": converted,
                   "production_adapter_root_targeted": False}
        with (output / "STAGE_RECEIPT.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return receipt
    finally:
        converter.OUT = old_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    for agent in AGENTS:
        parser.add_argument(f"--{agent}-adapter", type=Path, required=True)
    args = parser.parse_args()
    sources = {agent: getattr(args, f"{agent}_adapter") for agent in AGENTS}
    receipt = stage(args.out, sources, Path(converter.MODEL_PATH) / "config.json")
    print(json.dumps({"status": receipt["status"], "output": str(args.out.resolve()),
                      "global_layers": len(receipt["global_layers"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
