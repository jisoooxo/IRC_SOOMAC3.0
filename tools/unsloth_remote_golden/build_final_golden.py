#!/usr/bin/env python3
"""Build the receipt-bearing Decision train image from the EXACT GPU-tested smoke image.

Fail-closed checks around Dockerfile.final (legacy builder, resource-capped):
  before: tested tag -> tested image ID; receipt digest == tested ID; final tag absent;
          receipt dir holds exactly the receipt + its 3 SHA-pinned evidence files
  after : tested tag unchanged; final RootFS layers start with the tested RootFS layers;
          ENTRYPOINT/CMD/USER = run_training.sh / train / 1000:1000; tested label matches
Never overwrites a tag or directory. Prints the final image ID.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def inspect(ref, template):
    return json.loads(subprocess.check_output(["docker", "image", "inspect", ref, "--format", template]))


def image_id(ref):
    return subprocess.check_output(["docker", "image", "inspect", ref, "--format", "{{.Id}}"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tested-tag", required=True)
    parser.add_argument("--tested-image", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--final-tag", required=True)
    parser.add_argument("--cpuset-cpus", default="24-31", help="build CPUs; default fits this 32-CPU host")
    args = parser.parse_args()

    if image_id(args.tested_tag) != args.tested_image:
        raise SystemExit("tested tag does not resolve to the tested image ID")
    if subprocess.run(["docker", "image", "inspect", args.final_tag],
                      capture_output=True).returncode == 0:
        raise SystemExit("final tag already exists; refusing to overwrite")
    receipt = json.loads((args.receipt_dir / "VALIDATION_RECEIPT.json").read_text())
    if receipt.get("status") != "PASS" or receipt.get("tested_candidate_image_digest") != args.tested_image:
        raise SystemExit("receipt is not a PASS for the tested image")
    files = receipt["evidence_files"]
    expected = {"VALIDATION_RECEIPT.json", *(item["path"] for item in files.values())}
    if {p.name for p in args.receipt_dir.iterdir()} != expected:
        raise SystemExit("receipt dir must contain exactly the receipt and its evidence")
    for item in files.values():
        if hashlib.sha256((args.receipt_dir / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit(f"evidence changed: {item['path']}")

    args.context.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.receipt_dir, args.context / "gpu-validation")
    shutil.copyfile(HERE / "Dockerfile.final", args.context / "Dockerfile.final")
    subprocess.run(["nice", "-n", "10", "docker", "build", "--memory", "4g", "--memory-swap", "4g",
                    "--cpuset-cpus", args.cpuset_cpus, "--cpu-shares", "256", "-f", str(args.context / "Dockerfile.final"),
                    "--build-arg", f"TESTED_CANDIDATE_IMAGE={args.tested_image}",
                    "--build-arg", f"TESTED_CANDIDATE_DIGEST={args.tested_image}",
                    "-t", args.final_tag, str(args.context)],
                   check=True, env={"DOCKER_BUILDKIT": "0", "PATH": "/usr/bin:/bin"})

    if image_id(args.tested_tag) != args.tested_image:
        raise SystemExit("tested tag moved during build")
    tested_layers = inspect(args.tested_image, "{{json .RootFS.Layers}}")
    final_layers = inspect(args.final_tag, "{{json .RootFS.Layers}}")
    config = inspect(args.final_tag, "{{json .Config}}")
    problems = []
    if final_layers[:len(tested_layers)] != tested_layers or len(final_layers) <= len(tested_layers):
        problems.append("final RootFS does not extend the tested image")
    if config["Entrypoint"] != ["/app/run_training.sh"] or config["Cmd"] != ["train"]:
        problems.append(f"entrypoint/cmd: {config['Entrypoint']} {config['Cmd']}")
    if config["User"] != "1000:1000":
        problems.append(f"user: {config['User']}")
    if config["Labels"].get("soomac.tested_candidate") != args.tested_image:
        problems.append("tested label mismatch")
    if problems:
        raise SystemExit("; ".join(problems))
    print(json.dumps({"final_image": image_id(args.final_tag), "tested_image": args.tested_image,
                      "tested_layers": len(tested_layers), "final_layers": len(final_layers)}))


if __name__ == "__main__":
    sys.exit(main())
