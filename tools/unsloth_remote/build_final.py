"""Build the receipt-bearing final image only from the GPU-tested local candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def image_info(reference):
    result = subprocess.run(["docker", "image", "inspect", reference], check=True,
                            capture_output=True, text=True)
    items = json.loads(result.stdout)
    if len(items) != 1:
        raise ValueError("Docker image inspect returned an ambiguous image")
    return items[0]


def verify_lineage(candidate, final, expected_id):
    """RootFS prefix proves final adds layers onto the exact inspected candidate."""
    if candidate.get("Id") != expected_id or not DIGEST.fullmatch(expected_id):
        raise ValueError("candidate image ID differs from validation receipt")
    before = candidate.get("RootFS", {}).get("Layers", [])
    after = final.get("RootFS", {}).get("Layers", [])
    if not before or len(after) <= len(before) or after[:len(before)] != before:
        raise ValueError("final image is not a layer extension of tested candidate")
    if final.get("Os") != "linux" or final.get("Architecture") != "amd64":
        raise ValueError("final platform is not linux/amd64")


def validate_receipt_files(context):
    root = context / "validation"
    receipt_path = root / "VALIDATION_RECEIPT.json"
    if not root.is_dir() or root.is_symlink() or not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("validation receipt is missing or linked")
    receipt = json.loads(receipt_path.read_text())
    expected_id = receipt.get("tested_candidate_image_digest")
    if receipt.get("status") != "PASS" or receipt.get("agent") != "decision" or not isinstance(expected_id, str) or not DIGEST.fullmatch(expected_id):
        raise ValueError("validation receipt has no passing Decision candidate ID")
    evidence = receipt.get("evidence_files")
    if not isinstance(evidence, dict) or set(evidence) != {"loss", "gradient", "restore"}:
        raise ValueError("validation receipt must reference loss, gradient and restore evidence")
    names = {"VALIDATION_RECEIPT.json"}
    for item in evidence.values():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("invalid evidence descriptor")
        name, expected_sha = item["path"], item["sha256"]
        if (not isinstance(name, str) or Path(name).name != name or not name.endswith(".json")
                or not isinstance(expected_sha, str) or not SHA.fullmatch(expected_sha)):
            raise ValueError("invalid evidence name or SHA")
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 65536:
            raise ValueError(f"missing, linked or oversized evidence: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"evidence SHA mismatch: {name}")
        names.add(name)
    if {path.name for path in root.iterdir()} != names or receipt_path.stat().st_size > 65536:
        raise ValueError("validation directory must contain exactly four bounded JSON files")
    return expected_id


def build(context, candidate_tag, final_tag, dockerfile):
    context = context.resolve(strict=True)
    if not context.is_dir():
        raise ValueError("build context must be a directory")
    expected_id = validate_receipt_files(context)
    candidate = image_info(candidate_tag)
    if candidate.get("Id") != expected_id:
        raise ValueError("local candidate tag does not identify GPU-tested image")
    if candidate.get("Os") != "linux" or candidate.get("Architecture") != "amd64":
        raise ValueError("candidate platform is not linux/amd64")
    # Never replace an existing final image/tag. Docker inspect returns 1 for absent tags.
    existing = subprocess.run(["docker", "image", "inspect", final_tag],
                              capture_output=True, text=True)
    if existing.returncode == 0:
        raise ValueError("final image tag already exists; choose a new tag")
    subprocess.run([
        "docker", "build", "--platform", "linux/amd64", "--pull=false",
        "--network", "none", "--target", "final",
        "--build-arg", f"TESTED_CANDIDATE_IMAGE={candidate_tag}",
        "--build-arg", f"TESTED_CANDIDATE_DIGEST={expected_id}",
        "-f", str(dockerfile.resolve(strict=True)), "-t", final_tag, str(context),
    ], check=True)
    if image_info(candidate_tag).get("Id") != expected_id:
        raise ValueError("candidate tag changed during final build; do not save TAR")
    final = image_info(final_tag)
    verify_lineage(candidate, final, expected_id)
    return final["Id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--final-tag", required=True)
    parser.add_argument("--dockerfile", type=Path, default=Path(__file__).with_name("Dockerfile"))
    args = parser.parse_args()
    print(json.dumps({"final_image_id": build(args.context, args.candidate_tag,
                                               args.final_tag, args.dockerfile)}, sort_keys=True))


if __name__ == "__main__":
    main()
