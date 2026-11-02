"""Bind an installed Python environment to the golden Response training environment.

export: run with the GOLDEN interpreter (CPU only, imports nothing heavy). Checks every
        on-disk file of the required distributions against its RECORD hash, then writes
        GOLDEN_RECORDS.json (version + per-file sha256 for files under site-packages).
verify: run with the CANDIDATE interpreter. Requires the same distribution set and
        versions, and every recorded file byte-identical to the golden install.

Only stdlib is used so this never triggers torch/unsloth import or CUDA.
"""
import argparse
import base64
import csv
import hashlib
import importlib.metadata as md
import json
import re
import sys
from pathlib import Path

SKIP_NAMES = {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()


def effective_distributions():
    """First distribution per canonical name in sys.path order, like importlib."""
    seen = {}
    for dist in md.distributions():
        seen.setdefault(canonical(dist.metadata["Name"]), dist)
    return seen


def recorded_files(dist):
    root = Path(dist.locate_file(""))
    text = dist.read_text("RECORD") or ""
    for row in csv.reader(text.splitlines()):
        if len(row) < 2 or not row[1].startswith("sha256="):
            continue
        rel = row[0]
        if rel.startswith("..") or rel.startswith("/") or rel.endswith(".pyc"):
            continue
        if rel.split("/")[-1] in SKIP_NAMES and ".dist-info/" in rel:
            continue
        yield rel, row[1][len("sha256="):], root / rel


def export(args):
    required = set(json.loads(Path(args.required).read_text()))
    dists = effective_distributions()
    missing = sorted(required - set(dists))
    if missing:
        raise SystemExit(f"golden lacks required distributions: {missing}")
    records, modified = {}, {}
    for name in sorted(required):
        dist = dists[name]
        files = {}
        for rel, expected, path in recorded_files(dist):
            if not path.is_file():
                modified.setdefault(name, []).append(f"missing:{rel}")
                continue
            actual = file_hash(path)
            if actual != expected:
                modified.setdefault(name, []).append(f"changed:{rel}")
            files[rel] = actual
        records[name] = {"name": dist.metadata["Name"], "version": dist.version, "files": files}
    out = {"schema_version": 1, "python": sys.version, "prefix": sys.prefix,
           "distributions": records, "golden_modified_vs_record": modified}
    Path(args.output).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"required": len(records), "files": sum(len(r["files"]) for r in records.values()),
                      "golden_modified_vs_record": {k: len(v) for k, v in modified.items()}}))


def verify(args):
    golden = json.loads(Path(args.records).read_text())
    allowed_extra = {canonical(x) for x in args.allow_extra}
    dists = effective_distributions()
    want = golden["distributions"]
    problems = []
    extra = sorted(set(dists) - set(want) - allowed_extra)
    missing = sorted(set(want) - set(dists))
    if extra:
        problems.append(f"unexpected distributions: {extra}")
    if missing:
        problems.append(f"missing distributions: {missing}")
    checked = 0
    for name, item in sorted(want.items()):
        dist = dists.get(name)
        if dist is None:
            continue
        if dist.version != item["version"]:
            problems.append(f"{name}: version {dist.version} != golden {item['version']}")
            continue
        root = Path(dist.locate_file(""))
        for rel, expected in item["files"].items():
            path = root / rel
            if not path.is_file():
                problems.append(f"{name}: missing file {rel}")
            elif file_hash(path) != expected:
                problems.append(f"{name}: content differs {rel}")
            checked += 1
    result = {"status": "PASS" if not problems else "FAIL", "python": sys.version,
              "distributions": len(want), "files_checked": checked,
              "problems": problems[:200], "problem_count": len(problems)}
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "problems"}))
    if problems:
        for line in problems[:20]:
            print(line, file=sys.stderr)
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--required", required=True)
    e.add_argument("--output", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--records", required=True)
    v.add_argument("--output", required=True)
    v.add_argument("--allow-extra", nargs="*", default=[])
    args = parser.parse_args()
    export(args) if args.command == "export" else verify(args)


if __name__ == "__main__":
    main()
