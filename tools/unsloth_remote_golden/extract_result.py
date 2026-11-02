#!/usr/bin/env python3
"""Safely extract results/ from a lab result TAR (docker image save of the result image).

Untrusted input: only regular files and directories under results/ are written, into a NEW
directory; absolute paths, '..', links, devices and whiteouts are skipped. Layers are applied
in manifest order so later layers win. Run with `python3 -I`.
  extract_result.py <result.tar> <new output dir>
"""
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile


def main():
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=False)
    written, skipped = 0, 0
    with tarfile.open(source) as outer:
        manifest = json.load(outer.extractfile("manifest.json"))
        if len(manifest) != 1:
            raise SystemExit("expected one image in result TAR")
        for layer in manifest[0]["Layers"]:
            stream = outer.extractfile(layer)
            head = stream.read(2)
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r:gz" if head == b"\x1f\x8b" else "r:") as inner:
                for member in inner:
                    name = PurePosixPath(member.name)
                    if (name.is_absolute() or ".." in name.parts or not name.parts
                            or name.parts[0] != "results" or name.name.startswith(".wh.")):
                        continue
                    target = output.joinpath(*name.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with inner.extractfile(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 22)
                        written += 1
                    else:
                        skipped += 1
    print(json.dumps({"written_files": written, "skipped_non_regular": skipped,
                      "output": str(output / "results")}))


if __name__ == "__main__":
    main()
