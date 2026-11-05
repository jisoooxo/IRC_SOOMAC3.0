"""Read OCI result layer without loading Docker or executing archived code.

Extract only adapter weights and bounded text metadata. Optimizer/pickle files
are deliberately not deserialized. Exclusive outputs permit a preserved audit.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time


class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data


def run(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    inventory = output / 'ARCHIVE_INVENTORY.jsonl'
    start = time.monotonic()
    total = 0
    with tarfile.open(archive, 'r:') as outer, inventory.open('x') as ledger:
        manifests = json.load(outer.extractfile('manifest.json'))
        assert len(manifests) == 1
        layer = manifests[0]['Layers'][-1]
        reader = DigestReader(outer.extractfile(layer))
        inner = tarfile.open(fileobj=reader, mode='r|*')
        for member in inner:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != 'results':
                raise ValueError('Unsafe/non-result path: ' + member.name)
            if member.issym() or member.islnk() or path.name.startswith('.wh.'):
                raise ValueError('Result link/whiteout requires separate review: ' + member.name)
            keep = member.isfile() and (
                path.name == 'adapter_model.safetensors' or
                (path.suffix in {'.json', '.jsonl', '.md', '.txt', '.log', '.jinja', '.yaml', '.yml'}
                 and member.size <= 64 * 1024 * 1024))
            record = dict(name=member.name, size=member.size, type=member.type.decode(), extracted=keep)
            if keep:
                total += member.size
                if total > 35 * 1024 ** 3:
                    raise ValueError('Extraction budget exceeded')
                target = output / 'extracted' / path
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with inner.extractfile(member) as source, open(target, 'xb') as dest:
                    while chunk := source.read(4 * 1024 ** 2):
                        digest.update(chunk)
                        dest.write(chunk)
                record['sha256'] = digest.hexdigest()
                if path.name == 'adapter_model.safetensors':
                    print(json.dumps(dict(adapter=member.name, extracted_bytes=total)), flush=True)
            ledger.write(json.dumps(record) + '\n')
            ledger.flush()
        while reader.read(4 * 1024 ** 2):
            pass
        actual = reader.digest.hexdigest()
        if actual != layer.rsplit('/', 1)[-1]:
            raise ValueError('Result layer SHA mismatch')
        stat = archive.stat()
        receipt = dict(archive=str(archive), archive_size=stat.st_size,
                       archive_mtime_ns=stat.st_mtime_ns, layer=layer,
                       verified_compressed_layer_sha256=actual, compressed_bytes=reader.size,
                       extracted_bytes=total, seconds=time.monotonic()-start,
                       whole_archive_sha256=None,
                       scope='Last result layer only; no image execution or optimizer deserialization')
        (output / 'EXTRACTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.archive, args.output)
