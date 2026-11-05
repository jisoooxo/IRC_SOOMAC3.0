"""Bounded, CPU-only overlap audit; matches are evidence, not semantic verdicts."""
import collections
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path('/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed/dataset')
OUT = ROOT / 'training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/audit'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def norm(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value).casefold() if c.isalnum())


def utterance(row):
    i = row['input']
    return i.get('message') or i.get('current_user_text') or i.get('user_text') or ''


def index(rows, key):
    out = collections.defaultdict(list)
    for row in rows:
        k = key(row)
        if k:
            out[k].append(row)
    return out


def run():
    OUT.mkdir(exist_ok=True)
    summary, evidence, hashes = [], [], {}
    for agent in ('decision', 'recommendation', 'response'):
        splits = {}
        for path in sorted((RELEASE / 'records' / agent).glob('*.jsonl')):
            data = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(data).hexdigest()
            splits[path.stem] = [json.loads(line) for line in data.splitlines()]
        train = splits['train']
        inputs = index(train, lambda r: canonical(r['input']))
        messages = index(train, lambda r: norm(utterance(r)))
        groups = index(train, lambda r: r.get('metadata', {}).get('dialogue_group'))
        histories = collections.defaultdict(set)
        for row in train:
            for msg in row['input'].get('recent_history', []):
                if msg.get('role') == 'user' and norm(msg.get('content', '')):
                    histories[norm(msg['content'])].add(row['id'])
        for split, rows in splits.items():
            if split == 'train':
                continue
            counts = collections.Counter()
            for row in rows:
                exact = inputs.get(canonical(row['input']), [])
                same_text = messages.get(norm(utterance(row)), [])
                group = row.get('metadata', {}).get('dialogue_group')
                same_group = groups.get(group, []) if group else []
                conflict = [x for x in exact if canonical(x['target']) != canonical(row['target'])]
                matched_history = sorted(histories.get(norm(utterance(row)), set()))
                hits = dict(exact_input=[x['id'] for x in exact],
                            exact_input_different_target=[x['id'] for x in conflict],
                            normalized_current_text=[x['id'] for x in same_text],
                            train_user_history=matched_history,
                            dialogue_group=[x['id'] for x in same_group])
                for kind, ids in hits.items():
                    if ids:
                        counts[kind] += 1
                if any(hits.values()):
                    evidence.append(dict(agent=agent, split=split, id=row['id'], text=utterance(row),
                                         group=group, matches=hits))
            summary.append(dict(agent=agent, split=split, count=len(rows), train=len(train), overlaps=dict(counts)))
    previous = ROOT / 'training/learning_first_v43/final/goals/GL10/qp_vllm_validation_20261103_r4/inputs'
    used = []
    for name in ('VALIDATION_REVIEWED.json', 'FUNCTIONAL_CASES.json', 'ROW_REVIEW.json'):
        p = previous / name
        data = p.read_bytes()
        hashes[str(p)] = hashlib.sha256(data).hexdigest()
        content = json.loads(data)
        used.append(dict(path=str(p), rows=len(content.get('rows', [])), classification='previously exposed development/selection set'))
    result = dict(status='EXACT_AND_NORMALIZED_OVERLAP_AUDIT', rows=summary,
                  previous_evaluation=used, source_sha256=hashes,
                  limitations=['Different target is a conflict candidate, not proof of wrong semantic label.',
                              'Same short utterance or generic dialogue_group alone does not prove leakage.',
                              'No exhaustive paraphrase or manual semantic certification performed.',
                              'Required scenarios in train are functional regression, not held-out generalization.'])
    for name, value in [('OVERLAP_SUMMARY.json', result), ('OVERLAP_EVIDENCE.json', evidence)]:
        with (OUT / name).open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
