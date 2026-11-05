"""CPU tokenizer-only verification of saved actual requests; no model or HTTP."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def audit(root,run):
    cases={r['record']['id']:r for r in json.loads((root/'inputs/CASES.json').read_text())['rows']}
    readlines=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
    expected=[]
    for row in readlines(run/'INDIVIDUAL.jsonl'):
        case=cases[row['id']]
        expected.append(dict(id=row['id'],agent=row['agent'],model_input=case['record']['input'],
                             raw=row['traces'][-1].get('raw') if row['traces'] else None))
    for filename in ('TURNS.jsonl','D_ONLY_TURNS.jsonl'):
        path=run/filename
        if not path.exists():continue
        for row in readlines(path):
            for role in ('decision','recommendation','response'):
                for trace in row['traces'][role]:
                    if trace.get('frozen_fixture'):continue
                    expected.append(dict(id=f"{row['episode']}/{row['turn']}/{role}",agent=role,
                                         model_input=trace['model_input'],raw=trace.get('raw')))
    requests=readlines(run/'REQUESTS.jsonl')
    assert len(expected)==len(requests),(len(expected),len(requests))
    tok=Tokenizer.from_file('/home/roma/Desktop/sLLM/gemma-4-12B-it/tokenizer.json')
    canonical=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    proof=[]
    for index,(want,actual) in enumerate(zip(expected,requests),1):
        request=actual['request'];decoded=tok.decode(request['prompt_token_ids'],skip_special_tokens=False)
        assert request['model']==want['agent'],want['id']
        block=decoded.rsplit('<|turn>user\n',1)[1];value,end=json.JSONDecoder().raw_decode(block)
        assert canonical(value)==canonical(want['model_input']),want['id']
        if want['raw'] is not None:assert want['raw']==actual['response']['text'].strip(),want['id']
        proof.append(dict(line=index,id=want['id'],agent=want['agent'],exact_json_value_and_type=True,
                          raw_matches_http=want['raw'] is not None))
    record=dict(count=len(proof),rows=proof,requests_sha256=hashlib.sha256((run/'REQUESTS.jsonl').read_bytes()).hexdigest(),
                cpu_tokenizer_only=True,key_order_ignored=True,value_type_preserved=True,
                frozen_QP_fixtures_excluded=True)
    with (run/'TRANSPORT_PROOF.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(run=run.name,actual_requests_verified=len(proof))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    audit(a.run.resolve().parent.parent,a.run.resolve())
