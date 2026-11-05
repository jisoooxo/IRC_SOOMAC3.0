"""Real D, fixed reviewed Q/P fixtures; no adapter merging or gold D injection."""
import copy
import json
from unittest.mock import patch
import dqp_evaluation as e


class AnnotatedStream:
    def __init__(self, stream):self.stream=stream
    def write(self, text):
        row=json.loads(text)
        row.update(real_decision=True,real_recommendation=False,real_response=False,
                   scope='Actual sequential Decision with frozen Q/P fixture outputs')
        row['episode']='decision_only_'+row['episode']
        if row['traces']['response']:
            row['reply_evidence']['source']='frozen_response_fixture'
        self.stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    def flush(self):self.stream.flush()


def freeze():
    original=e.BASE.parent/'fixed_history_vllm_20261104/TURNS.jsonl'
    rows=[json.loads(s) for s in original.read_text().splitlines()]
    selected=[r for r in rows if r['episode'] in ('compound','split_control')]
    e.b.write_new(e.BASE/'inputs/D_ONLY_FIXTURE.json',dict(source=str(original),source_sha256=e.b.sha(original),
        response=[dict(episode=r['episode'],user_text=r['user_text'],reply=r['reply']) for r in selected],
        recommendation=dict(proposal=dict(sauce='크림',noodle_type=None,noodle_portion=None,toppings={}),reason_tags=['느끼한 소스 요청']),
        classification='Known functional fixture, not novel heldout, no model-generated Q/P in this run'))


def evaluate(model,processor,decision,stream):
    fixture=e.b.read(e.BASE/'inputs/D_ONLY_FIXTURE.json')
    results=[]
    for compound in (True,False):
        name='compound' if compound else 'split_control'
        replies={r['user_text']:r['reply'] for r in fixture['response'] if r['episode']==name}
        def q(*args):
            value=copy.deepcopy(fixture['recommendation'])
            q.trace_events.append(dict(stage='recommendation',frozen_fixture=True,
                model_input=e.build_recommendation_model_input(*args),parsed=value))
            return value
        def p(*args):
            value=replies[args[0]]
            p.trace_events.append(dict(stage='response',frozen_fixture=True,
                model_input=e.build_response_model_input(*args),route=args[8],raw=value))
            return value
        p.trace_events=[];q.trace_events=[]
        with patch('soomac_irc.decision_model.make_call_decision',lambda *a,**k:decision),\
             patch('soomac_irc.recommendation_model.make_call_recommendation',lambda *a,**k:q),\
             patch('soomac_irc.response_model.make_call_response',lambda *a,**k:p):
            result=e.probe.run_episode(compound,model,processor,AnnotatedStream(stream),execution_reply_mode='fixed')
        result['episode']='decision_only_'+name
        result.pop('reply_sources',None)
        result['real_recommendation']=False;result['real_response']=False
        results.append(result)
    return results


if __name__=='__main__':freeze()
