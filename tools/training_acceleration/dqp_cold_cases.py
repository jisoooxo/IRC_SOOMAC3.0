"""Authored before new Decision inference; semantic acceptance, not exact wording."""

def episodes():
    def t(text, decision=None, changes=None, *, route='task', execute=False, review=''):
        return dict(text=text, decision={'route': route, **(decision or {})},
                    changes=changes or {}, execute=execute, review=review)
    return [dict(name='cold_general_mixed', section='noodle', turns=[
        t('반가워, 오늘 처음 이용해 봐.', route='general', review='인사, 주문/실행 창작 없음'),
        t('여기 주문 도우미는 무슨 역할을 맡고 있어?', route='general', review='역할 설명'),
        t('너도 점심 먹고 일하는 거야?', route='general', review='실제 식사 경험 창작 없음'),
        t('얇은면은 조금으로 정할게. 그런데 파스타는 원래 어느 나라 음식이야?',
          {'order': {'noodle_type':'얇은면','noodle_portion':'low'}},
          {'noodle_type':'얇은면','noodle_portion':'low'}, route='mixed', review='주문과 일반 질문 모두 반영'),
        t('소스는 오일로 골라 놓고 아직 조리는 시작하지 마.', {'order':{'sauce':'오일'}}, {'sauce':'오일'}),
        t('지금 정해 둔 면 양하고 소스를 짧게 확인해 줘.', review='얇은면 low/오일, 미실행'),
        t('좋아, 지금 선택한 걸 바로 담기 시작해 줘.', {'commit':True}, execute=True),
    ]), dict(name='cold_quantity_reference', section='veggie', turns=[
        t('채소는 양파를 많이, 버섯을 보통으로 선택해 둘게.',
          {'order':{'toppings':{'양파':'high','버섯':'normal'}}}, {'toppings':{'양파':'high','버섯':'normal'}}),
        t('양파가 아니라 버섯 쪽 양을 적게 수정해 줘.', {'order':{'toppings':{'버섯':'low'}}}, {'toppings':{'버섯':'low'}}),
        t('양파도 조금으로 줄이자. 양파는 익히면 왜 단맛이 나?', {'order':{'toppings':{'양파':'low'}}},
          {'toppings':{'양파':'low'}}, route='mixed', review='low 정정+질문 응답'),
        t('담기는 잠시 보류하고, 방금 수정된 두 채소 양을 말해 줘.', review='둘 다 low, 실행 없음'),
        t('이제 그 선택 그대로 바로 담아 주면 돼.', {'commit':True}, execute=True),
    ]), dict(name='cold_restriction_situation', section='meat', turns=[
        t('나는 게살을 먹으면 알레르기가 생겨. 게살 없이 소시지 보통으로 선택해 둬.',
          {'restrictions':[{'target':'게살','reason':'allergy','action':'add'}],
           'order':{'toppings':{'게살':'none','소시지':'normal'}}},
          {'restrictions':[{'target':'게살','reason':'allergy'}],'toppings':{'소시지':'normal'}}),
        t('현재 소시지는 선택만 한 상태야, 담기가 끝난 상태야?', review='선택만 했음; 이전 면 완료와 구분'),
        t('게살은 적게라도 추가해서 골라 줘.', {'order':{'toppings':{'게살':'low'}}}, review='D 추출, runtime 알레르기 차단'),
        t('그러면 소시지만 많은 양으로 수정해 두자.', {'order':{'toppings':{'소시지':'high'}}}, {'toppings':{'소시지':'high'}}),
        t('내가 못 먹는 재료하고 지금 고른 소시지 양을 확인해 줘.', review='게살 allergy/소시지 high'),
        t('확인했어. 그 소시지를 지금 바로 담아 줘.', {'commit':True}, execute=True),
    ]), dict(name='cold_preference_recommendation', section='noodle', turns=[
        t('얇은면 보통으로 먼저 고르고, 소스만 하나 권해 줘. 오늘은 느끼한 소스가 당겨.',
          {'order':{'noodle_type':'얇은면','noodle_portion':'normal'},'recommendation':{'action':'request'}},
          {'noodle_type':'얇은면','noodle_portion':'normal'}, review='취향 근거 크림 후보, 수락 전 미확정'),
        t('네가 제안한 소스가 내 요청에 어울리는 이유가 뭐야?', review='후보 설명; 새 추천이나 확정 없음'),
        t('그 후보로 선택해 둘게. 지금 담으라는 뜻은 아니야.', {'confirmation':'accept'}, {'sauce':'크림'}),
        t('내 소스는 확정했지만 면은 아직 담기 전인 거 맞지?', review='크림 확정, 담기 전'),
        t('이제 선택한 면을 그 양 그대로 바로 담아 줘.', {'commit':True}, execute=True),
    ])]
