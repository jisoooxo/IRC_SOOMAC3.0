import json
import uuid          # uuid: 전역적으로 겹치지 않는 임의 ID 생성기. 도구호출 1개마다 call_id 로 붙여
                     # 나중에 그 호출과 결과를 짝지을 때 이 번호로 매칭한다

import torch
import xgrammar as xgr  # xgrammar: 모델이 생성할 수 있는 JSON 형식을 토큰 단계에서 강제
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from peft import PeftModel # 파인튜닝 어뎁터 스껄
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForMultimodalLM as _AutoVLM
    print("AutoModelForMultimodalLM 로딩 성공")
except ImportError:
    from transformers import AutoModelForImageTextToText as _AutoVLM
    print("AutoModelForImageTextToText 로딩 성공")


MODEL_PATH = "/home/roma/Desktop/sLLM/gemma-4-12B-it"
TOOL_MAX_TOKEN = 1024
REPLY_MAX_TOKEN = 512
VLM_MAX_TOKEN = 1024
LOAD_4BIT = True


# 모델이 선택할 수 있는 행동 이름

TOOL_NAMES = [
    "set_order", "set_order_and_confirm", "refuse_section", "recommend_order",
    "confirm_section", "cancel_order", "respond", "describe_scene",
]


# JSON Schema에서 사용하는 말
# object               : Python의 dict처럼 key와 value를 묶은 JSON 객체
# string               : "set_order"처럼 따옴표로 감싼 문자열
# properties           : object 안에 들어갈 수 있는 key와 각 key의 형식
# enum                 : 모델이 선택할 수 있는 문자열 목록
# required             : 반드시 출력해야 하는 key 목록
# additionalProperties : properties에 없는 key를 추가로 허용할지 여부
# array                : Python의 list처럼 값 여러 개를 순서대로 담는 JSON 배열
# items                : array 안에 들어갈 값의 형식


# XGrammar는 바깥 JSON 형식만 강제
# changes 내부의 실제 주문값은 order_v2.validate_delta()에서 검사
# 기존처럼 재귀 함수와 oneOf로 Tool마다 Schema를 자동 생성하지 않음
TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "enum": TOOL_NAMES},
        "changes": {"type": "object"},
    },
    "required": ["name", "changes"],
    "additionalProperties": False,
}

TOOL_CALL_SCHEMA_TEXT = json.dumps(TOOL_CALL_SCHEMA, ensure_ascii=False)



CONSTRAINT_SYSTEM = """
입력에 있는 section_order가 실제 주문 진행 순서이다.
사용자가 현재 section 밖의 실제 메뉴를 명확히 요청해도 요청값은 changes에 그대로 넣는다.
실제 주문 반영 여부는 Python의 현재 section 검증 결과가 결정한다.
section_rule에서 미리 선택할 수 있다고 명시한 값은 Python 검증에서 먼저 반영될 수 있다.
모델이 임의로 section을 건너뛰거나 순서를 변경했다고 답하지 않는다.

constraints는 알레르기, 먹을 수 없음, 비건 같은 명확한 식이 제한에만 사용한다.
현재 주문을 먹을 사람이 따로 명시되지 않으면 사용자가 먹는 주문으로 판단한다.
사용자가 자신이 먹지 못한다고 말하면 현재 주문의 constraints에 반영한다.

가족, 친구, 손님, 심부름처럼 현재 주문을 다른 사람이 먹는다고 명확히 말하면
사용자의 개인적인 식이 제한은 현재 주문의 constraints에 넣지 않는다.
이 경우 실제로 먹을 사람에게 명시된 식이 제한만 constraints에 반영한다.

맛·식감·상황·기분을 반영하기 위해 선택한 재료를 constraints에 넣지 않는다.
사용자 또는 실제로 먹을 사람이 피하거나 먹지 못한다고 말하지 않았다면 constraints를 추측하지 않는다.
""".strip()



# Tool 선택용 프롬프트
# 자연어 답변은 여기서 만들지 않고 generate_reply()에서 별도로 생성
TOOL_SYSTEM = f"""
너는 파스타 주문에서 이번 사용자 발화의 행동을 하나 선택한다.

사용 가능한 행동:
- set_order: 주문값이나 식이 제약을 추가·변경한다.
- set_order_and_confirm: 같은 발화에서 주문 변경과 현재 단계 선택 완료를 모두 명확히 요청했다.
- refuse_section: 사용자가 현재 야채, 육류 또는 추가 재료 단계의 모든 재료를 먹지 못하거나 모두 싫다고 명확히 말했다.
- recommend_order: 사용자가 현재 단계 또는 남은 주문을 추천하거나 알아서 선택해 달라고 명확히 요청했다.
- confirm_section: 사용자가 현재 단계 선택을 끝내겠다고 명확히 말했다.
- cancel_order: 사용자가 전체 주문 취소를 명확히 요청했다.
- respond: 주문 변경 없이 질문·안내·잡담에 답해야 한다.
- describe_scene: 최신 카메라 화면을 실제로 봐야 답할 수 있는 요청이다.

STT 결과에는 문장의 앞이나 뒤가 잘린 불완전한 발화가 들어올 수 있다.
현재 message만으로 사용자의 요청이나 확정 의도가 완전하게 이해되지 않으면 respond를 선택한다.
history를 이용해 잘린 부분을 임의로 복원하거나 확정 의도를 추측하지 않는다.
특히 set_order_and_confirm, confirm_section, cancel_order는 현재 발화에서 의도가 명확할 때만 선택한다.
변경 요청만 있으면 set_order를 선택하고, 확정 요청만 있으면 confirm_section을 선택한다.
불완전한 발화에는 changes를 비워서 출력한다.

refuse_section은 현재 section이 veggie, meat 또는 extra이고,
그 단계의 모든 재료를 명확히 거부한 경우에만 선택한다.
재료 하나만 싫다고 했거나 의미가 불명확하면 refuse_section을 선택하지 않는다.

refuse_section의 changes 형식:
{{"reason": "cannot_eat", "items": ["양파", "버섯"]}}

reason은 다음 둘 중 하나만 사용한다.
- cannot_eat: 현재 단계의 모든 재료를 못 먹거나 알레르기 때문에 피해야 한다.
- dislike: 먹을 수는 있지만 현재 단계의 모든 재료가 싫다.

items에는 사용자가 모두 거부한 현재 단계 재료를 한 번씩 넣는다.
다른 단계의 재료를 섞지 않는다.
같은 재료를 반복해서 말했다고 섹션 전체 거부로 판단하지 않는다.
실제 스킵 여부와 dislike 재확인 여부는 Python이 현재 section을 확인하여 결정한다.

recommend_order는 사용자가 “아무거나”, “추천해줘”, “나머지는 알아서”처럼
선택을 맡기겠다고 명확히 요청한 경우에만 사용한다.
현재 단계에서 고를 수 있는 메뉴를 질문하거나,
선택하지 않은 다른 재료를 안내받은 경우에는 recommend_order를 선택하지 않는다.

추천 범위가 불명확한 경우:
{{"scope": "ask", "excluded": [], "constraints": {{}}}}

현재 단계 추천안이 만들어진 경우:
{{"scope": "section", "excluded": ["버섯"], "constraints": {{}}, "recommended_order": {{"toppings": {{"양파": "normal"}}}}}}

사용자가 지정한 단계까지만 추천하는 경우:
{{"scope": "selected", "target_sections": ["veggie", "meat"], "excluded": [], "constraints": {{}}, "recommended_order": {{"toppings": {{"양파": "normal", "소시지": "normal"}}}}}}

면 단계에서 아직 아무것도 선택하지 않았고 남은 주문 전체 추천안이 만들어진 경우:
{{"scope": "all", "excluded": [], "constraints": {{}}, "recommended_order": {{"sauce": "토마토", "noodle_type": "넓적면", "noodle_portion": "normal", "toppings": {{"양파": "normal", "소시지": "normal", "치즈": "normal"}}}}}}

scope는 ask, section, selected, all 중 하나만 사용한다.
ask는 추천 범위를 다시 물어야 하는 상태이며 recommended_order와 target_sections를 넣지 않는다.
section은 현재 단계만 추천한다.
selected는 target_sections에 지정된 현재 또는 미래 단계만 추천한다.
all은 현재 단계부터 남은 음식 단계 전체를 추천한다.
section, selected, all에는 이번 추천으로 새로 제안하는 값만 recommended_order에 넣는다.
selected에만 target_sections를 넣으며 noodle, veggie, meat, extra 순서를 지킨다.
이미 지나간 섹션은 target_sections에 넣지 않는다.
order에 이미 선택된 값과 completed_tasks에 있는 재료는 recommended_order에 다시 넣지 않는다.
기존 선택과 완료 재료는 새로운 재료를 고르는 추천 근거로만 사용한다.
사용자가 기존 값을 명확히 바꿔 달라고 하지 않았다면 추천으로 덮어쓰지 않는다.
추천 대상인 veggie, meat, extra에는 아직 선택되지 않은 실제 재료를 최소 하나 넣는다.
한 단계의 재료 두 개를 모두 추천해도 된다.
all에는 제약이나 명시적 제외로 모두 막힌 경우가 아니라면 extra 재료도 최소 하나 넣는다.
excluded에는 사용자가 싫다고 명확히 말한 실제 메뉴만 넣는다.
constraints에는 먹지 못하거나 알레르기가 있는 제약만 넣는다.
추천안 확정과 실제 order 반영은 Python이 결정한다.

recommendation_state의 phase가 confirming이면 현재 추천안을 확인하는 중이다.
추천안을 그대로 진행하겠다는 명확한 요청은 confirm_section을 선택한다.
추천안 일부를 바꾸거나 빼달라는 요청은 recommend_order를 선택한다.
이때 기존 scope를 유지하고 recommended_order에는 이전에 제안한 신규 추천값을 수정한 전체 결과를 넣는다.
실제 주문에서 이미 선택했거나 완료한 과거 값은 recommended_order에 넣지 않는다.
추천안을 수정하면서 진행해 달라고 해도 먼저 recommend_order로 수정된 추천안을 다시 제시한다.
사용자가 “다른 조합”, “다른 추천”을 요청하면 recommendation_state의 기존 추천안과 다른 실제 메뉴값을 최소 하나 포함한다.
설명이나 순서만 바꾸고 같은 recommended_order를 다시 출력하지 않는다.
추천 확인 중에는 set_order와 set_order_and_confirm으로 실제 order를 직접 변경하지 않는다.
추천 이유나 내용을 묻는 질문은 respond를 선택한다.

입력의 flow_state는 Python이 관리하는 현재 진행 상태이다.
refusal_prompted_section이 현재 section과 같으면 dislike 재확인을 이미 한 번 한 상태이다.
recommendation_state의 phase와 scope를 참고하여 추천 범위 질문, 추천안 생성, 추천 확인을 구분한다.
remaining_item_prompted_section은 남은 메뉴를 이미 안내한 섹션이다.
skipped_sections에 있는 섹션을 완료된 섹션으로 표현하지 않는다.

set_order와 set_order_and_confirm의 changes에는 이번 발화에서 요청한 주문값을 넣는다.
현재 section 밖의 실제 메뉴를 명확히 요청한 경우에도 해당 값을 changes에서 숨기지 않는다.
실제 반영 가능 여부는 Python의 현재 section 검증 결과가 결정한다.
각 주문값은 다음 문자열을 그대로 사용한다.
- sauce: 오일, 토마토, 크림
- noodle_type: 얇은면, 넓적면
- noodle_portion: low, normal, high
- toppings: 양파, 버섯, 소시지, 게살, 치즈, 페퍼론치노
- toppings의 양: low, normal, high, none
사용 가능한 이름은 sauce, noodle_type, noodle_portion, toppings, constraints이다.

사용자가 원하는 맛·식감 또는 자신의 상황·기분을
주문에 반영해 달라고 명확히 요청하면,
사용자 발화와 현재 주문 상태를 바탕으로 현재 단계에서 허용된 주문값 중
요청에 가장 적합한 값을 의미적으로 판단하여 changes에 넣는다.

입력의 action_history는 이전 발화에서 실제로 처리된 최근 행동 결과이다.
“그거”, “아까 요청한 것”, “방금 빠진 것”은 자연어 대화와 action_history를 함께 참고한다.
과거 changed를 현재 changes에 자동으로 복사하지 않는다.
현재 사용자가 다시 요청하거나 변경한 값만 changes에 넣는다.

현재 카메라 화면이나 로봇 앞에 실제로 무엇이 보이는지 묻는 경우에는 describe_scene을 선택한다.
주문 내역, 완료한 작업, 다음 단계처럼 현재 주문 상태만으로 답할 수 있는 질문은 respond를 선택한다.
일반적인 재료 설명처럼 현재 카메라를 볼 필요가 없는 질문도 respond를 선택한다.

사용자가 재료명을 직접 말하지 않았더라도
맛·식감·상황·기분과 주문값의 관계가 명확할 때만 반영할 수 있다.
양을 따로 말하지 않았다면 normal을 사용한다.
여러 주문값을 무조건 추가하지 않고 요청을 반영하는 데 필요한 값만 최소한으로 넣는다.

이미 선택한 소스, 면 종류, 면 양은 사용자가 명확히 변경해 달라고 하지 않으면 바꾸지 않는다.
사용자가 상황이나 기분만 말하고 주문에 반영해 달라고 요청하지 않은 경우,
적절한 값을 확실하게 판단할 수 없는 경우에는 respond를 선택한다.
현재 카메라 화면을 직접 확인해야 하는 요청은 describe_scene을 선택한다.
그 밖에 주문 변경이 없는 질문, 설명 요청, 잡담은 respond를 선택한다.

toppings 형식:
{{"양파": "normal", "버섯": "high"}}

토핑 양:
low, normal, high, none

constraints는 사용자의 자연어 의미를 판단하여 다음 표준 이름으로 출력한다:
갑각류, 유제품, 육류, 비건

constraints 형식:
{{"유제품": true, "갑각류": false}}

true는 현재 주문에 제약을 추가한다.
false는 현재 주문에 있던 제약을 철회한다.
constraints를 배열로 출력하지 않는다.
사용자가 제약을 철회하지 않았다면 기존 제약을 false로 출력하지 않는다.
확실하게 분류할 수 없는 제약은 임의로 추측하지 않는다.

{CONSTRAINT_SYSTEM}

반드시 아래 Schema를 만족하는 JSON 객체 하나만 출력한다.
설명, 마크다운, 특수 Tool 토큰은 출력하지 않는다.

{TOOL_CALL_SCHEMA_TEXT}
""".strip()

"""
현재 단계 밖의 주문 변경 요청도 respond가 아니라 set_order를 선택한다.
실제 반영 가능 여부는 Python 검증 함수가 결정한다. -> 이거는 지수랑 얘기 해봐야 함

나중에 시나리오 구체적으로 어떻게 가져가야할지 ㅇㅇ...
"""


# Reply는 JSON이 아니라 TTS로 보낼 자연어 문자열

# REPLY_SYSTEM = """
# 너는 파스타 주문 안내 직원이다.
# 제공된 확정 사실만 사용해서 자연스러운 한국어 한두 문장으로 답한다.
# JSON, 코드, 필드 이름, thought, 생각 과정은 출력하지 않는다.

# changed는 이번 발화로 실제 주문에서 변경된 항목이다.
# blocked와 dropped에 있는 재료는 주문에 넣었다고 말하지 않는다.
# section_tasks는 현재 단계를 확정하면 실행할 예정 작업이며, 아직 완료한 작업이 아니다.

# action이 set_order이면 changed에 있는 변경만 반영했다고 안내한다.
# 사용자가 재료명을 직접 말하지 않고 맛·식감·상황·기분을 주문에 반영해 달라고 했다면,
# 어떤 요청을 위해 어떤 주문값을 반영했는지 자연스럽게 설명한다.
# “미리 추가했다”는 표현은 면 단계에서 치즈나 페퍼론치노를 조기 선택했을 때만 사용한다. 
# 소스, 면, 야채, 육류, 추가 재료를 현재 단계에서 선택했다면 “선택했다” 또는 “추가했다”고 표현한다.

# action이 confirm_section이고 missing이 비어 있으면 현재 선택이 완료된 것이다.
# section_tasks가 있으면 진행 여부를 다시 묻지 말고 해당 작업을 시작한다고 안내한다.
# section_tasks가 비어 있으면 현재 선택이 끝났다고만 안내하고 작업을 시작했다고 말하지 않는다.

# completed_tasks는 로봇 작업 성공까지 확인된 작업이다.
# completed_tasks에 있는 재료만 실제로 담았다고 표현한다.
# 완료한 작업 전체를 매번 반복해서 나열하지 않고, 현재 대화에 필요한 경우에만 언급한다.

# repeat_count는 로봇 내부 반복 횟수이며 재료 개수가 아니다.
# 사용자에게 repeat_count 숫자를 말하지 않는다.
# 재료 양은 order의 low, normal, high를 기준으로 설명하거나 생략한다.

# 확실하지 않은 사실을 만들어내지 않는다.
# """.strip()

REPLY_SYSTEM = """
너는 파스타 주문 안내 직원이다.
제공된 확정 사실만 사용해서 자연스러운 한국어 한두 문장으로 답한다.
JSON, 코드, 필드 이름, thought, 생각 과정은 출력하지 않는다.

changed는 이번 발화로 실제 주문에서 변경된 항목이다.
blocked는 식이 제약과 충돌하여 주문에서 빠진 항목이다.
dropped는 현재 단계에서 허용되지 않거나 형식이 잘못되어 반영되지 않은 항목이다.
dropped 값이 completed.로 시작하면 이미 로봇이 담은 재료이므로 변경하거나 제거할 수 없다고 안내한다.
section_tasks는 현재 단계를 확정하면 실행할 예정 작업이며 아직 완료되지 않았다.
completed_tasks는 로봇의 성공 신호까지 받은 실제 완료 작업이다.

실제 주문 메뉴는 아래 목록이 전부이다.
- 소스: 오일, 토마토, 크림
- 면: 얇은면, 넓적면
- 양: 적게, 보통, 많이
- 야채: 양파, 버섯
- 육류: 소시지, 게살
- 추가 재료: 치즈, 페퍼론치노

메뉴 질문에는 위 목록에 있는 항목만 안내한다.
보통면, 중간면, 굵은면처럼 목록에 없는 메뉴를 만들어내지 않는다.
현재 단계의 메뉴를 물으면 section에 해당하는 메뉴만 안내한다.
전체 메뉴를 명확히 물었을 때만 전체 목록을 안내한다.
사용자가 추천을 요청하지 않으면 다른 단계의 재료를 먼저 권하지 않는다.

order의 constraints에 포함된 제약과 충돌하는 메뉴는 선택할 수 있다고 안내하지 않는다.
- 갑각류: 게살 제외
- 유제품: 치즈, 크림 제외
- 육류: 소시지 제외
- 비건: 소시지, 치즈 제외

history는 이번 발화 이전에 손님과 로봇이 실제로 주고받은 최근 자연어 대화이다.
“그거”, “아까 말한 것”, “방금 선택한 것” 같은 표현은 history를 참고한다.
history의 과거 내용과 현재 order 또는 completed_tasks가 다르면 현재 상태를 우선한다.

action_history는 이전 발화에서 Tool 선택과 Python 검증까지 끝난 실제 처리 결과이다.
이전 요청이 반영됐는지 물으면 changed, blocked, dropped, missing을 근거로 답한다.
자연어 history와 action_history가 다르면 action_history를 우선한다.
현재 order와 과거 action_history가 다르면 현재 order를 우선한다.


action이 set_order 또는 set_order_and_confirm이면 changed에 있는 내용만 반영했다고 안내한다.
set_order_and_confirm의 실제 확정 여부는 Python이 missing, blocked, dropped를 검사한 뒤 결정한다.
Reply 모델은 현재 선택을 확정했거나 로봇 작업을 시작했다고 임의로 말하지 않는다.
changed가 비어 있으면 주문을 변경했다고 말하지 않는다.
blocked나 dropped에 있는 항목은 선택·추가·변경했다고 말하지 않는다.
changed와 blocked 또는 dropped가 함께 있으면 실제 반영된 내용과 빠진 내용을 구분해서 안내한다.

사용자가 재료명을 직접 말하지 않고 맛·식감·상황·기분을 주문에 반영해 달라고 했다면,
어떤 요청을 위해 어떤 주문값을 반영했는지 자연스럽게 설명한다.
“미리 추가했다”는 표현은 면 단계에서 치즈나 페퍼론치노를 조기 선택했을 때만 사용한다.

action이 respond이면 사용자의 질문이나 잡담에만 답한다.
주문을 변경·확정했다고 말하지 않는다.
로봇 작업을 시작했거나 완료했다고 말하지 않는다.
발화가 잘렸거나 의미를 확정할 수 없으면 내용을 만들어내지 말고 다시 말해 달라고 요청한다.
주문 내역 질문은 order를, 실제 완료 작업 질문은 completed_tasks를 기준으로 답한다.
야채 단계에서 양파와 버섯을 모두 거부하면 둘 중 하나는 선택해야 한다고 안내한다.
육류 단계에서 소시지와 게살을 모두 거부하면 둘 중 하나는 선택해야 한다고 안내한다.

action이 recommend_order이면 recommended_order에 있는 이번 신규 추천값만 안내한다.
order와 completed_tasks는 추천 이유로만 사용하고 기존 선택을 새 추천처럼 다시 나열하지 않는다.
recommendation_scope가 section이면 현재 단계, selected이면 지정 단계까지, all이면 남은 전체 추천임을 안내하고 진행 여부를 한 번 묻는다.

action이 confirm_section이고 missing이 비어 있으면 현재 선택이 완료된 것이다.
section_tasks가 있으면 진행 여부를 다시 묻지 말고 예정 작업 전체를 시작한다고 안내한다.
section_tasks가 한 개이면 해당 재료 하나만 말한다.
section_tasks가 여러 개이면 빠뜨리지 말고 모든 재료를 자연스럽게 함께 말한다.
section_tasks가 비어 있으면 작업을 시작했다고 말하지 않는다.

completed_tasks에 있는 재료만 실제로 담았다고 표현한다.
사용자가 완료 내역을 물으면 completed_tasks 전체를 빠뜨리지 말고 안내한다.
그 밖의 답변에서는 완료 내역을 불필요하게 반복하지 않는다.

repeat_count는 로봇 내부 반복 횟수이며 재료 개수가 아니다.
사용자에게 repeat_count 숫자를 말하지 않는다.
재료 양은 order의 low, normal, high를 기준으로 설명하거나 생략한다.

확실하지 않은 사실을 만들어내지 않는다.
""".strip()

# 여기서 지금 페퍼론치노, 치즈 이런거는 지금 시나리오에 맞춘건데 나중에 한번 바꿀때도 봐야할듯

# 최신 카메라 화면을 자연어로 설명할 때 사용하는 프롬프트
SCENE_SYSTEM = """
너는 로봇의 월드카메라 화면을 사용자에게 설명한다.
현재 이미지에서 직접 확인되는 대상과 상태만 자연스러운 한국어 한두 문장으로 말한다.
화면에 없는 물체를 만들거나 사람의 신원, 감정, 의도를 추측하지 않는다.
잘 보이지 않으면 무엇이 보이지 않는지 솔직하게 안내한다.
JSON, 코드, 판정 문자열은 출력하지 않는다.
""".strip()


def parse_tool_call(raw: str) -> dict | None:
    # XGrammar가 만든 JSON 문자열을 Python 딕셔너리로 변환
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    # 바깥쪽에는 name과 changes 두 key만 있어야 함
    if not isinstance(parsed, dict):
        return None

    if set(parsed) != {"name", "changes"}:
        return None

    name = parsed["name"]
    changes = parsed["changes"]

    if name not in TOOL_NAMES:
        return None

    if not isinstance(changes, dict):
        return None

    # 변경 없이 확정만 요청했다면 confirm_section을 사용해야 한다.
    if name == "set_order_and_confirm" and not changes:
        return None

    if name == "refuse_section":
        if set(changes) != {"reason", "items"}:
            return None

        if changes["reason"] not in ("cannot_eat", "dislike"):
            return None

        items = changes["items"]

        if not isinstance(items, list) or not items:
            return None

        if any(not isinstance(item, str) for item in items):
            return None

        if len(items) != len(set(items)):
            return None

        if any(item not in (
            "양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노") for item in items):
            return None

    if name == "recommend_order":
        if "scope" not in changes:
            return None

        if set(changes) - {"scope", "target_sections", "excluded", "constraints", "recommended_order"}:
            return None

        scope = changes["scope"]
        target_sections = changes.get("target_sections")
        excluded = changes.get("excluded", [])
        constraints = changes.get("constraints", {})
        recommended_order = changes.get("recommended_order")

        if scope not in ("ask", "section", "selected", "all"):
            return None

        if scope == "selected":
            if not isinstance(target_sections, list) or not target_sections:
                return None

            if len(target_sections) != len(set(target_sections)):
                return None

            if any(section not in ("noodle", "veggie", "meat", "extra") for section in target_sections):
                return None

            if target_sections != sorted(target_sections, key=("noodle", "veggie", "meat", "extra").index):
                return None

        elif "target_sections" in changes:
            return None

        if not isinstance(excluded, list):
            return None

        if any(not isinstance(item, str) for item in excluded):
            return None

        allowed_menu = (
            "오일", "토마토", "크림", "얇은면", "넓적면",
            "양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노",
        )

        if len(excluded) != len(set(excluded)):
            return None

        if any(item not in allowed_menu for item in excluded):
            return None

        if not isinstance(constraints, dict):
            return None

        for constraint, enabled in constraints.items():
            if constraint not in ("갑각류", "유제품", "육류", "비건"):
                return None

            if not isinstance(enabled, bool):
                return None

        if scope == "ask":
            if "recommended_order" in changes:
                return None

        else:
            if not isinstance(recommended_order, dict) or not recommended_order:
                return None

            if set(recommended_order) - {"sauce", "noodle_type", "noodle_portion", "toppings"}:
                return None

    return {
        "name": name,
        "changes": changes,
    }


def make_call_model(model, processor):
    # 모델과 processor는 load_model()에서 한 번만 생성해서 이 함수에 넣음
    tokenizer = processor.tokenizer

    # 이 모델은 일반 eos 외에 <turn|> 토큰으로도 한 턴을 끝냄
    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<turn|>"),
    ]

    # tokenizer 단어 목록을 XGrammar가 이해할 수 있는 형식으로 변환
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=len(tokenizer),
        stop_token_ids=stop_ids,
    )

    # JSON Schema는 모델 로드 후 한 번만 컴파일
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_json_schema(TOOL_CALL_SCHEMA)

    @torch.inference_mode()  # 학습이 아니라 추론만 하므로 gradient 계산 안함
    def call_model(messages: list[dict]) -> dict | None:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)

        # prompt_length 이후만 잘라야 기존 프롬프트가 출력에 다시 안 들어감
        prompt_length = inputs["input_ids"].shape[1]

        # LogitsProcessor는 generate() 한 번마다 새로 만들어야 함
        json_processor = XGrammarLogitsProcessor(compiled_grammar)

        output = model.generate(
            **inputs,                   # **은 inputs 딕셔너리 unpacking
            max_new_tokens=TOOL_MAX_TOKEN,
            do_sample=False,            # Tool은 같은 입력에 같은 결과가 나오게 greedy 사용
            use_cache=True,             # 앞 토큰의 KV 계산 결과 재사용
            eos_token_id=stop_ids,
            logits_processor=[json_processor],
        )

        # XGrammar 출력은 일반 JSON이라 특수토큰을 남길 필요 없음
        raw = processor.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()

        parsed = parse_tool_call(raw)

        if parsed is None:
            return None

        # call_id는 나중에 Tool 선택과 실행 결과를 짝지을 때 사용
        return {
            "call_id": uuid.uuid4().hex[:8],
            "name": parsed["name"],
            "changes": parsed["changes"],
        }

    return call_model


def make_generate_reply(model, processor):
    # Tool JSON과 자연어 Reply가 같은 모델과 processor를 공유함
    tokenizer = processor.tokenizer

    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<turn|>"),
    ]

    @torch.inference_mode()  # 학습이 아니라 추론만 진행
    def generate_reply(facts: dict) -> str:
        # facts는 validate → apply → enforce가 끝난 뒤 확정된 사실만 들어옴
        facts_text = json.dumps(facts, ensure_ascii=False)

        messages = [
            {"role": "system", "content": REPLY_SYSTEM},
            {"role": "user", "content": facts_text},
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)

        prompt_length = inputs["input_ids"].shape[1]

        # Tool LoRA는 자연어 reply에 사용하지 않는다
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                output = model.generate(
                            **inputs,
                            max_new_tokens=REPLY_MAX_TOKEN,
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=stop_ids,
                        )

        else:
            output = model.generate(
                **inputs,
                max_new_tokens=REPLY_MAX_TOKEN,
                do_sample=False,
                use_cache=True,
                eos_token_id=stop_ids,
            )

        # Reply는 JSON이 아니라 ui+TTS로 보낼 일반 문자열
        reply = processor.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()

        # Gemma가 thinking을 꺼도 앞에 thought를 일반 문자열로 남기는 경우 제거
        if reply.startswith("thought"):
            reply = reply[len("thought"):].lstrip()

        return reply


    return generate_reply


def load_model(tool_adapter_path=None):
    processor = AutoProcessor.from_pretrained(MODEL_PATH)  # AutoProcessor는 이미지, 텍스트 전처리 담당

    if LOAD_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # 양자화 자료형 타입. nf4, fp4, fp8, int8, int4 등 선택 가능
            bnb_4bit_use_double_quant=True, # double quantization은 VRAM을 줄이는 쪽, False로 놓으면 속도 빨라짐
            bnb_4bit_compute_dtype=torch.bfloat16, # 양자화 계산 시 사용할 자료형
            bnb_4bit_quant_storage=torch.bfloat16, # 양자화된 가중치를 저장할 자료형
        )

        model = _AutoVLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )

        """
        attn_implementation은 Transformer attention 연산을 어떤 backend로 계산할지 결정
        attn_implementation="eager" -> 가장 기본 구현, 호환성은 좋지만 느릴 수 있다.
        attn_implementation="sdpa" -> 파이토치 내장 최적화 attention, 설치 추가로 필요 없고, 보통 안정적이고 빠르다.
        attn_implementation="flash_attention_2" -> 추가 패키지가 필요하며 설치·호환성 이슈가 있을 수 있음
        """

        # eval()은 모델을 평가 모드로 바꿔 추론 결과를 일관되게 함
        model.eval()

    else:
        model = _AutoVLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
        )

        model.eval()

    if tool_adapter_path:
        model = PeftModel.from_pretrained(model, tool_adapter_path)
        model.eval() # 평가모드
    

    return model, processor


def make_call_vlm(model, processor):
    tokenizer = processor.tokenizer

    # VLM은 Tool Calling을 사용하지 않고 일반 자연어를 출력
    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<turn|>"),
    ]

    # stop id에 도달하면 생성 멈춤
    def call_vlm(pil_images, system_prompt, user_text):
        content = []

        # processor가 이미지로 인식할 수 있게 content에 하나씩 추가
        for image in pil_images:
            content.append({"type": "image", "image": image})

        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)

        # 프롬프트 길이를 저장했다가 생성 결과에서 프롬프트 부분을 잘라냄
        prompt_length = inputs["input_ids"].shape[1]

        # VLM 판정, 장면 묘사에서는 Tool LoRA 사용 X

        if isinstance(model, PeftModel):
            with model.disable_adapter():
                output = model.generate(
                            **inputs,
                            max_new_tokens=VLM_MAX_TOKEN,
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=stop_ids,
                        )
                

        # input_ids는 프롬프트를 토큰 번호 리스트로 바꾼 것
        # 이미지 자리도 특수 이미지 토큰으로 들어감
        
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=VLM_MAX_TOKEN,
                do_sample=False,
                use_cache=True,
                eos_token_id=stop_ids,
            )

        return processor.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()

    return call_vlm
