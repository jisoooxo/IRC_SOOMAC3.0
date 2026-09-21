import copy
import json
import time
import uuid

from soomac_irc.order import (
    ALL_TOPPINGS,
    AMOUNTS,
    NOODLE_TYPES,
    RESTRICTION_CATEGORY_ITEMS,
    RESTRICTION_REASONS,
    SAUCES,
    SECTION_ORDER,
    TOPPING_AMOUNTS,
    missing_requirements, # 필수값 확인용
)

from soomac_irc.prompts import (
    FREE_REPLY_SYSTEM,
    HISTORY_TURNS,
    build_tool_system_prompt,
    section_rule_for,
    selected_keys_for_section,
)

import torch
import xgrammar as xgr
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

try:
    from transformers import AutoModelForMultimodalLM as _AutoVLM
except ImportError:
    # transformers 구버전 호환
    from transformers import AutoModelForImageTextToText as _AutoVLM


MODEL_PATH = "/home/roma/Desktop/sLLM/gemma-4-12B-it"

# int8: 현재 Tool LoRA 추론
# nf4: 메모리를 더 줄이거나 차후 QLoRA adapter를 시험할 때 사용
# bf16: 양자화 없이 원본 정밀도로 실행
MODEL_QUANTIZATION = "int8"

VLM_MAX_TOKENS = 1024

TOOL_INPUT_MAX_TOKENS = 3328 # 입력 3328 + 합법 Tool JSON 최대 512 + 여유 256 = 학습 4096
TOOL_MAX_TOKENS = 1024       # 복합 추천·다중 restriction 출력 잘림 방지

# 기존 8개 Tool 이름은 유지한다.
TOOL_CHANGE_FIELDS = {
    "set_order": {"sauce", "noodle_type", "noodle_portion", "toppings", "restriction_changes"},
    "set_order_and_confirm": {"sauce", "noodle_type", "noodle_portion", "toppings", "restriction_changes"},
    "refuse_section": {"reason", "items"},
    "recommend_order": {
        "scope", "sections", "targets", "excluded", "preferences",
        "different", "recommended_order", "restriction_changes",
    },
    "confirm_section": set(),
    "cancel_order": set(),
    "respond": set(),
    "describe_scene": set(),
}

TOOL_NAMES = list(TOOL_CHANGE_FIELDS)

FOOD_SECTIONS = SECTION_ORDER[:4]

ALL_MENU = SAUCES + NOODLE_TYPES + ALL_TOPPINGS

RESTRICTION_TARGETS = ALL_MENU + list(RESTRICTION_CATEGORY_ITEMS)


# 추천에서 사용하는 고정 의미 태그
PREFERENCE_TAGS = [
    "꾸덕한", "담백한", "매콤한", "고소한",
    "푸짐한", "심플한", "재료_다양한",
]


def build_tool_call_schema() -> dict:
    # XGrammar가 생성할 수 있는 Tool 이름·필드·canonical 값을 제한한다.
    # 반환 Schema는 모델 출력 형식만 제한하며 주문 상태는 수정하지 않는다.
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": TOOL_NAMES,
            },
            "changes": {
                "type": "object",
                "properties": {
                    "sauce": {
                        "type": "string",
                        "enum": SAUCES,
                    },
                    "noodle_type": {
                        "type": "string",
                        "enum": NOODLE_TYPES,
                    },
                    "noodle_portion": {
                        "type": "string",
                        "enum": AMOUNTS,
                    },
                    "toppings": {
                        "type": "object",
                        "properties": {
                            item: {
                                "type": "string",
                                "enum": TOPPING_AMOUNTS,
                            }
                            for item in ALL_TOPPINGS
                        },
                        "additionalProperties": False,
                    },
                    "restriction_changes": {
                        "type": "array",
                        "maxItems": len(RESTRICTION_TARGETS),
                        "uniqueItems": True,
                        "items": {
                            "type": "object",
                            "properties": {
                                "target": {
                                    "type": "string",
                                    "enum": RESTRICTION_TARGETS,
                                },
                                "reason": {
                                    "type": "string",
                                    "enum": RESTRICTION_REASONS,
                                },
                                "enabled": {
                                    "type": "boolean",
                                },
                            },
                            "required": ["target", "reason", "enabled"],
                            "additionalProperties": False,
                        },
                    },
                    "reason": {
                        "type": "string",
                        "enum": RESTRICTION_REASONS,
                    },
                    "items": {
                        "type": "array",
                        "maxItems": len(ALL_TOPPINGS),
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": ALL_TOPPINGS,
                        },
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["ask", "current", "remaining", "specified"],
                    },
                    "sections": {
                        "type": "array",
                        "maxItems": len(FOOD_SECTIONS),
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": FOOD_SECTIONS,
                        },
                    },
                    "targets": {
                        "type": "array",
                        "maxItems": len(ALL_MENU),
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": ALL_MENU,
                        },
                    },
                    "excluded": {
                        "type": "array",
                        "maxItems": len(ALL_MENU),
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": ALL_MENU,
                        },
                    },
                    "preferences": {
                        "type": "array",
                        "maxItems": len(PREFERENCE_TAGS),
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": PREFERENCE_TAGS,
                        },
                    },
                    "different": {
                        "type": "boolean",
                    },
                    "recommended_order": {
                        "type": "object",
                        "properties": {
                            "sauce": {
                                "type": "string",
                                "enum": SAUCES,
                            },
                            "noodle_type": {
                                "type": "string",
                                "enum": NOODLE_TYPES,
                            },
                            "noodle_portion": {
                                "type": "string",
                                "enum": AMOUNTS,
                            },
                            "toppings": {
                                "type": "object",
                                "properties": {
                                    item: {
                                        "type": "string",
                                        "enum": AMOUNTS,
                                    }
                                    for item in ALL_TOPPINGS
                                },
                                "additionalProperties": False,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
        "required": ["name", "changes"],
        "additionalProperties": False,
    }


TOOL_CALL_SCHEMA = build_tool_call_schema() # 스키마 객체 반환

TOOL_CALL_SCHEMA_TEXT = json.dumps(
    TOOL_CALL_SCHEMA,
    ensure_ascii=False,
    separators=(",", ":"),
) # # json to string.

TOOL_SYSTEM = build_tool_system_prompt(TOOL_CALL_SCHEMA_TEXT)  # 프롬프트 + json schema(하단에 박아둠)


FREE_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {
            "type": "string",
        },
    },
    "required": ["explanation"], # 필수적으로 explanation 필드가 있어야한다는 소리.
    "additionalProperties": False,
}


"""
TOOL_CALL_SCHEMA: 생성 가능한 Tool명·필드·메뉴값 제한
XGrammarLogitsProcessor: 모델 생성 중 Schema 밖 토큰 차단
parse_tool_call과 주문 모듈: Tool별 필드 조합과 실제 주문 의미 검증
"""


def parse_free_reply(raw: str) -> str | None: # 자유응답용 파싱.
    # 자유응답을 explanation 한 필드로만 받는다.
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, dict) or set(parsed) != {"explanation"}: # 예외처리
        return None

    explanation = parsed["explanation"]

    if not isinstance(explanation, str) or not explanation.strip():
        return None

    return explanation.strip() # 문자열의 양 끝에 있는 공백이나 특정 문자를 제거


def parse_tool_call(raw: str) -> dict | None: # Tool용 파서
    # 모델 JSON을 읽고 Tool마다 허용한 필드와 기본 구조를 검사한다.
    # 메뉴값과 restriction 정본 검증은 각 V8 주문 모듈에서 한 번 더 수행한다.
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, dict) or set(parsed) != {"name", "changes"}:
        return None

    name = parsed["name"]
    changes = parsed["changes"]

    if name not in TOOL_CHANGE_FIELDS or not isinstance(changes, dict):
        return None

    if set(changes) - TOOL_CHANGE_FIELDS[name]:
        return None

    # 실제 변경이 없는 Tool은 changes가 반드시 빈 dictionary이다.
    if name in ("confirm_section", "cancel_order", "respond", "describe_scene"):
        return parsed if not changes else None

    # restriction은 여러 개가 들어올 수 있다.
    if "restriction_changes" in changes:
        restriction_changes = changes["restriction_changes"]

        if not isinstance(restriction_changes, list) or not restriction_changes:
            return None

        if len(restriction_changes) > len(RESTRICTION_TARGETS):
            return None

        for change in restriction_changes:
            if not isinstance(change, dict) or set(change) != {"target", "reason", "enabled"}:
                return None

            if not isinstance(change["target"], str) or not isinstance(change["reason"], str):
                return None

            if type(change["enabled"]) is not bool:
                return None

        restriction_keys = [
            (change["target"], change["reason"], change["enabled"])
            for change in restriction_changes
        ]

        if len(restriction_keys) != len(set(restriction_keys)):
            return None

    if name in ("set_order", "set_order_and_confirm"):
        if not changes:
            return None

        for field in ("sauce", "noodle_type", "noodle_portion"):
            if field in changes and not isinstance(changes[field], str):
                return None

        if "toppings" in changes and not isinstance(changes["toppings"], dict):
            return None

        return parsed

    if name == "refuse_section":
        if set(changes) != {"reason", "items"}:
            return None

        reason = changes["reason"]
        items = changes["items"]

        if reason not in RESTRICTION_REASONS:
            return None

        if not isinstance(items, list) or not items:
            return None

        if any(not isinstance(item, str) or item not in ALL_TOPPINGS for item in items):
            return None

        if len(items) != len(set(items)):
            return None

        return parsed

    # recommend_order는 adapter가 추천값을 만들고 Python이 부분 검증한다.
    if name == "recommend_order":
        scope = changes.get("scope")

        if scope not in ("ask", "current", "remaining", "specified"):
            return None

        for field in ("sections", "targets", "excluded", "preferences"):
            if field not in changes:
                continue

            values = changes[field]

            if not isinstance(values, list):
                return None

            if any(not isinstance(value, str) for value in values):
                return None

            if len(values) != len(set(values)):
                return None

            if field in ("targets", "excluded") and any(value not in ALL_MENU for value in values):
                return None

            if field == "preferences" and any(value not in PREFERENCE_TAGS for value in values):
                return None

        if "different" in changes and type(changes["different"]) is not bool:
            return None

        if scope == "specified":
            sections = changes.get("sections")

            if not sections or any(section not in FOOD_SECTIONS for section in sections):
                return None

        elif "sections" in changes:
            return None

        if scope == "ask":
            if "recommended_order" in changes:
                return None

            return parsed

        recommended_order = changes.get("recommended_order")

        if not isinstance(recommended_order, dict) or not recommended_order:
            return None

        if set(recommended_order) - {"sauce", "noodle_type", "noodle_portion", "toppings"}:
            return None

        if "toppings" in recommended_order and not isinstance(recommended_order["toppings"], dict):
            return None

        return parsed

    return None


def build_tool_messages(state: dict, user_text: str) -> list[dict]:
    # runtime·dataset builder·evaluator가 공통으로 사용할 모델 입력을 만든다.
    # state는 읽기만 하고 system + 최근 대화 + 현재 상태 JSON을 반환한다.
    if not isinstance(state, dict) or not isinstance(user_text, str):
        raise TypeError("state는 dict, user_text는 str이어야 함")

    order = state.get("order")
    execution = state.get("execution")

    if not isinstance(order, dict) or not isinstance(execution, dict):
        raise ValueError("state에 order와 execution dictionary가 필요함")

    section = execution.get("section")
    section_rule = section_rule_for(section)

    selected = execution.get("selected", [])
    current_selected = selected_keys_for_section(section, selected)

    skipped_sections = execution.get("skipped_sections", {})

    if not isinstance(skipped_sections, dict):
        raise ValueError("execution.skipped_sections는 dict여야 함")

    history = state.get("history", [])
    action_history = state.get("action_history", [])

    if not isinstance(history, list) or not isinstance(action_history, list):
        raise ValueError("history와 action_history는 list여야 함")

    current_state = {
        # order 전체와 현재 section에 필요한 진행값만 넣어서 과거 대화보다 현재 상태를 우선함
        "order": copy.deepcopy(order),
        "section": section,
        "section_rule": section_rule,
        "missing": missing_requirements(order, [section]).get(section, []),
        "selected": current_selected,
        "skipped_reason": copy.deepcopy(skipped_sections.get(section)),
        "recommendation": copy.deepcopy(state.get("recommendation", {})),
        "action_history": copy.deepcopy(action_history[-HISTORY_TURNS:]),
        "message": user_text,
    } # rule 주기

    user_message = {
        "role": "user",
        "content": json.dumps(
            current_state,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }

    return [
        {
            "role": "system",
            "content": TOOL_SYSTEM,
        },
        *copy.deepcopy(history[-(HISTORY_TURNS * 2):]),
        user_message,
    ]


def build_tool_inputs(state: dict, user_text: str, processor):
    # token 예산을 넘으면 오래된 대화부터 제거한다.
    # 원본 state를 자르면 다음 턴 기억까지 사라지므로 복사본만 줄인다.
    trimmed_state = copy.deepcopy(state)

    history = trimmed_state.get("history", [])
    action_history = trimmed_state.get("action_history", [])

    trimmed_state["history"] = history[-(HISTORY_TURNS * 2):]
    trimmed_state["action_history"] = action_history[-HISTORY_TURNS:]

    while True:
        messages = build_tool_messages(trimmed_state, user_text)

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        prompt_length = inputs["input_ids"].shape[1]

        if prompt_length <= TOOL_INPUT_MAX_TOKENS:
            return messages, inputs

        # user와 assistant 한 쌍을 오래된 쪽부터 제거한다.
        if trimmed_state["history"]:
            remove_count = 2 if len(trimmed_state["history"]) >= 2 else 1
            trimmed_state["history"] = trimmed_state["history"][remove_count:]
            continue

        # 자연어 대화가 모두 빠진 뒤에만 오래된 처리 기록을 제거한다.
        if trimmed_state["action_history"]:
            trimmed_state["action_history"] = trimmed_state["action_history"][1:]
            continue

        raise ValueError(
            f"고정 Tool prompt가 입력 한도를 초과함 : "
            f"{prompt_length}/{TOOL_INPUT_MAX_TOKENS}"
        )


def make_call_model(model, processor, logger=None):
    # XGrammar는 모델 로드 후 한 번만 컴파일한다.
    # 반환하는 call_model은 한 턴을 ToolCall로 바꾸며 주문 상태는 직접 수정하지 않는다.
    tokenizer = processor.tokenizer

    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<turn|>"),
    ]
    stop_ids = list(dict.fromkeys(
        token_id
        for token_id in stop_ids
        if isinstance(token_id, int) and token_id >= 0
    ))

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=len(tokenizer),
        stop_token_ids=stop_ids,
    )

    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_json_schema(TOOL_CALL_SCHEMA)

    @torch.inference_mode()
    def call_model(state: dict, user_text: str) -> dict | None:
        # token 예산을 적용한 공용 입력 builder를 runtime에서도 사용한다.
        # parser가 거부하면 None을 반환하고 agent가 error transaction으로 처리한다.
        messages, inputs = build_tool_inputs(state, user_text, processor)
        inputs = inputs.to(model.device)

        prompt_length = inputs["input_ids"].shape[1]

        if logger is not None:
            prompt_state = json.loads(messages[-1]["content"])
            source_history_count = min(
                len(state.get("history", [])),
                HISTORY_TURNS * 2,
            )
            kept_history_count = max(len(messages) - 2, 0)

            logger.info(
                f"Tool prompt tokens : {prompt_length}/{TOOL_INPUT_MAX_TOKENS}, "
                f"history : {source_history_count}->{kept_history_count}, "
                f"action_history : {len(prompt_state['action_history'])}"
            )

        # 호출마다 processor를 새로 만들어 이전 생성 상태를 공유하지 않는다.
        json_processor = XGrammarLogitsProcessor(compiled_grammar)

        started_at = time.perf_counter()

        output = model.generate(
            **inputs,
            max_new_tokens=TOOL_MAX_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=stop_ids,
            logits_processor=[json_processor],
        )

        raw = processor.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()

        parsed = parse_tool_call(raw)

        call_model.last_trace = {
            "raw_output": raw,
            "parsed_output": copy.deepcopy(parsed),
            "prompt_tokens": prompt_length,
            "generated_tokens": int(output[0].shape[0] - prompt_length),
            "latency_ms": round(
                (time.perf_counter() - started_at) * 1000,
                3,
            ),
        }

        if parsed is None:
            if logger is not None:
                logger.warning(f"Tool parser 거부 : {raw}")

            return None

        if logger is not None:
            logger.info(f"Tool 출력 : {raw}")

        return {
            "call_id": uuid.uuid4().hex[:8],
            "name": parsed["name"],
            "changes": parsed["changes"],
        }

    call_model.last_trace = None

    return call_model


def make_generate_reply(model, processor):
    # Tool adapter는 JSON 해석 전용이다.
    # 자연어 답변에서는 adapter를 끄고 같은 base model만 사용한다.
    tokenizer = processor.tokenizer

    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(token_id for token_id in stop_ids if isinstance(token_id, int) and token_id >= 0))

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=len(tokenizer),
        stop_token_ids=stop_ids,
    ) # 모델의 어휘 정보를 캡슐화

    grammar_compiler = xgr.GrammarCompiler(tokenizer_info) # 모델에 대한 문법을 ​​컴파일. 모델당 하나의 영구 컴파일러를 유지하여 컴파일 캐시가 요청 간에 공유되도록함.
    compiled_grammar = grammar_compiler.compile_json_schema(FREE_REPLY_SCHEMA) 
    # 컴파일 결과. 동일한 문법을 ​​사용하는 요청은 하나의 컴파일된 문법을 공유할 수 있다.



    @torch.inference_mode()
    def generate_reply(state: dict, user_text: str) -> str:
        if not isinstance(state, dict) or not isinstance(user_text, str):
            raise TypeError("state는 dict, user_text는 str이어야 함")

        execution = state.get("execution", {})

        facts = {
            "order": copy.deepcopy(state.get("order", {})),
            "section": execution.get("section"),
            "selected": copy.deepcopy(execution.get("selected", [])),
            "active_task": copy.deepcopy(execution.get("active_task")),
            "task_queue": copy.deepcopy(execution.get("task_queue", [])),
            "completed_tasks": copy.deepcopy(execution.get("completed_tasks", [])),
            "message": user_text,
        }

        messages = [
            {
                "role": "system",
                "content": FREE_REPLY_SYSTEM,
            },
            *copy.deepcopy(state.get("history", [])[-(HISTORY_TURNS * 2):]),
            {
                "role": "user",
                "content": json.dumps(
                    facts,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
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
        json_processor = XGrammarLogitsProcessor(compiled_grammar)
        started_at = time.perf_counter()

        if isinstance(model, PeftModel):
            with model.disable_adapter():
                output = model.generate(
                    **inputs,
                    max_new_tokens=TOOL_MAX_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=stop_ids,
                    logits_processor=[json_processor],
                )
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=TOOL_MAX_TOKENS,
                do_sample=False,
                use_cache=True,
                eos_token_id=stop_ids,
                logits_processor=[json_processor],
            )

        raw = processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()

        reply = parse_free_reply(raw)

        generate_reply.last_trace = {
            "raw_output": raw,
            "parsed_reply": reply,
            "prompt_tokens": prompt_length,
            "generated_tokens": int(output[0].shape[0] - prompt_length),
            "latency_ms": round(
                (time.perf_counter() - started_at) * 1000,
                3,
            ),
        }

        return reply or "질문을 정확히 이해하지 못했어요."

    generate_reply.last_trace = None

    return generate_reply

def make_call_vlm(model, processor, logger=None):
    # 같은 base model을 사용하지만 Tool LoRA는 끄고 이미지 판정만 수행한다.
    # 입력: PIL 이미지 목록, VLM system prompt, 사용자 질문
    # 상태 변경: 없음
    # 반환: 모델이 생성한 VLM 자연어 판정
    tokenizer = processor.tokenizer

    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<turn|>"),
    ]
    stop_ids = list(dict.fromkeys(
        token_id
        for token_id in stop_ids
        if isinstance(token_id, int) and token_id >= 0
    ))

    @torch.inference_mode()
    def call_vlm(pil_images: list, system_prompt: str, user_text: str) -> str:
        if not isinstance(pil_images, list) or not pil_images:
            raise ValueError("VLM 입력 이미지가 없음")

        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("VLM system prompt가 없음")

        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("VLM 사용자 질문이 없음")

        content = []

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

        prompt_length = inputs["input_ids"].shape[1]

        if logger is not None:
            logger.info(
                f"VLM prompt tokens : {prompt_length}, "
                f"images : {len(pil_images)}"
            )

        generate_args = {
            **inputs,
            "max_new_tokens": VLM_MAX_TOKENS,
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": stop_ids,
        }

        # Tool adapter는 JSON 출력용이므로 이미지 판정에서는 사용하지 않는다.
        # INT8 language model과 BF16 vision tower가 함께 계산될 때 남은 FP32 연산도
        # BF16 입력에 맞춰 실행해야 Float/BFloat16 dtype 충돌이 나지 않는다.
        autocast_device = model.device.type

        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=autocast_device == "cuda",
        ):
            if isinstance(model, PeftModel):
                with model.disable_adapter():
                    output = model.generate(**generate_args)
            else:
                output = model.generate(**generate_args)

        reply = processor.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()

        if reply.startswith("thought"):
            reply = reply[len("thought"):].lstrip()

        if logger is not None:
            logger.info(f"VLM 출력 : {reply}")

        return reply

    return call_vlm


def load_model(tool_adapter_path=None):
    # Tool·Reply·VLM이 공유할 processor와 base model을 한 번만 로드한다.
    # 입력: Tool LoRA 또는 QLoRA adapter 경로. None이면 base model만 사용한다.
    # 상태 변경: GPU에 model을 로드한다.
    # 반환: model, processor
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if MODEL_QUANTIZATION == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        model = _AutoVLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            local_files_only=True,
        )

    elif MODEL_QUANTIZATION == "nf4":
        # QLoRA에서 일반적으로 사용하는 4bit NF4 형식이다.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

        model = _AutoVLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            local_files_only=True,
        )

    elif MODEL_QUANTIZATION == "bf16":
        model = _AutoVLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            local_files_only=True,
        )

    else:
        raise ValueError(
            "MODEL_QUANTIZATION은 int8, nf4, bf16 중 하나여야 함 : "
            f"{MODEL_QUANTIZATION}"
        )

    model.eval()

    if tool_adapter_path is not None:
        # LoRA와 QLoRA 모두 추론 시에는 PEFT adapter로 base model 위에 연결한다.
        model = PeftModel.from_pretrained(
            model,
            tool_adapter_path,
            is_trainable=False,
        )
        model.eval()

    return model, processor
