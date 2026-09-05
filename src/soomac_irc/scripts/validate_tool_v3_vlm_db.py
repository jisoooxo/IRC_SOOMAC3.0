#!/usr/bin/env python3

import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "soomac_irc"
sys.path.insert(0, str(PACKAGE_ROOT))

from call_model_v2 import load_model, make_call_model, make_call_vlm
from llm_node_v2 import LLMNode
from vlm_rag import VlmRag


ADAPTER_PATH = PROJECT_ROOT / "outputs" / "gemma4_tool_lora_v3"
TEST_PATH = PROJECT_ROOT / "finetune" / "v3" / "tool_v3_test.jsonl"
VALID_ROOT = PROJECT_ROOT / "vlm_data" / "mealkit_coco_v13" / "valid"

# 전체 272개 평가는 evaluate_tool_lora.py가 담당한다.
# 여기서는 실제 장애와 경계 사례를 빠르게 다시 확인한다.
TOOL_CASE_IDS = [
    "v3_stt_incomplete_001",
    "v3_single_choice_correction_002",
    "v3_compound_dropped_005",
    "v3_compound_blocked_002",
    "v3_completed_status_001",
    "v3_multiturn_reference_001",
    "v3_cancel_explicit_003",
    "v3_action_boundary_005",
    "v3_korean_style_refuse_003",
    "v3_recommend_ask_002",
    "v3_recommend_section_balance_003",
    "v3_stateful_recommend_002",
    "v3_stateful_recommend_008",
]

VLM_POSITIVE_CASES = [
    ("치즈", "cheese"),
    ("페퍼론치노", "pepperoncino"),
    ("크림", "sauce_cream"),
    ("오일", "sauce_oil"),
    ("토마토", "sauce_tomato"),
]

VLM_NEGATIVE_CASES = [
    ("치즈", "pepperoncino"),
    ("크림", "sauce_tomato"),
    ("양파", "mushroom"),
]


class VlmTestNode:
    _parse_vlm_predict = LLMNode._parse_vlm_predict


def load_tool_rows():
    rows = {}

    for line in TEST_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row

    missing = [case_id for case_id in TOOL_CASE_IDS if case_id not in rows]

    if missing:
        raise ValueError(f"Tool 검증 ID가 없음 : {missing}")

    return rows


def load_valid_crops(class_names):
    coco = json.loads((VALID_ROOT / "_annotations.coco.json").read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    images = {image["id"]: image["file_name"] for image in coco["images"]}
    crops = {}

    for annotation in coco["annotations"]:
        class_name = categories[annotation["category_id"]]

        if class_name not in class_names or class_name in crops:
            continue

        image_path = VALID_ROOT / images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]

        with Image.open(image_path) as image:
            crops[class_name] = image.convert("RGB").crop((x, y, x + width, y + height))

    missing = sorted(set(class_names) - set(crops))

    if missing:
        raise ValueError(f"VLM 검증 crop이 없음 : {missing}")

    return crops


def run_tool_cases(call_model, rows):
    passed = 0

    print("\n===== Tool v3 예외 사례 =====")

    for case_id in TOOL_CASE_IDS:
        row = rows[case_id]
        facts = json.loads(row["messages"][-2]["content"])
        expected = json.loads(row["messages"][-1]["content"])
        actual_call = call_model(row["messages"][:-1])
        actual = None

        if actual_call is not None:
            actual = {"name": actual_call["name"], "changes": actual_call["changes"]}

        success = actual == expected
        passed += int(success)
        print(f"[{case_id}] {'PASS' if success else 'FAIL'}")
        print("입력:", facts["message"])
        print("기대:", json.dumps(expected, ensure_ascii=False))
        print("실제:", json.dumps(actual, ensure_ascii=False))

    return passed, len(TOOL_CASE_IDS)


def run_vlm_cases(call_vlm, crops):
    node = VlmTestNode()
    node.call_vlm = call_vlm
    node.vlm_reference = {}
    node.vlm_rag = VlmRag()
    passed = 0
    total = 0

    print("\n===== v13 DB 이미지 VLM 사례 =====")

    for expected, camera_class in VLM_POSITIVE_CASES:
        predict, raw = LLMNode._judge_place(node, [crops[camera_class]], expected)
        success = predict == "pass"
        passed += int(success)
        total += 1
        print(f"[정재료] {expected} <- {camera_class}: {'PASS' if success else 'FAIL'} / parser={predict}")
        print("raw:", raw)

    for expected, camera_class in VLM_NEGATIVE_CASES:
        predict, raw = LLMNode._judge_place(node, [crops[camera_class]], expected)
        success = predict == "wrong_ingredient"
        passed += int(success)
        total += 1
        print(f"[오재료] {expected} <- {camera_class}: {'PASS' if success else 'FAIL'} / parser={predict}")
        print("raw:", raw)

    return passed, total


def main():
    try:
        rows = load_tool_rows()
        needed_classes = {class_name for _expected, class_name in VLM_POSITIVE_CASES + VLM_NEGATIVE_CASES}
        crops = load_valid_crops(needed_classes)

        print("Gemma4 + Tool LoRA v3 로딩")
        model, processor = load_model(tool_adapter_path=ADAPTER_PATH)
        call_model = make_call_model(model, processor)
        call_vlm = make_call_vlm(model, processor)

        tool_passed, tool_total = run_tool_cases(call_model, rows)
        vlm_passed, vlm_total = run_vlm_cases(call_vlm, crops)

        print("\n========================================")
        print(f"Tool 결과: {tool_passed}/{tool_total}")
        print(f"VLM 결과: {vlm_passed}/{vlm_total}")
        print(f"최종 결과: {tool_passed + vlm_passed}/{tool_total + vlm_total}")

        if tool_passed != tool_total or vlm_passed != vlm_total:
            print("FAIL: 실제 raw 결과를 확인하고 실패 사례를 holdout에 유지")
            return 1

        print("PASS: Tool LoRA v3 예외 사례 + v13 DB 이미지 VLM")
        return 0

    except Exception as exc:
        print(f"FAIL: 자동 검증 / {type(exc).__name__}: {exc} / 경로와 첫 실패 확인")
        return 1


if __name__ == "__main__":
    sys.exit(main())
