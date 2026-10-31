"""Approved static Q quantity alignment, not a model performance test."""
import ast
import copy
import json
from pathlib import Path

from soomac_irc import agent_prompts, prompt_examples
from soomac_irc.prompt_contrast_examples import extend_contrast_examples


def test_only_four_approved_static_output_fields_change():
    tree = ast.parse(Path(prompt_examples.__file__).read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(name, ast.Name) and name.id == "EXAMPLES"
                              for name in node.targets))
    previous = json.loads(ast.literal_eval(assignment.value.args[0]))
    extend_contrast_examples(previous)
    expected = copy.deepcopy(previous)
    examples = {item["id"]: item for item in expected["recommendation"]}
    for identifier, ingredient in (("R01", "양파"), ("R01", "소시지"),
                                   ("R02", "양파"), ("R06", "치즈")):
        assert examples[identifier]["output"]["proposal"]["toppings"][ingredient] != "normal"
        examples[identifier]["output"]["proposal"]["toppings"][ingredient] = "normal"
    assert prompt_examples.EXAMPLES == expected
    for group in ("decision", "task", "general", "mixed"):
        assert prompt_examples.EXAMPLES[group] == previous[group]


def test_item_specific_quantity_evidence_and_confirmed_amounts_preserved():
    examples = {item["id"]: item for item in prompt_examples.EXAMPLES["recommendation"]}
    for identifier in ("R01", "R02", "R06"):
        assert examples[identifier]["input"]["order"]["noodle_portion"] == "low"
    assert examples["R04"]["output"]["proposal"]["noodle_portion"] == "low"
    assert examples["R02"]["output"]["proposal"]["sauce"] is None
    rule = "그 항목의 양 근거가 없으면 새 제안량은 normal로 둔다."
    assert agent_prompts.RECOMMENDATION_SYSTEM.count(rule) == 1
    assert rule not in agent_prompts.DECISION_SYSTEM
    assert rule not in agent_prompts.TASK_RESPONSE_SYSTEM
    assert agent_prompts.RECOMMENDATION_SYSTEM.endswith(prompt_examples.render_examples("recommendation"))
