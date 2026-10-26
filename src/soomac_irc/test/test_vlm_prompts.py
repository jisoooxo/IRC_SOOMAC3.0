import pytest

from soomac_irc.vlm_prompts import INGREDIENT_GUIDES, build_ingredient_prompt


@pytest.mark.parametrize("expected", sorted(INGREDIENT_GUIDES))
@pytest.mark.parametrize("has_comparison", [False, True])
def test_every_ingredient_prompt_names_its_compartment_and_appearance(expected, has_comparison):
    compartment, appearance = INGREDIENT_GUIDES[expected]
    system_prompt, _ = build_ingredient_prompt(expected, 5, has_comparison, "pass")
    assert compartment in system_prompt and appearance in system_prompt
    assert "공급 용기" in system_prompt


def test_pepperoncino_description_is_not_attached_to_other_ingredients():
    system_prompt, _ = build_ingredient_prompt("소시지", 5, True, "pass")
    assert "고추" not in system_prompt


def test_small_insertion_passes_without_judging_the_ordered_amount():
    system_prompt, _ = build_ingredient_prompt("소시지", 5, True, "pass")
    assert "한두 조각이라도 새로 들어갔으면 PASS" in system_prompt
    assert "주문량에 맞는지는 판정하지 않는다" in system_prompt


@pytest.mark.parametrize("expected, phrase", [("버섯", "버섯이 아닌"), ("소시지", "소시지가 아닌")])
def test_korean_subject_particle_follows_the_final_consonant(expected, phrase):
    system_prompt, _ = build_ingredient_prompt(expected, 5, True, "pass")
    assert phrase in system_prompt


@pytest.mark.parametrize("expected", sorted(set(INGREDIENT_GUIDES) - {"소시지"}))
def test_insertion_only_rule_is_limited_to_sausage(expected):
    system_prompt, _ = build_ingredient_prompt(expected, 5, True, "pass")
    assert "주문량에 맞는지는 판정하지 않는다" not in system_prompt
