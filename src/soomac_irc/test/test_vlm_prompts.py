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
    assert "한 개라도 새로 들어갔으면 PASS" in system_prompt
    assert "주문량에 맞는지는 판정하지 않는다" in system_prompt


@pytest.mark.parametrize("expected, phrase", [("버섯", "버섯이 아닌"), ("소시지", "소시지가 아닌")])
def test_korean_subject_particle_follows_the_final_consonant(expected, phrase):
    system_prompt, _ = build_ingredient_prompt(expected, 5, True, "pass")
    assert phrase in system_prompt


@pytest.mark.parametrize("expected", sorted(set(INGREDIENT_GUIDES) - {"소시지"}))
def test_insertion_only_rule_is_limited_to_sausage(expected):
    system_prompt, _ = build_ingredient_prompt(expected, 5, True, "pass")
    assert "주문량에 맞는지는 판정하지 않는다" not in system_prompt


def test_sausage_orientation_does_not_have_to_match_the_reference():
    system_prompt, _ = build_ingredient_prompt("소시지", 5, True, "pass")
    assert "비스듬하게" in system_prompt
    assert "소시지 개수를 각각 세어" in system_prompt


def test_cheese_requires_a_pile_and_names_look_alikes():
    system_prompt, _ = build_ingredient_prompt("치즈", 5, True, "pass")
    assert "겹쳐 쌓인 더미" in system_prompt
    assert "낱개 한두 조각이나 흩어진 부스러기만 있으면 FAIL" in system_prompt
    assert "빛 반사" in system_prompt and "양파" in system_prompt
    # 치즈 전용 기준이 일반 기준을 대체한다(한 조각만 보여도 PASS가 되는 문장 금지).
    assert "새로 보이거나 해당 재료 영역이 증가했으면 PASS" not in system_prompt


@pytest.mark.parametrize("expected", ["양파", "버섯", "게살", "페퍼론치노"])
def test_other_ingredients_keep_the_general_rule(expected):
    system_prompt, _ = build_ingredient_prompt(expected, 5, True, "pass")
    assert "새로 보이거나 해당 재료 영역이 증가했으면 PASS" in system_prompt
    assert system_prompt.count("근거에는") == 1
