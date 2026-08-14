from dynamicprompts.wildcards import WildcardManager

from sd_dynamic_prompts.random_combinatorial_generator import (
    RandomCombinatorialPromptGenerator,
)


def make_generator(tmp_path, seed):
    return RandomCombinatorialPromptGenerator(
        WildcardManager(tmp_path),
        seed=seed,
    )


def test_random_subset_is_unique_and_seeded(tmp_path):
    template = "{A|B|C|D|E|F|G|H|I|J}"
    first = make_generator(tmp_path, 10).generate(template, 4)
    repeated = make_generator(tmp_path, 10).generate(template, 4)
    different_seed = make_generator(tmp_path, 11).generate(template, 4)

    assert len(first) == len(set(first)) == 4
    assert repeated == first
    assert different_seed != first
    assert first != ["A", "B", "C", "D"]


def test_random_subset_stops_when_prompt_space_is_exhausted(tmp_path):
    prompts = make_generator(tmp_path, 42).generate("{A|B}", 10)
    assert len(prompts) == 2
    assert set(prompts) == {"A", "B"}


def test_unlimited_generation_remains_exhaustive(tmp_path):
    prompts = make_generator(tmp_path, 42).generate("{A|B|C}", None)
    assert prompts == ["A", "B", "C"]
