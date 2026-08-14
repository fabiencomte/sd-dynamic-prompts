from __future__ import annotations

from unittest import mock

import pytest

from sd_dynamic_prompts.helpers import (
    generate_prompt_cross_product,
    generate_prompts,
    get_seeds,
    load_magicprompt_models,
    repeat_prompt_batches,
    sample_prompt_pairs,
)


def test_get_seeds_with_fixed_seed(processing):
    num_seeds = 10

    seeds, subseeds = get_seeds(processing, num_seeds, use_fixed_seed=True)
    assert seeds == [processing.seed] * num_seeds
    assert subseeds == [processing.subseed] * num_seeds

    processing.subseed_strength = 0.5

    seeds, subseeds = get_seeds(processing, num_seeds, use_fixed_seed=True)
    assert seeds == [processing.all_seeds[0]] * num_seeds
    assert subseeds == [processing.all_subseeds[0]] * num_seeds


def test_get_seeds_with_fixed_seed_batched_combinatorial(processing):
    num_seeds = 10
    combinatorial_batches = 3
    seeds, subseeds = get_seeds(
        processing,
        num_seeds,
        use_fixed_seed=True,
        is_combinatorial=True,
        combinatorial_batches=combinatorial_batches,
    )
    seed0 = processing.seed
    assert seeds == (
        [seed0] * 4
        + [seed0 + 1] * 4
        + [seed0 + 2] * 2
    )
    assert subseeds == [processing.subseed] * num_seeds

    processing.subseed_strength = 0.5

    seeds, subseeds = get_seeds(
        processing,
        num_seeds,
        use_fixed_seed=True,
        is_combinatorial=True,
        combinatorial_batches=combinatorial_batches,
    )
    seed0 = processing.all_seeds[0]
    assert seeds == (
        [seed0] * 4
        + [seed0 + 1] * 4
        + [seed0 + 2] * 2
    )
    assert subseeds == [processing.all_subseeds[0]] * num_seeds


def test_get_seeds_with_random_seed(processing):
    num_seeds = 10

    seed, subseed = processing.seed, processing.subseed
    seeds, subseeds = get_seeds(processing, num_seeds=num_seeds, use_fixed_seed=False)
    assert seeds == list(range(seed, seed + num_seeds))
    assert subseeds == list(range(subseed, subseed + num_seeds))

    processing.subseed_strength = 0.5

    seed, subseed = processing.all_seeds[0], processing.all_subseeds[0]
    seeds, subseeds = get_seeds(processing, num_seeds=num_seeds, use_fixed_seed=False)
    assert seeds == [seed] * num_seeds
    assert subseeds == list(range(subseed, subseed + num_seeds))


def test_load_magicprompt_models(tmp_path):
    s = """# a comment
model1 # another comment
# empty lines below


model 2


    """
    p = tmp_path / "magicprompt_models.txt"
    p.write_text(s)
    assert load_magicprompt_models(p) == ["model1", "model 2"]


def test_cross_product():
    prompts = []
    negative_prompts = []
    expected_output = [], []
    assert generate_prompt_cross_product(prompts, negative_prompts) == expected_output

    prompts = ["A", "B", "C"]
    negative_prompts = ["X", "Y"]
    expected_output = (["A", "A", "B", "B", "C", "C"], ["X", "Y", "X", "Y", "X", "Y"])
    assert generate_prompt_cross_product(prompts, negative_prompts) == expected_output


def test_repeat_prompt_batches_caps_aligned_pairs():
    prompts, negative_prompts = repeat_prompt_batches(
        ["A", "B", "C"],
        ["X", "Y", "Z"],
        batches=2,
        max_prompts=5,
    )
    assert prompts == ["A", "B", "C", "A", "B"]
    assert negative_prompts == ["X", "Y", "Z", "X", "Y"]


def test_sample_prompt_pairs_is_unique_seeded_and_does_not_materialize_product():
    prompts = ["A", "B", "A"]
    negative_prompts = ["X", "Y", "Z", "X"]

    first = sample_prompt_pairs(prompts, negative_prompts, max_pairs=5, seed=123)
    repeated = sample_prompt_pairs(prompts, negative_prompts, max_pairs=5, seed=123)
    changed = sample_prompt_pairs(prompts, negative_prompts, max_pairs=5, seed=456)

    assert first == repeated
    assert first != changed
    assert len(first[0]) == len(first[1]) == 5
    assert len(set(zip(*first))) == 5
    assert set(zip(*first)) <= {
        (positive, negative)
        for positive in {"A", "B"}
        for negative in {"X", "Y", "Z"}
    }


def test_sample_prompt_pairs_returns_every_available_pair_below_limit():
    prompts, negative_prompts = sample_prompt_pairs(
        ["A", "B"],
        ["X", "Y"],
        max_pairs=10,
        seed=123,
    )

    assert list(zip(prompts, negative_prompts)) == [
        ("A", "X"),
        ("A", "Y"),
        ("B", "X"),
        ("B", "Y"),
    ]


@pytest.mark.parametrize("num_prompts", [5, None])
def test_generate_with_num_prompts(num_prompts: int | None):
    prompt_generator = mock.Mock()
    negative_prompt_generator = mock.Mock()
    prompt_generator.generate.return_value = [
        "Positive Prompt 1",
        "Positive Prompt 2",
        "Positive Prompt 3",
        "Positive Prompt 4",
        "Positive Prompt 5",
    ]
    negative_prompt_generator.generate.return_value = [
        "Negative Prompt 1",
        "Negative Prompt 2",
    ]
    prompt = "This is a positive prompt."
    negative_prompt = "This is a negative prompt."
    seeds = [1, 2, 3, 4, 5]

    positive_prompts, negative_prompts = generate_prompts(
        prompt_generator,
        negative_prompt_generator,
        prompt,
        negative_prompt,
        num_prompts,
        seeds,
    )
    assert isinstance(positive_prompts, list)
    assert isinstance(negative_prompts, list)

    if num_prompts:
        assert positive_prompts == [
            "Positive Prompt 1",
            "Positive Prompt 2",
            "Positive Prompt 3",
            "Positive Prompt 4",
            "Positive Prompt 5",
        ]
        assert negative_prompts == [
            "Negative Prompt 1",
            "Negative Prompt 2",
            "Negative Prompt 1",
            "Negative Prompt 2",
            "Negative Prompt 1",
        ]
    else:
        assert positive_prompts == [
            "Positive Prompt 1",
            "Positive Prompt 1",
            "Positive Prompt 2",
            "Positive Prompt 2",
            "Positive Prompt 3",
            "Positive Prompt 3",
            "Positive Prompt 4",
            "Positive Prompt 4",
            "Positive Prompt 5",
            "Positive Prompt 5",
        ]
        assert negative_prompts == [
            "Negative Prompt 1",
            "Negative Prompt 2",
            "Negative Prompt 1",
            "Negative Prompt 2",
            "Negative Prompt 1",
            "Negative Prompt 2",
            "Negative Prompt 1",
            "Negative Prompt 2",
            "Negative Prompt 1",
            "Negative Prompt 2",
        ]


def test_generate_aligns_negatives_with_short_positive_result():
    prompt_generator = mock.Mock()
    negative_prompt_generator = mock.Mock()
    prompt_generator.generate.return_value = ["Positive 1", "Positive 2"]
    negative_prompt_generator.generate.return_value = ["Negative"]

    positive_prompts, negative_prompts = generate_prompts(
        prompt_generator,
        negative_prompt_generator,
        "positive",
        "negative",
        5,
        None,
    )

    assert positive_prompts == ["Positive 1", "Positive 2"]
    assert negative_prompts == ["Negative", "Negative"]
