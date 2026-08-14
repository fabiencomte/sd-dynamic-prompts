from unittest import mock

from sd_dynamic_prompts.frozenprompt_generator import FrozenPromptGenerator


def test_repeats_correctly():
    source_generator = mock.Mock()
    source_generator.generate.side_effect = [["A"], ["B"]]
    generator = FrozenPromptGenerator(source_generator)
    template = "{A|B|C|D|E|F|G|H|I|J|K}"
    prompts = generator.generate(template, 40)

    assert len(prompts) == 40
    assert len(set(prompts)) == 1

    prompts2 = generator.generate(template, 40)

    assert len(prompts2) == 40
    assert len(set(prompts2)) == 1
    assert prompts[0] != prompts2[0]
