import pytest


def run_script(script, processing, **overrides):
    arguments = {
        "is_enabled": True,
        "is_combinatorial": True,
        "combinatorial_batches": 1,
        "is_magic_prompt": False,
        "is_feeling_lucky": False,
        "is_attention_grabber": False,
        "min_attention": 0,
        "max_attention": 1,
        "magic_prompt_length": 0,
        "magic_temp_value": 1,
        "use_fixed_seed": False,
        "unlink_seed_from_prompt": False,
        "disable_negative_prompt": False,
        "enable_jinja_templates": False,
        "no_image_generation": False,
        "max_generations": 0,
        "magic_model": "magic",
        "magic_blocklist_regex": None,
    }
    arguments.update(overrides)
    script.process(p=processing, **arguments)


@pytest.mark.parametrize("enable_hr", [True, False], ids=["yes_hr", "no_hr"])
@pytest.mark.parametrize("is_combinatorial", [True, False], ids=["yes_comb", "no_comb"])
def test_script(
    monkeypatch,
    monkeypatch_webui,
    processing,
    enable_hr,
    is_combinatorial,
):
    from scripts.dynamic_prompting import Script

    s = Script()
    if not is_combinatorial:
        processing.batch_size = 3
    processing.set_prompt_for_test("{red|green|blue} ball")
    processing.set_negative_prompt_for_test("ugly")
    processing.enable_hr = enable_hr
    s.process(
        p=processing,
        is_enabled=True,
        is_combinatorial=is_combinatorial,
        combinatorial_batches=1,
        is_magic_prompt=False,
        is_feeling_lucky=False,
        is_attention_grabber=False,
        min_attention=0,
        max_attention=1,
        magic_prompt_length=0,
        magic_temp_value=1,
        use_fixed_seed=False,
        unlink_seed_from_prompt=False,
        disable_negative_prompt=False,
        enable_jinja_templates=False,
        no_image_generation=False,
        max_generations=0,
        magic_model="magic",
        magic_blocklist_regex=None,
    )
    assert isinstance(processing.all_prompts, list)
    assert isinstance(processing.all_negative_prompts, list)
    assert isinstance(processing.all_hr_prompts, list)
    assert isinstance(processing.all_hr_negative_prompts, list)
    assert processing.main_prompt == processing.all_prompts[0]
    assert processing.main_negative_prompt == processing.all_negative_prompts[0]
    assert "{" not in processing.main_prompt

    if is_combinatorial:
        assert processing.all_prompts == ["red ball", "green ball", "blue ball"]
        assert processing.all_negative_prompts == ["ugly"] * 3

        if enable_hr:
            assert processing.all_hr_prompts == processing.all_prompts
            assert processing.all_hr_negative_prompts == processing.all_negative_prompts
    else:
        assert len(processing.all_prompts) == 3  # can't assert on the contents


def test_limited_combinatorial_is_random_batched_and_strictly_capped(
    monkeypatch_webui,
    processing,
):
    from scripts.dynamic_prompting import Script

    processing.batch_size = 2
    processing.set_prompt_for_test("{A|B|C|D|E|F}")
    processing.set_negative_prompt_for_test("bad")

    run_script(
        Script(),
        processing,
        combinatorial_batches=2,
        max_generations=5,
        use_fixed_seed=True,
    )

    assert len(processing.all_prompts) == 5
    assert len(set(processing.all_prompts[:3])) == 3
    assert processing.all_prompts[3:] == processing.all_prompts[:2]
    assert processing.all_negative_prompts == ["bad"] * 5
    assert processing.all_seeds == [1000, 1000, 1000, 1001, 1001]
    assert len(processing.all_subseeds) == 5
    assert processing.n_iter == 3


def test_unlimited_combinatorial_batches_do_not_square_cross_product(
    monkeypatch_webui,
    processing,
):
    from scripts.dynamic_prompting import Script

    processing.set_prompt_for_test("{A|B}")
    processing.set_negative_prompt_for_test("{X|Y}")
    run_script(Script(), processing, combinatorial_batches=2)

    assert len(processing.all_prompts) == 8
    assert processing.all_prompts[:4] == processing.all_prompts[4:]
    assert processing.all_negative_prompts[:4] == processing.all_negative_prompts[4:]


def test_no_image_generation_keeps_every_generation_list_aligned(
    monkeypatch_webui,
    processing,
):
    from scripts.dynamic_prompting import Script

    processing.n_iter = 4
    processing.batch_size = 2
    processing.enable_hr = True
    processing.set_prompt_for_test("{A|B|C}")
    processing.set_negative_prompt_for_test("bad")
    script = Script()

    run_script(
        script,
        processing,
        combinatorial_batches=2,
        no_image_generation=True,
    )

    assert processing.n_iter == processing.batch_size == 1
    assert len(processing.all_prompts) == 1
    assert len(processing.all_negative_prompts) == 1
    assert len(processing.all_seeds) == 1
    assert len(processing.all_subseeds) == 1
    assert len(processing.all_hr_prompts) == 1
    assert len(processing.all_hr_negative_prompts) == 1
    assert len(script._prompt_writer._positive_prompts) == 6


def test_precomputed_forge_job_count_tracks_updated_iterations(
    monkeypatch_webui,
    processing,
):
    from modules.shared import state

    from scripts.dynamic_prompting import Script

    processing.n_iter = 2
    processing.batch_size = 1
    processing.set_prompt_for_test("{A|B|C|D}")
    processing.set_negative_prompt_for_test("bad")
    state.job_count = 6

    run_script(Script(), processing)

    assert processing.n_iter == 4
    assert state.job_count == 12
