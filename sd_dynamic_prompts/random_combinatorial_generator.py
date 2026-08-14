from __future__ import annotations

from dynamicprompts.generators import (
    CombinatorialPromptGenerator,
    PromptGenerator,
    RandomPromptGenerator,
)
from dynamicprompts.parser.parse import default_parser_config


class RandomCombinatorialPromptGenerator(PromptGenerator):
    """Select unique random combinations without enumerating the whole prompt space."""

    def __init__(
        self,
        wildcard_manager,
        *,
        seed: int | None = None,
        unlink_seed_from_prompt: bool = False,
        ignore_whitespace: bool = False,
        parser_config=default_parser_config,
    ) -> None:
        self._random_generator = RandomPromptGenerator(
            wildcard_manager,
            seed=seed,
            unlink_seed_from_prompt=unlink_seed_from_prompt,
            ignore_whitespace=ignore_whitespace,
            parser_config=parser_config,
        )
        self._combinatorial_generator = CombinatorialPromptGenerator(
            wildcard_manager,
            ignore_whitespace=ignore_whitespace,
            parser_config=parser_config,
        )

    def generate(
        self,
        template: str | None,
        max_prompts: int | None,
        **kwargs,
    ) -> list[str]:
        if max_prompts is None:
            return self._combinatorial_generator.generate(template, None)
        if max_prompts <= 0:
            return []
        if not template:
            return [""]

        prompts: list[str] = []
        seen: set[str] = set()
        stalled_rounds = 0

        # Prompt/image seeds describe the final generation list, not the random
        # search attempts. The generator itself was seeded in __init__.
        kwargs.pop("seeds", None)

        while len(prompts) < max_prompts and stalled_rounds < 4:
            remaining = max_prompts - len(prompts)
            draw_count = max(8, min(remaining * 2, 256))
            candidates = self._random_generator.generate(
                template,
                draw_count,
                **kwargs,
            )
            previous_count = len(prompts)
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    prompts.append(candidate)
                    if len(prompts) == max_prompts:
                        break
            stalled_rounds = stalled_rounds + 1 if len(prompts) == previous_count else 0

        # A small prompt space eventually produces collisions. Walking the
        # combinatorial generator fills any missing cases and also lets us stop
        # cleanly when fewer combinations exist than requested.
        if len(prompts) < max_prompts:
            for candidate in self._combinatorial_generator.generate(template, None):
                if candidate not in seen:
                    seen.add(candidate)
                    prompts.append(candidate)
                    if len(prompts) == max_prompts:
                        break

        return prompts
