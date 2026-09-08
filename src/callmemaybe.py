"""Entry point: read the input files, fill one call per prompt, write them."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .generate import build_system_prompt, generate_call
from .validation import parse_args, validate_answer, validate_json
from .vocab import get_model


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


def call_me_maybe() -> int:
    args = parse_args()
    if not validate_json(args.functions_definition, args.input):
        return 1
    try:
        functions = load_json(args.functions_definition)
        inputs = load_json(args.input)
    except (OSError, json.JSONDecodeError) as err:
        print(f"Could not read the input files: {err}")
        return 1
    if not functions:
        print(f"No functions defined in {args.functions_definition}.")
        return 1

    try:
        model = get_model()
    except Exception as err:
        print(f"Could not load the model: {err}")
        return 1

    system_prompt = build_system_prompt(functions)
    results: list[dict[str, Any]] = []
    for entry in inputs:
        prompt = entry["prompt"]
        try:
            call = generate_call(model, system_prompt, functions, prompt)
            validate_answer(call)
            results.append(call)
        except (ValueError, RuntimeError, ValidationError) as err:
            print(f"Skipped {prompt!r}: {err}")

    try:
        write_json(args.output, results)
    except OSError as err:
        print(f"Could not write {args.output}: {err}")
        return 1
    print(f"Wrote {len(results)}/{len(inputs)} calls to {args.output}")
    return 0
