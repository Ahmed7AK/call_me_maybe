import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .generate import build_system_prompt, generate_call
from .validation import parse_args, validate_answer, validate_json
from .vocab import get_model


def load_json(path: Path) -> Any:
    """Loads a JSON file from path"""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, results: list[dict[str, Any]]) -> None:
    """Writes into a JSON file the results"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


def call_me_maybe() -> int:
    """
    Loads the inputs and the model, generates a call for every prompt
    and writes the results.
    """
    args = parse_args()
    try:
        if not validate_json(args.functions_definition, args.input):
            return 1
        try:
            functions = load_json(args.functions_definition)
            inputs = load_json(args.input)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
            print(f"Could not read the input files: {err}", file=sys.stderr)
            return 1
        if not isinstance(functions, list) or not functions:
            print(
                f"{args.functions_definition} must be a non-empty JSON array.",
                file=sys.stderr,
            )
            return 1
        if not isinstance(inputs, list):
            print(f"{args.input} must be a JSON array.", file=sys.stderr)
            return 1

        try:
            model = get_model()
        except Exception as err:
            print(f"Could not load the model: {err}", file=sys.stderr)
            return 1

        system_prompt = build_system_prompt(functions)
        results: list[dict[str, Any]] = []
        try:
            for index, entry in enumerate(inputs, start=1):
                prompt = entry["prompt"]
                if not prompt.strip():
                    print(
                        f"Skipped prompt {index}: it is empty",
                        file=sys.stderr,
                    )
                    continue
                try:
                    call = generate_call(
                        model, system_prompt, functions, prompt
                    )
                    validate_answer(call)
                    results.append(call)
                except (ValueError, RuntimeError, ValidationError) as err:
                    print(f"Skipped {prompt!r}: {err}", file=sys.stderr)
                except Exception as err:
                    print(
                        f"Skipped {prompt!r}: unexpected "
                        f"{type(err).__name__}: {err}",
                        file=sys.stderr,
                    )
                print(f"[{index}/{len(inputs)}] {prompt}")
        except KeyboardInterrupt:
            print(
                "\nInterrupted, saving the results so far.", file=sys.stderr
            )

        try:
            write_json(args.output, results)
        except (OSError, TypeError, ValueError) as err:
            print(f"Could not write {args.output}: {err}", file=sys.stderr)
            return 1
        print(f"Wrote {len(results)}/{len(inputs)} calls to {args.output}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as err:
        print(
            f"Unexpected error: {type(err).__name__}: {err}", file=sys.stderr
        )
        return 1
