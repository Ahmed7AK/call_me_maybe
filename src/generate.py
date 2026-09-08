import re
from typing import Any

import numpy as np
from llm_sdk import Small_LLM_Model

from .constrain import (
    ClosedChoice,
    masked_argmax,
    number_token_ids,
    string_token_ids,
)
from .vocab import get_bytes_to_id, token_to_bytes

# A slot that runs this long is stuck; close it and move on.
MAX_SLOT_TOKENS = 24

# Used to repair whatever the number charset filter let through.
NUMBER_PATTERN = re.compile(rb"-?\d+(?:\.\d+)?")


def build_system_prompt(functions: list[dict[str, Any]]) -> str:
    catalogue = "\n".join(
        "- {name}({params}): {description}".format(
            name=function["name"],
            params=", ".join(
                f"{key}: {spec['type']}"
                for key, spec in function["parameters"].items()
            ),
            description=function["description"],
        )
        for function in functions
    )
    return (
        "You are a function calling assistant.\n"
        "Choose the function that answers the request and take its "
        "arguments from the wording of the request.\n"
        "Do not answer the request yourself.\n"
        "Available functions:\n"
        f"{catalogue}"
    )


def build_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    """Wrap the two messages in Qwen3's chat template, thinking disabled."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _next_logits(model: Small_LLM_Model, prefix: bytes) -> list[float]:
    text = prefix.decode("utf-8", errors="ignore")
    return model.get_logits_from_input_ids(model.encode(text)[0].tolist())


def _fill_choice(
    model: Small_LLM_Model, prefix: bytes, slot: ClosedChoice
) -> bytes:
    for _ in range(MAX_SLOT_TOKENS):
        if slot.done:
            return prefix
        logits = _next_logits(model, prefix)
        token_id = masked_argmax(logits, slot.allowed())
        slot.advance(token_id)
        prefix += token_to_bytes(token_id)
    raise RuntimeError("closed choice did not settle within the token budget")


def _fill_string(model: Small_LLM_Model, prefix: bytes) -> tuple[str, bytes]:
    quote_id = get_bytes_to_id().get(b'"')
    allowed = set(string_token_ids())
    if quote_id is not None:
        allowed.add(quote_id)
    value = b""
    for _ in range(MAX_SLOT_TOKENS):
        logits = _next_logits(model, prefix)
        token_id = masked_argmax(logits, allowed)
        if token_id == quote_id:
            break
        data = token_to_bytes(token_id)
        value += data
        prefix += data
    return value.decode("utf-8", errors="ignore").strip(), prefix


def _fill_number(model: Small_LLM_Model, prefix: bytes) -> tuple[float, bytes]:
    allowed = set(number_token_ids())
    raw = b""
    for _ in range(MAX_SLOT_TOKENS):
        logits = _next_logits(model, prefix)
        if raw and int(np.argmax(logits)) not in allowed:
            break
        token_id = masked_argmax(logits, allowed)
        data = token_to_bytes(token_id)
        raw += data
        prefix += data
    match = NUMBER_PATTERN.search(raw)
    value = float(match.group()) if match else 0.0
    repaired = repr(value).encode("utf-8")
    return value, prefix[: len(prefix) - len(raw)] + repaired


def generate_call(
    model: Small_LLM_Model,
    system_prompt: str,
    functions: list[dict[str, Any]],
    user_prompt: str,
) -> dict[str, Any]:
    by_name = {function["name"]: function for function in functions}
    prefix = build_chat_prompt(system_prompt, user_prompt).encode("utf-8")
    prefix += b'{"name": "'

    name_slot = ClosedChoice(list(by_name), terminator='"')
    prefix = _fill_choice(model, prefix, name_slot)
    name = name_slot.value()

    prefix += b', "parameters": {'
    parameters: dict[str, Any] = {}
    declared = by_name[name]["parameters"].items()
    for position, (key, spec) in enumerate(declared):
        if position:
            prefix += b", "
        prefix += f'"{key}": '.encode("utf-8")
        kind = spec["type"]
        if kind == "string":
            prefix += b'"'
            value, prefix = _fill_string(model, prefix)
            prefix += b'"'
        elif kind == "number":
            value, prefix = _fill_number(model, prefix)
        else:
            flag = ClosedChoice(["true", "false"], terminator="")
            prefix = _fill_choice(model, prefix, flag)
            value = flag.value() == "true"
        parameters[key] = value

    return {"prompt": user_prompt, "name": name, "parameters": parameters}
