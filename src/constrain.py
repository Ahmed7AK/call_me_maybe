from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .trie import get_trie
from .vocab import get_vocab_bytes, token_to_bytes

# Bytes that a number can contain
NUMBER_BYTES = set(b"0123456789.-")

# Bytes that an integer can contain
INTEGER_BYTES = set(b"0123456789-")

# Control bytes are never allowed inside a JSON string
CONTROL_BYTES = set(range(0x20))

# Bytes allowed right after a backslash (\uXXXX is left out)
ESCAPE_BYTES = set(b'"\\/bfnrt')

BACKSLASH = ord("\\")
QUOTE = ord('"')


class ClosedChoice(BaseModel):
    """Limits output to any fixed list of strings"""
    model_config = ConfigDict(extra="forbid")

    options: list[str] = Field(min_length=1)
    terminator: str = '"'
    emitted: bytes = b""
    match: int | None = None
    _targets: list[bytes] = PrivateAttr(default_factory=list)
    _alive: list[int] = PrivateAttr(default_factory=list)

    def model_post_init(self, context: Any) -> None:
        """Encodes every option with its terminator and marks all alive"""
        self._targets = [
            (option + self.terminator).encode("utf-8")
            for option in self.options
        ]
        self._alive = list(range(len(self.options)))

    def allowed(self) -> set[int]:
        """
        Keeps track of tokens that are allowed as long as
        it adheres to possible options
        """
        trie = get_trie()
        allowed: set[int] = set()
        for index in self._alive:
            suffix = self._targets[index][len(self.emitted):]
            for token_id, _ in trie.prefixes_of(suffix):
                allowed.add(token_id)
        return allowed

    def advance(self, token_id: int) -> None:
        """Adds one token's bytes and drops options that don't fit"""
        self.emitted += token_to_bytes(token_id)
        self._alive = [
            index for index in self._alive
            if self._targets[index].startswith(self.emitted)
        ]
        if not self._alive:
            raise ValueError(f"token {token_id} left no option reachable")
        for index in self._alive:
            if self._targets[index] == self.emitted:
                self.match = index
                break

    def value(self) -> str:
        """Returns the chosen option"""
        if self.match is None:
            raise RuntimeError("closed choice slot is not finished")
        return self.options[self.match]


@lru_cache(maxsize=1)
def number_token_ids() -> set[int]:
    """Returns token_ids for valid number patterns"""
    return set(
        token_id
        for token_id, data in enumerate(get_vocab_bytes())
        if data and all(byte in NUMBER_BYTES for byte in data)
    )


@lru_cache(maxsize=1)
def integer_token_ids() -> set[int]:
    """Returns token_ids for valid integer patterns"""
    return set(
        token_id
        for token_id, data in enumerate(get_vocab_bytes())
        if data and all(byte in INTEGER_BYTES for byte in data)
    )


@lru_cache(maxsize=2)
def number_opening_ids(integer: bool) -> set[int]:
    """
    Returns token_ids that may open a number value right after the
    colon: a plain number token, or one led by a single space (" -")
    """
    digits = INTEGER_BYTES if integer else NUMBER_BYTES
    return set(
        token_id
        for token_id, data in enumerate(get_vocab_bytes())
        if data
        and all(byte in digits for byte in data.removeprefix(b" "))
    )


def _scan_string_token(
    data: bytes, escaped: bool
) -> tuple[int | None, bool] | None:
    """
    Handles string bytes by scanning for valid/invalid data
    """
    for index, byte in enumerate(data):
        if byte in CONTROL_BYTES:
            return None
        if escaped:
            if byte not in ESCAPE_BYTES:
                return None
            escaped = False
        elif byte == BACKSLASH:
            escaped = True
        elif byte == QUOTE:
            return index, False
    return None, escaped


@lru_cache(maxsize=2)
def string_tokens(
    escaped: bool,
) -> tuple[dict[int, bool], dict[int, bytes], set[int]]:
    """
    Returns valid string tokens while taking care of backslashes
    """
    content: dict[int, bool] = {}
    closing: dict[int, bytes] = {}
    for token_id, data in enumerate(get_vocab_bytes()):
        if not data:
            continue
        scan = _scan_string_token(data, escaped)
        if scan is None:
            continue
        quote_index, ends_escaped = scan
        if quote_index is None:
            content[token_id] = ends_escaped
        else:
            closing[token_id] = data[:quote_index]
    return content, closing, set(content) | set(closing)


@lru_cache(maxsize=1)
def string_opening_tokens() -> tuple[dict[int, bool], dict[int, bytes]]:
    """
    Returns tokens that may open a string value right after the colon:
    an optional space, the opening quote, then any valid content (e.g.
    ' "/' for a path). Maps each to the escape state it leaves behind,
    or to its content when the token also closes the string
    """
    content: dict[int, bool] = {}
    closing: dict[int, bytes] = {}
    for token_id, data in enumerate(get_vocab_bytes()):
        rest = data.removeprefix(b" ")
        if not rest.startswith(b'"'):
            continue
        scan = _scan_string_token(rest[1:], False)
        if scan is None:
            continue
        quote_index, ends_escaped = scan
        if quote_index is None:
            content[token_id] = ends_escaped
        else:
            closing[token_id] = rest[1:][:quote_index]
    return content, closing


def masked_argmax(logits: Sequence[float], allowed: set[int]) -> int:
    """
    Returns the allowed token with the highest logit value.
    Instead of setting disallowed values to -inf, they are
    ignored and filtered out from the possible options.
    """
    if not allowed:
        raise ValueError("no token satisfies the current constraint")
    scores = np.asarray(logits, dtype=np.float32)
    index = np.fromiter(allowed, dtype=np.int64, count=len(allowed))
    index = index[index < scores.size]
    if index.size == 0:
        raise ValueError("every permitted token is outside the logit range")
    return int(index[int(np.argmax(scores[index]))])
