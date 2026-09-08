from functools import lru_cache
from typing import Sequence

import numpy as np

from .trie import get_trie
from .vocab import get_vocab_bytes, token_to_bytes

# Bytes that a number can contain
NUMBER_BYTES = set(b"0123456789.-")

# Bytes that are unsafe for strings
STRING_UNSAFE = set(b'"\\') | set(range(0x20))


class ClosedChoice:
    def __init__(self, options: Sequence[str], terminator: str = '"') -> None:
        if not options:
            raise ValueError("a closed choice needs at least one option")
        self.options: list[str] = list(options)
        self._targets: list[bytes] = [
            (option + terminator).encode("utf-8") for option in self.options
        ]
        self._emitted: bytes = b""
        self._alive: list[int] = list(range(len(self.options)))
        self._match: int | None = None

    @property
    def done(self) -> bool:
        return self._match is not None

    @property
    def emitted(self) -> bytes:
        return self._emitted

    def allowed(self) -> set[int]:
        trie = get_trie()
        allowed: set[int] = set()
        for index in self._alive:
            suffix = self._targets[index][len(self._emitted):]
            for token_id, _ in trie.prefixes_of(suffix):
                allowed.add(token_id)
        return allowed

    def advance(self, token_id: int) -> None:
        self._emitted += token_to_bytes(token_id)
        self._alive = [
            index for index in self._alive
            if self._targets[index].startswith(self._emitted)
        ]
        if not self._alive:
            raise ValueError(f"token {token_id} left no option reachable")
        for index in self._alive:
            if self._targets[index] == self._emitted:
                self._match = index
                break

    def value(self) -> str:
        if self._match is None:
            raise RuntimeError("closed choice slot is not finished")
        return self.options[self._match]


@lru_cache(maxsize=1)
def number_token_ids() -> set[int]:
    return set(
        token_id
        for token_id, data in enumerate(get_vocab_bytes())
        if data and all(byte in NUMBER_BYTES for byte in data)
    )


@lru_cache(maxsize=1)
def string_token_ids() -> set[int]:
    return set(
        token_id
        for token_id, data in enumerate(get_vocab_bytes())
        if data and not any(byte in STRING_UNSAFE for byte in data)
    )


def masked_argmax(logits: Sequence[float], allowed: set[int]) -> int:
    if not allowed:
        raise ValueError("no token satisfies the current constraint")
    scores = np.asarray(logits, dtype=np.float32)
    index = np.fromiter(allowed, dtype=np.int64, count=len(allowed))
    index = index[index < scores.size]
    if index.size == 0:
        raise ValueError("every permitted token is outside the logit range")
    return int(index[int(np.argmax(scores[index]))])
