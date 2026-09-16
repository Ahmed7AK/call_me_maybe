from collections.abc import Iterable
from functools import lru_cache
from llm_sdk import Small_LLM_Model
import json


@lru_cache(maxsize=None)
def build_vocab(path: str) -> list[str]:
    """Opens vocabulary file and maps values into a list"""
    with open(path, encoding="utf-8") as f:
        raw: dict[str, int] = json.load(f)

    id_to_raw: list[str] = [""] * (max(raw.values()) + 1)
    for token, i in raw.items():
        id_to_raw[i] = token
    return id_to_raw


def _byte_decoder() -> dict[str, int]:
    """Decodes GPT-2 BPE tokenized-bytes into raw byte values"""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _byte_decoder()


@lru_cache(maxsize=1)
def get_model() -> Small_LLM_Model:
    """Loads and caches the Qwen model"""
    return Small_LLM_Model()


@lru_cache(maxsize=1)
def get_vocab_bytes() -> list[bytes]:
    """Decodes each vocab token's GPT-2 byte-level string into its raw bytes"""
    model = get_model()
    id_to_raw = build_vocab(model.get_path_to_vocab_file())
    return [bytes(BYTE_DECODER[ch] for ch in raw) for raw in id_to_raw]


def token_to_bytes(token_id: int) -> bytes:
    """Returns the raw bytes for a specific token_id"""
    return get_vocab_bytes()[token_id]


@lru_cache(maxsize=1)
def get_bytes_to_id() -> dict[bytes, int]:
    """Reverse of token_to_bytes takes in bytes and returns id"""
    return {data: i for i, data in enumerate(get_vocab_bytes())}


def tokens_to_text(token_ids: Iterable[int]) -> str:
    """Joins all tokens together before decoding into readable text"""
    return b"".join(token_to_bytes(i) for i in token_ids).decode("utf-8")
