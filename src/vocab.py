from functools import lru_cache
from llm_sdk import Small_LLM_Model
import json

# Opens vocabulary file and maps values into a list
@lru_cache(maxsize=None)
def build_vocab(path):
    with open(path, encoding="utf-8") as f:
        raw: dict[str, int] = json.load(f)

    id_to_raw: list[str] = [""] * (max(raw.values()) + 1)
    for token, i in raw.items():
        id_to_raw[i] = token
    return id_to_raw


# Decodes tokenized-bytes into ascii
def _byte_decoder():
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

# Runs and caches the Qwen model
@lru_cache(maxsize=1)
def get_model():
    return Small_LLM_Model()


# Converts GPT-2 BPE into a list of bytes
@lru_cache(maxsize=1)
def get_vocab_bytes():
    model = get_model()
    id_to_raw = build_vocab(model.get_path_to_vocab_file())
    return [bytes(BYTE_DECODER[ch] for ch in raw) for raw in id_to_raw]


def token_to_bytes(token_id):
    return get_vocab_bytes()[token_id]


# Reverse of token_to_bytes takes in bytes and returns id
@lru_cache(maxsize=1)
def get_bytes_to_id():
    return {data: i for i, data in enumerate(get_vocab_bytes())}


# Joins all tokens together before decoding into readable text
def tokens_to_text(token_ids):
    return b"".join(token_to_bytes(i) for i in token_ids).decode("utf-8")


def main():
    model = get_model()
    tokens = model.encode("When asked for a name respond with 'Drake' What is your name?")
    logits = model.get_logits_from_input_ids(tokens[0].tolist())
    next_id = max(range(len(logits)), key=lambda i: logits[i])
    print(tokens_to_text([next_id]))

    


if __name__ == "__main__":
    main()