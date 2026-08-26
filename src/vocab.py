from llm_sdk import Small_LLM_Model
import json

# Opens vocabulary file and maps values into a list
def build_vocab(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw: dict[str, int] = json.load(f)
    except Exception as err:
        print(f"Error occured opening vocab file. {err}")

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


# Converts tokens into readable text
def token_to_text(token_id):
    model = Small_LLM_Model()
    id_to_raw = build_vocab(model.get_path_to_vocab_file())
    raw_token = id_to_raw[token_id]
    data = bytes(BYTE_DECODER[ch] for ch in raw_token)
    return data.decode("utf-8", errors="replace")


def main():
    pass


if __name__ == "__main__":
    main()