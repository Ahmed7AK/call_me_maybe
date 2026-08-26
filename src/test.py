from llm_sdk import Small_LLM_Model

qwen = Small_LLM_Model()
eos_id = qwen._tokenizer.eos_token_id
max_new_tokens = 200

prompt = input("prompt> ")
messages = [{"role": "user", "content": prompt}]
ids = qwen._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)["input_ids"]
prompt_len = len(ids)

for _ in range(max_new_tokens):
    logits = qwen.get_logits_from_input_ids(ids)
    next_id = max(range(len(logits)), key=lambda i: logits[i])
    if next_id == eos_id:
        break
    ids.append(next_id)

res = qwen.decode(ids[prompt_len:])
print(res)
