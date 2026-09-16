*This project has been created as part of the 42 curriculum by akheiral*

# Call Me Maybe

## Description

Call Me Maybe turns natural language requests into structured function calls
using a small language model (Qwen3-0.6B). Given a request such as
*"What is the sum of 2 and 3?"*, the program does not answer it. It
returns the function to call and its typed arguments:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

Small models are unreliable at producing structured output when simply
asked to. This project uses **constrained decoding**: at each generation
step, every token that would break the JSON structure or the function
schema is ruled out before the next token is picked. The output is
therefore always valid JSON that matches the definitions in
`functions_definition.json`.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
make install        # runs: uv sync
```

The model weights are downloaded from Hugging Face on the first run.

### Running

```bash
make run            # runs: uv run python -m src
```

With custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

| Option | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

### Other Makefile rules

| Rule | Description |
|---|---|
| `make debug` | Runs the program under `pdb` |
| `make lint` | Runs `flake8` and `mypy` with the required flags |
| `make clean` | Removes `__pycache__`, `.pyc` files, tool caches and `data/output` |

## Example usage

Input (`function_calling_tests.json`):

```json
[
  {"prompt": "Greet shrek"},
  {"prompt": "Replace all vowels in 'Programming is fun' with asterisks"}
]
```

Run:

```bash
$ make run
[1/2] Greet shrek
[2/2] Replace all vowels in 'Programming is fun' with asterisks
Wrote 2/2 calls to data/output/function_calling_results.json
```

Output (`function_calling_results.json`):

```json
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  {
    "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Programming is fun",
      "regex": "([aeiouAEIOU])",
      "replacement": "*"
    }
  }
]
```

Errors are reported on stderr with a clear message, for example:

```bash
$ uv run python -m src --input missing.json
Trouble opening file - missing.json: [Errno 2] No such file or directory: 'missing.json'
```

## Algorithm explanation

### 1. Vocabulary and byte mapping

Qwen uses a GPT-2 style byte-level BPE tokenizer. Its vocabulary file
(`get_path_to_vocab_file()`) stores every token as a string where each raw
byte is replaced by a printable character. `src/vocab.py` reverses that
mapping, so every token id maps to the exact bytes it produces.

### 2. Prefix trie

`src/trie.py` builds a trie over those bytes: one node per byte, with a
token id stored on the node where a token ends. Given a target such as
`fn_greet"`, walking the trie returns every token that is a prefix of
the target in a single pass, instead of checking all ~150k tokens.

### 3. JSON skeleton

The program writes the fixed parts of the JSON itself and only lets the
model fill in the values:

```
{"name": "<name slot>", "parameters": {"<key>": <value slot>, ...}}
```

The keys come from the chosen function's definition, so they are always
exactly the declared ones, in order.

### 4. Filling the slots

At each step the full prefix (chat prompt + JSON so far) is encoded,
`get_logits_from_input_ids` returns the logits, and the highest-scoring
**allowed** token is picked (`masked_argmax`). Disallowed tokens are
simply never considered, which is equivalent to setting their logits to
negative infinity.

| Slot | Allowed tokens |
|---|---|
| Function name | Tokens that keep the text a prefix of `<function name>"` for at least one remaining function (`ClosedChoice` + trie). The slot ends when an option is fully matched. |
| `string` | Tokens that are valid JSON string content: no control characters, and every backslash is followed by a valid escape. A token such as `",` closes the string; only the part before the quote is kept. The value is then unescaped with `json.loads`. |
| `number` | Tokens made only of `0-9`, `.` and `-`. Generation stops when the model's own top choice is no longer a number token, and the result is cleaned with a regex (`1.2.3` becomes `1.2`) and written as a float. |
| `boolean` | A `ClosedChoice` between `true` and `false`. |

Every slot is limited to `MAX_SLOT_TOKENS` (64) tokens so generation
always ends.

### 5. Prompt

The request is wrapped in Qwen3's chat template with an empty
`<think></think>` block (thinking disabled). The system prompt lists
every function with its parameter types and description, tells the model
to write actual values rather than descriptions, and shows one worked
example with a made-up function.

### 6. Aliases

For functions that take a regex or pattern parameter, a value that is
exactly a descriptive word is replaced with what it names
(`src/aliases.py`): `numbers` becomes `\d+` in the regex parameter, and
`asterisks` becomes `*` in the other string parameters, unless the prompt
quotes that word.

## Design decisions

- **Skeleton instead of free JSON generation.** Writing the structure
  directly avoids having to constrain braces, commas and key names, and
  guarantees the keys match the schema.
- **The LLM chooses the function.** The function name is generated
  under a closed choice of the defined names, so the model's own
  preference decides, not a keyword heuristic.
- **Trie over the vocabulary.** Finding allowed tokens for a closed
  choice is fast even with a 150k-token vocabulary.
- **Cached token sets.** The allowed sets for numbers and strings are
  computed once (`lru_cache`) and reused for every step.
- **Stopping numbers on the model's preference.** A number ends when the
  unconstrained top token is not numeric, which lets the model decide
  where the number ends without allowing invalid characters.
- **Escape-aware strings.** Allowing valid JSON escapes lets arguments
  contain backslashes (regex patterns, Windows paths) while the output
  stays valid JSON.
- **Alias dictionary.** The 0.6B model often writes the name of a value
  ("asterisk") instead of the value (`*`). The dictionary is general,
  only applies to regex-style functions, only replaces whole values, and
  never influences which function is chosen.
- **Pydantic everywhere.** Input files, generated calls and the internal
  classes (`TrieNode`, `Trie`, `ClosedChoice`) are pydantic models.
- **Fail per prompt, not per run.** A prompt that fails (or is empty) is
  skipped with a message; the rest are still written. Invalid input files
  stop the program with a clear error and exit code 1.

## Performance analysis

Measured on an Apple M3 Pro with the provided input files:

| Metric | Result |
|---|---|
| Function selection | 11/11 |
| Fully correct calls (name and all arguments) | 11/11 |
| Valid JSON / schema-compliant output | 100% (guaranteed by construction) |
| Total run time, including model loading | ~30 s |

Known weak spots:

- Quotes inside a string argument: the model tends to end the string at
  the first `"` instead of writing `\"`.
- The model sometimes drops the sign of a negative number.
- Each step re-encodes the whole prefix (no key/value cache), so the time
  per token grows with the prompt length.

## Challenges faced

- **Token boundaries.** Tokens do not line up with JSON syntax: one token
  can contain the end of a string and the following `",`. Closing tokens
  are recognised by the quote inside them, and only the part before the
  quote is kept.
- **Byte-level vocabulary.** Tokens are stored as mapped characters, not
  raw bytes, so the GPT-2 byte mapping had to be reversed before any
  comparison was possible.
- **Knowing when a number ends.** A number has no closing character, so
  the model's unconstrained top choice is used to decide when to stop.
- **Regex arguments.** The model copied text from the prompt (`34`
  instead of `\d+`) and backslashes were originally banned. This was
  fixed with escape-aware string decoding, a clearer system prompt with a
  generic example, and the alias dictionary.
- **Empty prompts.** An empty request made the model copy the example
  from the system prompt, so empty prompts are now skipped.

## Testing strategy

- `make lint` (flake8 and mypy with the required flags) must pass.
- Full runs on the provided files, checking every name and argument by
  hand.
- Extra prompt sets for strings that need escapes (backslashes, quotes,
  `\d+`, `\s+`), symbol words ("hashes", "dashes"), quoted words that
  must not be replaced, unicode, large and negative numbers, and empty
  prompts.
- Broken inputs: missing file, empty file, invalid JSON, an object
  instead of an array, wrong value types, unsupported parameter types, a
  function without parameters, an empty prompt list, and an output path
  that cannot be written.
- Prompt variants were compared on the full test set before a system
  prompt was chosen, to avoid fixing one prompt while breaking another.

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face: Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
- [OpenAI GPT-2 byte-level encoder (`encoder.py`)](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023)](https://arxiv.org/abs/2307.09702)
- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

### AI usage

AI (Claude) was used as an assistant during development:

- Reviewing the project against the subject and listing what was missing.
- Fixing flake8 and mypy errors, and turning comments into docstrings.
- Converting `TrieNode`, `Trie` and `ClosedChoice` to pydantic models.
- Adding exception handling to `src/callmemaybe.py` and improving the
  Makefile `clean` rule and `.gitignore`.
- Investigating the wrong regex arguments: comparing system prompt
  variants, adding escape-aware string decoding and the alias dictionary.
- Testing edge cases and drafting this README.

All generated code was reviewed, tested and adjusted before being kept.
