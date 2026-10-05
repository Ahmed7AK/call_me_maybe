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
    --output data/output/function_calls.json
```

| Option | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calls.json` |

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
Wrote 2/2 calls to data/output/function_calls.json
```

Output (`function_calls.json`):

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
      "regex": "[aeiou]",
      "replacement": "*"
    }
  }
]
```

Errors are reported with a clear message and exit code 1, for example:

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
`fn_greet",`, walking the trie returns every token that is a prefix of
the target in a single pass, instead of checking all ~150k tokens.

### 3. JSON skeleton

The program writes the fixed parts of the JSON itself and only lets the
model fill in the values:

```
{"name": "<name slot>", "parameters": {"<key>":<value slot>, ...}}
```

The keys come from the chosen function's definition, so they are always
exactly the declared ones, in order.

The skeleton stops right after each colon, without the usual space, and
the name slot ends with `",` rather than a lone `"`. The model's
tokenizer merges these characters with their neighbours (` -`, ` "/`,
`",`), so cutting the text in the middle of such a token leaves the
model at a boundary it never saw in training and pushes it toward the
wrong continuation: negative numbers lost their sign, paths lost their
leading `/`, and `fn_get` lost to `fn_get_user`. Letting the model's
first token carry the space and opening quote (a technique known as
token healing) avoids this.

### 4. Filling the slots

At each step the full prefix (chat prompt + JSON so far) is encoded,
`get_logits_from_input_ids` returns the logits, and the highest-scoring
**allowed** token is picked (`masked_argmax`). Disallowed tokens are
simply never considered, which is equivalent to setting their logits to
negative infinity.

| Slot | Allowed tokens |
|---|---|
| Function name | Tokens that keep the text a prefix of `<function name>",` for at least one remaining function (`ClosedChoice` + trie). The slot ends when an option is fully matched. |
| `string` | First token: an optional space, the opening quote, then any valid content (` "`, ` "/`, ` "\\`). After that, tokens that are valid JSON string content: no control characters, and every backslash is followed by a valid escape. A token such as `",` closes the string; only the part before the quote is kept. The value is then unescaped with `json.loads`. |
| `number` | First token: a number token, optionally led by a space (` -`, ` `). After that, tokens made only of `0-9`, `.` and `-`. Generation stops when the model's own top choice is no longer a number token, and the result is cleaned with a regex (`1.2.3` becomes `1.2`) and written as a float. |
| `integer` | Same as `number`, but `.` is never allowed, so the model cannot start a decimal. The result is cleaned with a regex (`1-2` becomes `1`) and written as an int. |
| `boolean` | A space, then a `ClosedChoice` between `true` and `false`. |

Every slot is limited to `MAX_SLOT_TOKENS` (64) tokens so generation
always ends.

### 5. Prompt

The request is wrapped in Qwen3's chat template with an empty
`<think></think>` block (thinking disabled). The system prompt lists
every function with its parameter types and description, tells the model
to write actual values rather than descriptions, and shows two worked
examples with a made-up function: a character class (`[,;]+`) and an
escaped regex (`\\s+`). With only the escaped example, the model
reached for a backslash pattern even when a character class was needed
(`\w+` for "vowels"); with a digit class as the example, it copied
`[0-9]` for "numbers". The two examples avoid both.

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
- **Token healing at value boundaries.** Each value's first token may
  carry the space and opening quote, so the model writes the token it
  would naturally produce (` -5`, ` "/home`) instead of being forced to
  continue from an unnatural split.
- **No post-processing of arguments.** An earlier version replaced
  descriptive words with symbols (`asterisks` to `*`) after generation.
  It was removed: it overwrote correct answers (a quoted `'numbers'`
  became `\d+`), and the arguments now come from the model alone.
- **Pydantic everywhere.** Input files, generated calls and the internal
  classes (`TrieNode`, `Trie`, `ClosedChoice`) are pydantic models.
- **Fail per prompt, not per run.** A prompt that fails (or is empty) is
  skipped with a message; the rest are still written. Invalid input files
  stop the program with a clear error and exit code 1.

## Performance analysis

Measured on an Apple M3 Pro with the evaluation tester (moulinette),
which calls each chosen function with the generated arguments and
compares the return value:

| Metric | Public set | Private set |
|---|---|---|
| Fully correct calls (name and all arguments) | 11/11 | 11/11 |
| Valid JSON / schema-compliant output | 100% | 100% |
| Total run time, including model loading | ~55 s | ~65 s |

Generation is deterministic, so repeated runs give the same output.

Known weak spots:

- Quotes inside a string argument: the model tends to end the string at
  the first `"` instead of writing `\"`.
- String values longer than 64 tokens are cut off, and numbers in
  exponent notation (`2.5e-3`) are not supported.
- Leading and trailing spaces are trimmed from string values, so an
  argument that is only spaces (`' '`) becomes empty.
- Each step re-encodes the whole prefix (no key/value cache), so the time
  per token grows with the prompt length.

## Challenges faced

- **Token boundaries.** Tokens do not line up with JSON syntax: one token
  can contain the end of a string and the following `",`. Closing tokens
  are recognised by the quote inside them, and only the part before the
  quote is kept. The same problem appears at the start of a value: the
  evaluation tester showed that negative signs, leading slashes and
  prefix function names were lost because the skeleton split tokens the
  model normally writes whole. This was fixed with token healing (see
  the JSON skeleton section).
- **Byte-level vocabulary.** Tokens are stored as mapped characters, not
  raw bytes, so the GPT-2 byte mapping had to be reversed before any
  comparison was possible.
- **Knowing when a number ends.** A number has no closing character, so
  the model's unconstrained top choice is used to decide when to stop.
- **Regex arguments.** The model copied text from the prompt (`34`
  instead of `\d+`) and backslashes were originally banned. This was
  fixed with escape-aware string decoding and a clearer system prompt
  with generic examples.
- **Empty prompts.** An empty request made the model copy the example
  from the system prompt, so empty prompts are now skipped.

## Testing strategy

- `make lint` (flake8 and mypy with the required flags) must pass.
- The evaluation tester (moulinette) on both its public and private
  sets, plus extra sets in the same format (negative numbers, paths,
  prefix function names, long strings) graded with the same rule.
- Full runs on the provided files, checking every name and argument by
  hand.
- Extra prompt sets for strings that need escapes (backslashes, quotes,
  `\d+`, `\s+`), symbol words ("hashes", "dashes"), quoted words that
  must be kept as written, unicode, large and negative numbers, paths,
  function names that are prefixes of each other, and empty prompts.
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
  variants, adding escape-aware string decoding and an alias dictionary
  (later removed).
- Adding the `integer` parameter type after it was caught during
  evaluation.
- Diagnosing the token boundary problems (negative signs, leading
  slashes, prefix function names) found with the evaluation tester, and
  comparing system prompt examples to choose the final pair.
- Testing edge cases, running the evaluation tester, and drafting this
  README.

All generated code was reviewed, tested and adjusted before being kept.
