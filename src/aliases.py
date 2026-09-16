import re
from typing import Any

# Parameter names that hold a regex pattern
PATTERN_PARAM = re.compile(r"regex|pattern", re.IGNORECASE)

# Words the model writes for a regex instead of the pattern itself
REGEX_ALIASES: dict[str, str] = {
    "number": r"\d+",
    "numbers": r"\d+",
    "digit": r"\d",
    "digits": r"\d",
    "vowel": "[aeiouAEIOU]",
    "vowels": "[aeiouAEIOU]",
    "consonant": "[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]",
    "consonants": "[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]",
    "letter": "[a-zA-Z]",
    "letters": "[a-zA-Z]",
    "word": r"\w+",
    "words": r"\w+",
    "space": " ",
    "spaces": " ",
    "whitespace": r"\s+",
    "whitespaces": r"\s+",
}

# Words the model writes for a symbol instead of the symbol itself
SYMBOL_ALIASES: dict[str, str] = {
    "asterisk": "*",
    "asterisks": "*",
    "star": "*",
    "stars": "*",
    "hyphen": "-",
    "hyphens": "-",
    "dash": "-",
    "dashes": "-",
    "underscore": "_",
    "underscores": "_",
    "dot": ".",
    "dots": ".",
    "period": ".",
    "periods": ".",
    "comma": ",",
    "commas": ",",
    "space": " ",
    "spaces": " ",
    "hash": "#",
    "hashes": "#",
    "hashtag": "#",
    "hashtags": "#",
    "plus": "+",
    "pluses": "+",
    "slash": "/",
    "slashes": "/",
    "backslash": "\\",
    "backslashes": "\\",
    "pipe": "|",
    "pipes": "|",
    "colon": ":",
    "colons": ":",
    "semicolon": ";",
    "semicolons": ";",
    "tilde": "~",
    "tildes": "~",
    "ampersand": "&",
    "ampersands": "&",
    "at sign": "@",
    "at signs": "@",
    "dollar sign": "$",
    "dollar signs": "$",
    "percent sign": "%",
    "percent signs": "%",
    "question mark": "?",
    "question marks": "?",
    "exclamation mark": "!",
    "exclamation marks": "!",
}


def _is_quoted(value: str, user_prompt: str) -> bool:
    """Tells whether the value appears in quotes in the prompt"""
    quoted = re.compile(r"""['"]""" + re.escape(value) + r"""['"]""")
    return quoted.search(user_prompt) is not None


def apply_aliases(
    user_prompt: str,
    function: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Replaces descriptive words with the value they name, for functions
    that take a regex (e.g. "numbers" -> \\d+, "asterisks" -> *).
    Only a value that is exactly one of these words is replaced, and a
    word the prompt quotes is kept as written.
    """
    declared = function["parameters"]
    if not any(PATTERN_PARAM.search(key) for key in declared):
        return parameters
    result = dict(parameters)
    for key, value in parameters.items():
        if declared[key]["type"] != "string":
            continue
        word = value.strip().lower()
        if PATTERN_PARAM.search(key):
            result[key] = REGEX_ALIASES.get(word, value)
        elif word in SYMBOL_ALIASES and not _is_quoted(value, user_prompt):
            result[key] = SYMBOL_ALIASES[word]
    return result
