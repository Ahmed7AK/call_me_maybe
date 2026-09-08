import json
import argparse
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Any


# This specifies that a parameter can be one of three things 
class ParameterType(BaseModel):
    ConfigDict(extra="forbid")

    type: Literal["number", "string", "boolean"]


# This is the acceptable template of a function
class FunctionDefinition(BaseModel):
    ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterType] = Field(min_length=1)
    returns: dict[str, str]


# A prompt must be a string
class PromptEntry(BaseModel):
    ConfigDict(extra="forbid")

    prompt: str


# Specifies how the output of a function call must look like
class FunctionCall(BaseModel):
    ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


# This is for parsing arguments from the terminal
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src",
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the function definitions file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the prompts file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to the function calls output."
    )
    return parser.parse_args()


def validate_answer(answer):
    validate = FunctionCall.model_validate(answer)
    return validate


# File validation + JSON validation + Pydantic model validation
def validate_json(funcs_path, prompts_path):
    try:
        f = funcs_path
        with open(funcs_path, "r") as funcs_raw:
            funcs = json.load(funcs_raw)
            for func in funcs:
                valid = FunctionDefinition.model_validate(func)
        f = prompts_path
        with open(prompts_path, "r") as prompts_raw:
            prompts = json.load(prompts_raw)
            for prompt in prompts:
                valid = PromptEntry.model_validate(prompt)
    except OSError as err:
        print(f"Trouble opening file - {f}: {err}")
        return False
    except json.JSONDecodeError as err:
        print(f"Trouble decoding json - {f}: {err}")
        return False
    except ValidationError as err:
        print(f"Pydantic validation error - {f}: {err.errors()[0]['msg']}")
        return False
    except Exception as err:
        print(f"Error - {f}: {err}")
        return False
    return True


def main():
    args = parse_args()
    validate_json(args.functions_definition, args.input)



if __name__ == "__main__":
    main()