import json
import argparse
from pathlib import Path
from pydantic import BaseModel, ConfigDict, ValidationError
from typing import Literal, Any


class ParameterType(BaseModel):
    """This specifies that a parameter can be one of four things"""
    model_config = ConfigDict(extra="forbid")

    type: Literal["number", "string", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    """This is the acceptable template of a function"""
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: dict[str, str]


class PromptEntry(BaseModel):
    """A prompt must be a string"""
    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCall(BaseModel):
    """Specifies how the output of a function call must look like"""
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """This is for parsing arguments, input and output, from the terminal"""
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
        default=Path("data/output/function_calls.json"),
        help="Path to the function calls output."
    )
    return parser.parse_args()


def validate_answer(answer: Any) -> FunctionCall:
    """Validates whether or not the LLM outputed a proper answer"""
    validate = FunctionCall.model_validate(answer)
    return validate


def validate_json(funcs_path: str, prompts_path: str) -> bool:
    """File validation + JSON validation + Pydantic model validation"""
    try:
        f = funcs_path
        with open(funcs_path, "r") as funcs_raw:
            funcs = json.load(funcs_raw)
            for func in funcs:
                FunctionDefinition.model_validate(func)
        f = prompts_path
        with open(prompts_path, "r") as prompts_raw:
            prompts = json.load(prompts_raw)
            for prompt in prompts:
                PromptEntry.model_validate(prompt)
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
