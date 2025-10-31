"""Prompt formatting utilities."""

import json
import pandas as pd


def format_suffix(fields: list[str]) -> str:
    """Suffix added to the prompt to explain the expected format of the response."""
    return f"Return exactly one json object with the following fields: {', '.join(fields)}."


def input_fields_and_values(fields: list[str], row: pd.Series) -> str:
    """Format the input fields and values for the prompt."""
    return '\n\n'.join([f"{field}:\n```\n{row[field]}\n```" for field in fields])


def create_prompt(user_prompt: str, input_fields: list[str], output_fields: list[str],
                  row: pd.Series, use_structured_outputs: bool) -> str:
    """Create a full prompt from user prompt, input fields, and row data."""
    prompt = f"{user_prompt}\n{input_fields_and_values(input_fields, row)}"
    if not use_structured_outputs:
        # If we are not using structured outputs, we need to add the description of the expected format to the prompt
        prompt = f"{prompt}\n{format_suffix(output_fields)}"
    return prompt


def create_example_messages(prompt: str, row: pd.Series, input_fields: list[str],
                            output_fields: list[str], use_structured_outputs: bool) -> list[dict]:
    """
    Create a few-shot example where the user message is the prompt and input fields from the given row,
    and the model response is the output fields of the row.
    """
    # The input of the example is the full prompt as it would be sent to the model
    full_prompt = create_prompt(prompt, input_fields, output_fields, row, use_structured_outputs)
    # The output of the example is a json object with the output fields of the row
    response = {field: row[field] for field in output_fields}
    return [
        {"role": "user", "content": full_prompt},
        {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)}
    ]
