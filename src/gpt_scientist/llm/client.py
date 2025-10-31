"""OpenAI client wrapper for LLM interactions."""

import json
import logging
from typing import Optional
from openai import NOT_GIVEN, NotGiven
from pydantic import create_model

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for OpenAI async client with response parsing."""

    def __init__(self, async_client, model: str, system_prompt: str, use_structured_outputs: bool,
                 num_results: int, num_retries: int, max_tokens: Optional[int], top_p: float, pricing: dict):
        self._client = async_client
        self.model = model
        self.system_prompt = system_prompt
        self.use_structured_outputs = use_structured_outputs
        self.num_results = num_results
        self.num_retries = num_retries
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.pricing = pricing
        self.examples = []

    def get_top_p(self) -> float | NotGiven:
        """Get the top p parameter if supported by the current model, otherwise return NOT_GIVEN."""
        if self.model in self.pricing and 'top_p' in self.pricing[self.model]:
            if not self.pricing[self.model]['top_p']:
                return NOT_GIVEN
        return self.top_p

    def set_examples(self, examples: list[dict]):
        """Set few-shot examples for the model."""
        self.examples = examples

    async def prompt_model(self, prompt: str, output_fields: list[str]) -> dict:
        """Send the prompt to the model and return the completions."""
        if not self.use_structured_outputs:
            fn = self._client.chat.completions.create
            response_format = {"type": "json_object"}
        else:
            fn = self._client.chat.completions.parse
            response_format = create_model("Response", **{field: (str, ...) for field in output_fields})

        messages = [{"role": "system", "content": self.system_prompt}] + self.examples + [{"role": "user", "content": prompt}]

        return await fn(
            model=self.model,
            messages=messages,
            n=self.num_results,
            max_completion_tokens=self.max_tokens,
            response_format=response_format,
            top_p=self.get_top_p(),
        )

    def parse_response(self, completion, output_fields: list[str]) -> Optional[dict]:
        """Parse model completion into a dictionary."""
        if not self.use_structured_outputs:
            try:
                response = json.loads(completion.content.strip())
                # Check for missing fields unless we are using structured outputs
                missing_fields = [field for field in output_fields if field not in response]
                if missing_fields:
                    logger.warning(f"Response is missing fields {missing_fields}: {response}")
                    return None
                # If there are extra fields, we just ignore them
                return {field: response[field] for field in output_fields if field in response}
            except Exception as _:
                logger.warning(f"Failed to parse response: {completion}")
                return None
        else:
            if completion.refusal:
                logger.warning(f"Completion was refused: {completion.refusal}")
                return None
            return completion.parsed.model_dump()

    async def get_response(self, prompt: str, output_fields: list[str] = []) -> tuple[Optional[dict], int, int]:
        """
        Prompt the model until we get a valid json completion that contains all the output fields.
        Return None if no valid completion is generated after num_retries attempts.
        """
        req_input_tokens = 0
        req_output_tokens = 0

        for attempt in range(self.num_retries):
            if attempt > 0:
                logger.warning(f"Attempt {attempt + 1}")

            try:
                completions = await self.prompt_model(prompt, output_fields)

                u = getattr(completions, "usage", None)
                if u:
                    req_input_tokens += u.prompt_tokens
                    req_output_tokens += u.completion_tokens
                else:
                    # For older models, we might not have usage information
                    logger.warning("No usage information in the response; cost will be reported as 0.")

                for i in range(self.num_results):
                    response = self.parse_response(completions.choices[i].message, output_fields)
                    if response is None:
                        continue
                    logger.debug(f"Response:\n{response}")
                    return response, req_input_tokens, req_output_tokens
            except Exception as e:
                logger.warning(f"Could not get a response from the model: {e}")

        return None, req_input_tokens, req_output_tokens

    async def generate_embedding(self, text: str) -> tuple[list[float], int]:
        """Generates an embedding for a given text."""
        response = await self._client.embeddings.create(
            input=[text],
            model=self.model
        )
        u = getattr(response, "usage", None)
        if u:
            return response.data[0].embedding, u.prompt_tokens
        else:
            logger.warning("No usage information in the embedding response; cost will be reported as 0.")
            return response.data[0].embedding, 0
