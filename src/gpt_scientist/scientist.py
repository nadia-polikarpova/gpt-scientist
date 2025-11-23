"""Main Scientist class - orchestrator for gpt_scientist."""

import os
from openai import AsyncOpenAI
from typing import Iterable, Optional
import logging
from pandas import DataFrame

from gpt_scientist.config import DEFAULT_MODEL, fetch_pricing
from gpt_scientist.llm.client import LLMClient
from gpt_scientist.processors.csv import analyze_csv, check_quotes_csv
from gpt_scientist.processors.sheets import analyze_google_sheet, check_quotes_google_sheet, get_gdoc_content, IN_COLAB
from gpt_scientist.utils import run_async
from gpt_scientist.stats import JobStats
from gpt_scientist.verification.quotes import check_quotes

logger = logging.getLogger(__name__)


class Scientist:
    """Configuration class for the GPT Scientist."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize configuration parameters.
        If no API key is provided, it will be read from the OPENAI_API_KEY environment variable.
        """
        if api_key:
            self._async_client = AsyncOpenAI(api_key=api_key)
        else:
            self._async_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        self.model = DEFAULT_MODEL
        self.use_structured_outputs = False  # Do not use structured outputs by default
        self.system_prompt = 'You are a social scientist analyzing textual data.'
        self.num_results = 1  # How many completions to generate at once?
        self.num_retries = 10  # How many times to retry if no valid completion?
        self.max_tokens = None  # Maximum number of tokens to generate
        self.top_p = 0.3  # Top p parameter for nucleus sampling
        self.similarity_mode = 'max'  # Similarity mode: 'max' (default) or 'mean'
        self.parallel_rows = 100  # How many rows to process in parallel?
        self.output_sheet = 'gpt_output'  # Name (prefix) of the worksheet in Google Sheets
        self.max_fuzzy_distance = 30  # Maximum distance for fuzzy search
        self.pricing = fetch_pricing()
        self.report_interval = self.parallel_rows  # How often to report cost (in number of rows processed)
        self._init_job_stats()  # We don't really need to init this here, but we do this to avoid mypy errors

    def _create_llm_client(self) -> LLMClient:
        """Create an LLM client with current configuration."""
        return LLMClient(
            self._async_client,
            self.model,
            self.system_prompt,
            self.use_structured_outputs,
            self.num_results,
            self.num_retries,
            self.max_tokens,
            self.top_p,
            self.pricing
        )

    def _init_job_stats(self):
        """Initialize JobStats with current model and pricing."""
        self.stats = JobStats(self.model, self.pricing, self.report_interval)

    # Configuration setters
    def set_model(self, model: str):
        """Set the model to use for the GPT Scientist."""
        self.model = model

    def set_use_structured_outputs(self, use_structured_outputs: bool):
        """Set whether to use OpenAI's structured outputs feature."""
        self.use_structured_outputs = use_structured_outputs

    def set_num_results(self, num_completions: int):
        """Set the number of results to generate at once."""
        self.num_results = num_completions

    def set_num_retries(self, num_retries: int):
        """Set the number of retries if no valid completion is generated."""
        self.num_retries = num_retries

    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt to use for the GPT Scientist."""
        self.system_prompt = system_prompt

    def load_system_prompt_from_file(self, path: str):
        """Load the system prompt from a file."""
        with open(path, 'r') as f:
            self.system_prompt = f.read()

    async def load_system_prompt_from_google_doc_async(self, doc_id: str):
        """Load the system prompt from a Google Doc. Async version."""
        if not IN_COLAB:
            logger.error("This method is only available in Google Colab.")
            return
        self.system_prompt = await get_gdoc_content(doc_id)

    def load_system_prompt_from_google_doc(self, doc_id: str):
        """Load the system prompt from a Google Doc. Sync wrapper."""
        return run_async(self.load_system_prompt_from_google_doc_async(doc_id))

    def set_max_tokens(self, max_tokens: int):
        """Set the maximum number of tokens to generate."""
        self.max_tokens = max_tokens

    def set_similarity_mode(self, similarity_mode: str):
        """Set the similarity mode: 'max' (default) or 'mean'."""
        if similarity_mode not in ['max', 'mean']:
            logger.error("Invalid similarity mode. Must be 'max' or 'mean'.")
            return
        self.similarity_mode = similarity_mode

    def set_top_p(self, top_p: float):
        """Set the top p parameter for nucleus sampling."""
        self.top_p = top_p

    def set_parallel_rows(self, parallel_rows: int):
        """Set the number of rows to process in parallel."""
        self.parallel_rows = parallel_rows

    def set_output_sheet(self, output_sheet: str):
        """Set the name (prefix) of the worksheet to save the output in Google Sheets."""
        self.output_sheet = output_sheet

    def set_pricing(self, pricing: dict):
        """
        Add or update pricing information.
        Pricing table must be in the format {'model_name': {'input': input_cost, 'output': output_cost}},
        where input_cost and output_cost are the costs per 1M tokens.
        """
        self.pricing = self.pricing | pricing

    def set_report_interval(self, report_interval: int):
        """Set the interval (in number of rows processed) to report cost. 0 means no reporting."""
        self.report_interval = report_interval

    def set_max_fuzzy_distance(self, max_fuzzy_distance: int):
        """Set the maximum distance for fuzzy search."""
        self.max_fuzzy_distance = max_fuzzy_distance

    # CSV processing methods
    async def analyze_csv_async(
        self,
        path: str,
        prompt: str = '',
        similarity_queries: list[str] = [],
        input_fields: list[str] = [],
        output_fields: list[str] = ['gpt_output'],
        rows: Optional[Iterable[int]] = None,
        examples: Optional[Iterable[int]] = None,
        overwrite: bool = False
    ):
        """Analyze a CSV file (in place) - async version."""
        llm_client = self._create_llm_client()
        # Reset stats for this analysis run
        self._init_job_stats()
        assert self.stats is not None
        return await analyze_csv(
            path, prompt, similarity_queries, input_fields, output_fields,
            rows, examples, overwrite, llm_client, self.similarity_mode, self.parallel_rows,
            self.stats
        )

    def analyze_csv(
        self,
        path: str,
        prompt: str = '',
        similarity_queries: list[str] = [],
        input_fields: list[str] = [],
        output_fields: list[str] = ['gpt_output'],
        rows: Optional[Iterable[int]] = None,
        examples: Optional[Iterable[int]] = None,
        overwrite: bool = False
    ):
        """Analyze a CSV file (in place) - sync wrapper."""
        return run_async(self.analyze_csv_async(
            path, prompt, similarity_queries, input_fields, output_fields, rows, examples, overwrite
        ))

    # Google Sheets processing methods
    async def analyze_google_sheet_async(
        self,
        sheet_key: str,
        prompt: str,
        similarity_queries: list[str] = [],
        input_fields: list[str] = [],
        output_fields: list[str] = ['gpt_output'],
        rows: str = ':',
        examples: str = '',
        overwrite: bool = False,
        worksheet_index: int = 0
    ):
        """
        When in Colab: analyze data in the Google Sheet with key `sheet_key`.
        Async version.
        """
        llm_client = self._create_llm_client()
        # Reset stats for this analysis run
        self._init_job_stats()
        assert self.stats is not None
        return await analyze_google_sheet(
            sheet_key, prompt, similarity_queries, input_fields, output_fields,
            rows, examples, overwrite, worksheet_index, llm_client,
            self.similarity_mode, self.parallel_rows, self.stats
        )

    def analyze_google_sheet(
        self,
        sheet_key: str,
        prompt: str,
        similarity_queries: list[str] = [],
        input_fields: list[str] = [],
        output_fields: list[str] = ['gpt_output'],
        rows: str = ':',
        examples: str = '',
        overwrite: bool = False,
        worksheet_index: int = 0
    ):
        """
        When in Colab: analyze data in the Google Sheet with key `sheet_key`.
        Sync wrapper.
        """
        return run_async(self.analyze_google_sheet_async(
            sheet_key, prompt, similarity_queries, input_fields, output_fields,
            rows, examples, overwrite, worksheet_index
        ))

    def check_quotes(
        self,
        data: DataFrame,
        output_field: str,
        input_fields: list[str] = [],
        rows: Iterable[int] | None = None
    ):
        """Check quotes in a DataFrame."""
        if rows is None:
            rows = range(len(data))
        check_quotes(data, output_field, input_fields, rows, self.max_fuzzy_distance)

    # Quote verification methods
    async def check_quotes_csv_async(
        self,
        path: str,
        output_field: str,
        input_fields: list[str] = [],
        rows: Iterable[int] | None = None
    ):
        """Check quotes in a CSV file. Async version."""
        await check_quotes_csv(path, output_field, input_fields, rows, self.max_fuzzy_distance)

    def check_quotes_csv(
        self,
        path: str,
        output_field: str,
        input_fields: list[str] = [],
        rows: Iterable[int] | None = None
    ):
        """Check quotes in a CSV file. Sync wrapper."""
        run_async(self.check_quotes_csv_async(path, output_field, input_fields, rows))

    async def check_quotes_google_sheet_async(
        self,
        sheet_key: str,
        output_field: str,
        input_fields: list[str] = [],
        rows: str = ':',
        worksheet_index: int = 0
    ):
        """Check quotes in a Google Sheet. Async version."""
        await check_quotes_google_sheet(
            sheet_key, output_field, input_fields, rows, worksheet_index, self.max_fuzzy_distance
        )

    def check_quotes_google_sheet(
        self,
        sheet_key: str,
        output_field: str,
        input_fields: list[str] = [],
        rows: str = ':',
        worksheet_index: int = 0
    ):
        """Check quotes in a Google Sheet. Sync wrapper."""
        run_async(self.check_quotes_google_sheet_async(
            sheet_key, output_field, input_fields, rows, worksheet_index
        ))
