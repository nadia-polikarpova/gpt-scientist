"""CSV file processing."""

import os
import asyncio
import pandas as pd
from typing import Iterable, Optional
from gpt_scientist.llm.client import LLMClient
from gpt_scientist.processors.core import analyze_data
from gpt_scientist.stats import JobStats
from gpt_scientist.verification.quotes import check_quotes

async def analyze_csv(
    path: str,
    prompt: str,
    similarity_queries: list[str],
    input_fields: list[str],
    output_fields: list[str],
    rows: Optional[Iterable[int]],
    examples: Optional[Iterable[int]],
    overwrite: bool,
    llm_client: LLMClient,
    similarity_mode: str,
    parallel_rows: int,
    stats: JobStats
):
    """Analyze a CSV file (in place) - async version."""
    # Create a unique output file name based on current time;
    # this file only serves as a backup, in case the finally block fails to run
    out_file_name = os.path.splitext(path)[0] + f'_output_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S")}.csv'

    def write_output_rows(data, indices):
        # Append the rows to the output file
        data.loc[indices].to_csv(out_file_name, mode='a', header=False, index=True)

    # Use asyncio.to_thread for blocking I/O operations
    data = await asyncio.to_thread(pd.read_csv, path, dtype=str, na_filter=False)
    # Write headers once at the top
    await asyncio.to_thread(data.iloc[[]].to_csv, out_file_name, mode='w', index=True)
    if rows is None:
        rows = range(len(data))
    if examples is None:
        examples = []
    try:
        await analyze_data(data, prompt, similarity_queries, input_fields, output_fields,
                          write_output_rows, rows, examples, overwrite, llm_client,
                          similarity_mode, parallel_rows, stats)
    except Exception as e:
        raise RuntimeError(f"Error analyzing CSV: {e}")
    finally:
        if os.path.exists(out_file_name):
            await asyncio.to_thread(data.to_csv, path, index=False)
            await asyncio.to_thread(os.remove, out_file_name)


async def check_quotes_csv(
    path: str,
    output_field: str,
    input_fields: list[str] = [],
    rows: Optional[Iterable[int]] = None,
    max_fuzzy_distance: int = 30
):
    """Check quotes in a CSV file. Async version."""
    # Read CSV asynchronously
    data = await asyncio.to_thread(pd.read_csv, path)
    if rows is None:
        rows = range(len(data))

    # Perform quote checks (CPU-bound, but quick enough to not need threading)
    check_quotes(data, output_field, input_fields, rows, max_fuzzy_distance)

    # Save the results asynchronously
    await asyncio.to_thread(data.to_csv, path, index=False)
