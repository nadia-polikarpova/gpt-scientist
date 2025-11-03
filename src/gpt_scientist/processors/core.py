"""Core data processing orchestration."""

import asyncio
import logging
import pandas as pd
from typing import Callable, Iterable
from gpt_scientist.llm.client import LLMClient
from gpt_scientist.stats import JobStats
from gpt_scientist.processors.workers import writer, analyze_row_worker, similarity_row_worker
from gpt_scientist.llm.prompts import create_example_messages
from gpt_scientist.config import is_embedding_model, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def validate_input(data: pd.DataFrame, input_fields: list[str], output_fields: list[str],
                   is_similarity: bool, model: str, pricing: dict) -> str:
    """
    Validate input parameters and adjust model if necessary.
    Return the (potentially adjusted) model to use.
    """
    adjusted_model = model

    if model not in pricing:
        logger.warning(f"No pricing available for {model}; cost will be reported as 0.")

    if is_similarity:
        if not is_embedding_model(model, pricing):
            logger.warning(f"You asked to compute similarity, but the current model is not an embedding model. Changing the model to an embedding model: {DEFAULT_EMBEDDING_MODEL}")
            adjusted_model = DEFAULT_EMBEDDING_MODEL
        # Check that there is exactly one input and output field
        if len(input_fields) != 1:
            raise ValueError("For similarity tasks, there must be exactly one input field (the text to compare to the prompts).")
        if len(output_fields) != 1:
            raise ValueError("For similarity tasks, there must be exactly one output field (the similarity score).")
    else:
        if is_embedding_model(model, pricing):
            logger.warning(f"You are using an embedding model ({model}) for a non-similarity task. Changing the model to a non-embedding model: {DEFAULT_MODEL}")
            adjusted_model = DEFAULT_MODEL

    # Check if all input fields are present in the dataframe
    for field in input_fields:
        if field not in data.columns:
            raise ValueError(f"Input field {field} not found in the data.")
    # If no input fields are specified, fail
    if not input_fields:
        raise ValueError("No input fields specified.")

    return adjusted_model


def prepare_output_fields(data: pd.DataFrame, output_fields: list[str]):
    """
    Ensure that all output fields are present in the dataframe.
    If an output field is missing, create it with empty strings.
    If an output field is present, convert it to string type.
    """
    for field in output_fields:
        if field not in data.columns:
            # If the output field is not in the dataframe, add it
            data[field] = ''
        else:
            # Otherwise, convert the field to string because the model will be returning strings
            # TODO: in the future, we may want to specify the type of the output fields
            data[field] = data[field].fillna('').astype(str)


async def analyze_data(
    data: pd.DataFrame,
    prompt: str,
    similarity_queries: list[str],
    input_fields: list[str],
    output_fields: list[str],
    write_output_rows: Callable[[pd.DataFrame, list[int]], None],
    rows: Iterable[int],
    examples: Iterable[int],
    overwrite: bool,
    llm_client: LLMClient,
    similarity_mode: str,
    parallel_rows: int,
    stats: JobStats,
    row_index_offset: int = 0
):
    """
    Analyze all the `rows` in a pandas dataframe:
    for every value in the input_field column,
    send to the model the `prompt`, together with names and values of `input_fields`;
    parse `output_fields` from the response and write the current row into the dataframe.
    The dataframe is modified in place.
    `write_output_row` is a function used to save progress after every row (e.g. write to a spreadsheet where data came from).
    `examples` is a sequence of row indexes to be used as few-shot examples for the model;
    if `overwrite` is false, rows where any of the `output_fields` is non-empty will be skipped;
    `row_index_offset` is only used for progress reporting,
    to account for the fact that the user might see a non-zero based row indexing.
    This function is asynchronous and uses `parallel_rows` workers to process this many rows in parallel,
    and a single writer to write the output rows.
    """
    is_similarity = len(similarity_queries) > 0

    # Validate and potentially adjust model
    adjusted_model = validate_input(data, input_fields, output_fields, is_similarity,
                                    llm_client.model, llm_client.pricing)
    if adjusted_model != llm_client.model:
        llm_client.model = adjusted_model
        stats.model = adjusted_model

    prepare_output_fields(data, output_fields)

    # Create task queues
    row_queue = asyncio.Queue(2 * parallel_rows)  # Double the size to avoid blocking
    output_queue = asyncio.Queue()

    # Prepare mode-specific setup and create worker coroutines
    if is_similarity:
        # Compute embeddings for the prompts
        tasks = [llm_client.generate_embedding(q) for q in similarity_queries]
        embeddings_and_tokens = await asyncio.gather(*tasks)
        query_embeddings = [emb for emb, _ in embeddings_and_tokens]
        input_tokens = sum(tokens for _, tokens in embeddings_and_tokens)
        stats.input_tokens += input_tokens
        # Create worker coroutines for similarity mode
        worker_coros = [
            similarity_row_worker(
                data, query_embeddings, input_fields[0], output_fields[0],
                row_queue, output_queue, llm_client, similarity_mode
            )
            for _ in range(parallel_rows)
        ]
    else:
        # Prepare the few-shot examples
        example_messages = []
        for i in examples:
            if i < 0 or i >= len(data):
                logger.warning(f"Skipping example {i + row_index_offset} (no such row)")
                continue
            row = data.loc[i]
            logger.info(f"Adding example row {i + row_index_offset}")
            example_messages.extend(create_example_messages(prompt, row, input_fields, output_fields,
                                                            llm_client.use_structured_outputs))
        llm_client.set_examples(example_messages)
        # Create worker coroutines for analyze mode
        worker_coros = [
            analyze_row_worker(
                data, prompt, input_fields, output_fields, row_queue, output_queue, llm_client
            )
            for _ in range(parallel_rows)
        ]

    # Start workers and writer in a task group
    async with asyncio.TaskGroup() as tg:
        # Start all workers
        for worker_coro in worker_coros:
            tg.create_task(worker_coro)
        # Start writer
        tg.create_task(writer(output_queue, write_output_rows, data, stats, row_index_offset))

        # Add rows to be processed by the workers
        for i in rows:
            if i < 0 or i >= len(data):
                logger.warning(f"Skipping row {i + row_index_offset} (no such row)")
                continue
            row = data.loc[i]
            if not overwrite and any(row[field] for field in output_fields):
                # If any of the output fields is already filled, skip the row
                logger.debug(f"Skipping row {i + row_index_offset} (already filled)")
                continue
            await row_queue.put(i)

        # Wait for input processing to finish
        await row_queue.join()

        # Tell workers and writer to shut down
        for _ in range(parallel_rows):
            await row_queue.put(None)
        await output_queue.put((None, None, 0, 0))
    # TaskGroup ensures all tasks complete or are cancelled when exiting the context
    stats.report_cost()
