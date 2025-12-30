"""Async worker functions for processing data rows."""

import asyncio
import logging
import pandas as pd
from typing import Callable
from gpt_scientist.stats import JobStats
from gpt_scientist.llm.prompts import create_prompt

logger = logging.getLogger(__name__)


async def writer(
    queue: asyncio.Queue,
    write_output_rows: Callable[[pd.DataFrame, list[int]], None],
    data: pd.DataFrame,
    job_stats: JobStats,
    row_index_offset: int = 0
):
    """
    Worker that writes all outputs currently available in the queue to the dataframe
    and calls `write_output_rows` to save the progress.
    """
    while True:
        batch = []
        # Wait until there's something in the queue
        first_row, response, input_tokens, output_tokens = await queue.get()
        batch.append((first_row, response))

        # Drain the rest of the queue and save all responses in a batch;
        # this is done because writing to google sheets one row at a time is slow.
        while not queue.empty():
            i, response, row_input_tokens, row_output_tokens = queue.get_nowait()
            batch.append((i, response))
            input_tokens += row_input_tokens
            output_tokens += row_output_tokens

        # Update the dataframe with the responses
        indices_to_write = []
        for i, response in batch:
            if i is None:  # sentinel
                break
            if response is None:
                logger.warning(f"The model failed to generate a valid response for row: {i + row_index_offset}. Try again later?")
                job_stats.log_error()
            else:
                indices_to_write.append(i)
                for field in response:
                    data.at[i, field] = response[field]

        # Write valid rows persistent storage
        if indices_to_write:
            indices_to_write.sort()  # Sort indices to avoid unneeded reordering
            await asyncio.to_thread(write_output_rows, data, indices_to_write)

        # Log the number of rows processed in this batch
        # We count unsuccessful rows as well, because they still consume tokens, but we don't count the sentinel row
        rows_processed = len([i for i, _ in batch if i is not None])
        job_stats.log_rows(rows_processed, input_tokens, output_tokens)

        # Mark all dequeued items as done
        for _ in batch:
            queue.task_done()

        # If last row was a sentinel, we are done
        if batch[-1][0] is None:
            break


async def analyze_row_worker(
    data: pd.DataFrame,
    prompt: str,
    input_fields: list[str],
    output_fields: list[str],
    row_queue: asyncio.Queue,
    output_queue: asyncio.Queue,
    llm_client
):
    """
    Worker that processes a single row from the dataframe, sends it to the model,
    and puts the response in the output queue.
    """
    while True:
        i = await row_queue.get()
        if i is None:
            break
        try:
            row = data.loc[i]
            full_prompt = create_prompt(prompt, input_fields, output_fields, row, llm_client.use_structured_outputs)
            if i == 0:
                logger.info(f"Example prompt (first row):\n{full_prompt}")
            response, input_tokens, output_tokens = await llm_client.get_response(full_prompt, output_fields)
            await output_queue.put((i, response, input_tokens, output_tokens))
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            # Put None response to indicate failure
            await output_queue.put((i, None, 0, 0))
        finally:
            row_queue.task_done()


async def similarity_row_worker(
    data: pd.DataFrame,
    query_embeddings: list[list[float]],
    input_field: str,
    output_field: str,
    row_queue: asyncio.Queue,
    output_queue: asyncio.Queue,
    llm_client,
    similarity_mode: str
):
    """
    Worker that processes a single row from the dataframe for similarity tasks.
    """
    while True:
        i = await row_queue.get()
        if i is None:
            break
        try:
            row = data.loc[i]
            embedding, input_tokens = await llm_client.generate_embedding(row[input_field])
            # Compute dot product between the row embedding and each of the query embeddings
            similarities = [sum(e1 * e2 for e1, e2 in zip(embedding, q_emb)) for q_emb in query_embeddings]
            # Compute the final similarity score based on the selected mode
            if similarity_mode == 'max':
                response = {output_field: max(similarities)}
            else:  # similarity_mode == 'mean'
                response = {output_field: sum(similarities) / len(similarities)}
            await output_queue.put((i, response, input_tokens, 0))
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            # Put None response to indicate failure
            await output_queue.put((i, None, 0, 0))
        finally:
            row_queue.task_done()
