"""Main module."""

import os
import re
import pandas as pd
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import tiktoken
import logging
import requests
import importlib.resources
from pydantic import create_model
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from gpt_scientist.google_doc_parser import convert_to_text, convert_to_markdown
from gpt_scientist.citation_checker import extract_citations, fuzzy_find_in_text
from typing import Callable, Iterable, Awaitable, TypeVar

# Check if we are in Google Colab, and if so authenticate and import libraries to work with Google Sheets
try:
    from google.colab import auth
    IN_COLAB = True
    import gspread
    from gspread.utils import rowcol_to_a1
    from google.auth import default
    from googleapiclient.discovery import build
    auth.authenticate_user()
except ImportError:
    IN_COLAB = False

# Github URL for the default pricing table
PRICING_URL = "https://raw.githubusercontent.com/nadia-polikarpova/gpt-scientist/main/src/gpt_scientist/model_pricing.json"
# Index of the first non-header row in google-sheet indexing
GSHEET_FIRST_ROW = 2
# Regular expression pattern for Google doc URL
GOOGLE_DOC_URL_PATTERN = re.compile(r'https://docs.google.com/document/d/(?P<doc_id>[^/]+)/.*')

class JobStats:
    '''Statistics for a table processing job.'''

    def __init__(self, pricing: dict, logger: logging.Logger):
        '''Initialize JobStats with optional pricing information.'''
        self.logger = logger
        self.pricing = pricing
        self.rows_processed = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def current_cost(self) -> dict:
        '''Return the cost corresponding to the current number of input and output tokens.'''
        input_cost = self.pricing['input'] * self.input_tokens / 1e6
        output_cost = self.pricing['output'] * self.output_tokens / 1e6
        return {'input': input_cost, 'output': output_cost}

    def report_cost(self):
        cost = self.current_cost()
        self.logger.info(f"PROCESSED {self.rows_processed} ROWS. TOTAL_COST: ${cost['input']:.4f} + ${cost['output']:.4f} = ${cost['input'] + cost['output']:.4f}")

    def log_rows(self, rows: int, input_tokens: int, output_tokens: int):
        '''Add the tokens used in the current row to the total and log the cost.'''
        self.rows_processed += rows
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self.rows_processed % 10 == 0:
            self.report_cost()

class Scientist:
    '''Configuration class for the GPT Scientist.'''
    def __init__(self, api_key: str = None):
        '''
            Initialize configuration parameters.
            If no API key is provided, the key is read from the .env file.
        '''
        if api_key:
            self._client = AsyncOpenAI(api_key=api_key)
        else:
            load_dotenv()
            self._client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = 'gpt-4o-mini' # Default model
        self.use_structured_outputs = False # Do not use structured outputs by default (this freezes with complex outputs)
        self.system_prompt = 'You are a social scientist analyzing textual data.' # Default system prompt
        self.num_results = 1 # How many completions to generate at once? The first valid completion will be used.
        self.num_reties = 10 # How many times to retry the request if no valid completion is generated?
        self.max_tokens = None # Maximum number of tokens to generate
        self.top_p = 0.3 # Top p parameter for nucleus sampling (this value is quite low, preferring more deterministic completions)
        self.parallel_rows = 100 # How many rows to process in parallel? This is the number of concurrent requests to the model.
        self.output_sheet = 'gpt_output' # Name (prefix) of the worksheet to save the output in Google Sheets
        self.max_fuzzy_distance = 30 # Maximum distance for fuzzy search
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._fetch_pricing() # Fetch the pricing table from GitHub or use the local file

    def set_model(self, model: str):
        '''Set the model to use for the GPT Scientist.'''
        self.model = model

    def set_use_structured_outputs(self, use_structured_outputs: bool):
        '''Set whether to use OpenAI's structured outputs feature to guarantee valid JSON responses.'''
        self.use_structured_outputs = use_structured_outputs

    def set_num_results(self, num_completions: int):
        '''Set the number of results to generate at once.'''
        self.num_results = num_completions

    def set_num_retries(self, num_retries: int):
        '''Set the number of retries if no valid completion is generated.'''
        self.num_reties = num_retries

    def set_system_prompt(self, system_prompt: str):
        '''Set the system prompt to use for the GPT Scientist.'''
        self.system_prompt = system_prompt

    def load_system_prompt_from_file(self, path: str):
        '''Load the system prompt from a file.'''
        with open(path, 'r') as f:
            self.system_prompt = f.read()

    def _get_gdoc_content(self, doc_id: str) -> str:
        '''Get the content of a Google Doc.'''
        creds, _ = default()
        service = build('docs', 'v1', credentials=creds)
        doc = service.documents().get(documentId=doc_id).execute()
        return convert_to_text(doc['body']['content'])

    def load_system_prompt_from_google_doc(self, doc_id: str):
        '''Load the system prompt from a Google Doc.'''
        if not IN_COLAB:
            self.logger.error("This method is only available in Google Colab.")
            return

        self.system_prompt = self._get_gdoc_content(doc_id)

    def set_max_tokens(self, max_tokens: int):
        '''Set the maximum number of tokens to generate.'''
        self.max_tokens = max_tokens

    def set_top_p(self, top_p: float):
        '''Set the top p parameter for nucleus sampling.'''
        self.top_p = top_p

    def set_parallel_rows(self, parallel_rows: int):
        '''Set the number of rows to process in parallel.'''
        self.parallel_rows = parallel_rows

    def set_output_sheet(self, output_sheet: str):
        '''Set the name (prefix) of the worksheet to save the output in Google Sheets.'''
        self.output_sheet = output_sheet

    def _fetch_pricing(self):
        try:
            # Try to fetch the pricing table from github
            resp = requests.get(PRICING_URL, timeout=2)
            if resp.ok:
                self.pricing = resp.json()
                self.logger.info(f"Fetched pricing table from {PRICING_URL}")
                return
        except requests.RequestException:
            pass
        # Otherwise: read the pricing table from the local file
        try:
            with importlib.resources.files("gpt_scientist").joinpath("model_pricing.json").open("r") as f:
                self.pricing = json.load(f)
                self.logger.info("Loaded pricing table from the local file.")
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Could not load the pricing table: {e}.")
            self.pricing = {}

    def set_pricing(self, pricing: dict):
        '''
            Add or update pricing information.
            Pricing table must be in the format {'model_name': {'input': input_cost, 'output': output_cost}},
            where input_cost and output_cost are the costs per 1M tokens.
        '''
        self.pricing = self.pricing | pricing

    def set_max_fuzzy_distance(self, max_fuzzy_distance: int):
        '''Set the maximum distance for fuzzy search.'''
        self.max_fuzzy_distance = max_fuzzy_distance

    def _format_suffix(self, fields: list[str]) -> str:
        '''Suffix added to the prompt to explain the expected format of the response.'''
        return f"Your response must be a json object with the following fields: {', '.join(fields)}. The response must start with {{, not with ```json."

    def _create_prompt(self, user_prompt: str, input_fields: list[str], output_fields: list[str], row: pd.Series) -> str:
        prompt = f"{user_prompt}\n{self._input_fields_and_values(input_fields, row)}"
        if not self.use_structured_outputs:
            # If we are not using structured outputs, we need to add the description of the expected format to the prompt
            prompt = f"{prompt}\n{self._format_suffix(output_fields)}"
        return prompt

    async def _prompt_model(self, prompt: str, output_fields: list[str]) -> dict:
        '''Send the prompt to the model and return the completions.'''
        if not self.use_structured_outputs:
            fn = self._client.chat.completions.create
            response_format={"type": "json_object"}
        else:
            fn = self._client.beta.chat.completions.parse
            response_format = create_model("Response", **{field: (str, ...) for field in output_fields})

        messages = [{"role": "system", "content": self.system_prompt}] + self._examples + [{"role": "user", "content": prompt}]

        return await fn(
                model=self.model,
                messages=messages,
                n=self.num_results,
                max_tokens=self.max_tokens,
                response_format=response_format,
                top_p=self.top_p,
            )

    def _parse_response(self, completion, output_fields: list[str]) -> dict:
        '''Parse model completion into a dictionary.'''
        if not self.use_structured_outputs:
            try:
                response = json.loads(completion.content.strip())
                # Check for missing fields unless we are using structured outputs
                missing_fields = [field for field in output_fields if field not in response]
                if missing_fields:
                    self.logger.warning(f"Response is missing fields {missing_fields}: {response}")
                    return None
                # If there are extra fields, we just ignore them
                return {field: response[field] for field in output_fields if field in response}
            except json.JSONDecodeError as _:
                self.logger.warning(f"Not a valid JSON: {completion}")
                return None
        else:
            if completion.refusal:
                self.logger.warning(f"Completion was refused: {completion.refusal}")
                return None
            return completion.parsed.dict()

    async def get_response(self, prompt: str, output_fields: list[str] = []) -> tuple[dict, int, int]:
        '''
            Prompt the model until we get a valid json completion that contains all the output fields.
            Return None if no valid completion is generated after scientist.num_reties attempts.
        '''
        req_input_tokens = len(self._tokenizer.encode(prompt))
        req_output_tokens = 0

        for attempt in range(self.num_reties):
            if attempt > 0:
                self.logger.warning(f"Attempt {attempt + 1}")

            try:
                completions = await self._prompt_model(prompt, output_fields)

                req_output_tokens += sum([len(self._tokenizer.encode(completions.choices[i].message.content)) for i in range(self.num_results)])

                for i in range(self.num_results):
                    response = self._parse_response(completions.choices[i].message, output_fields)
                    if response is None:
                        continue
                    self.logger.debug(f"Response:\n{response}")
                    return response, req_input_tokens, req_output_tokens
            except Exception as e:
                self.logger.warning(f"Could not get a response from the model: {e}")

        return None, req_input_tokens, req_output_tokens

    def _input_fields_and_values(self, fields: list[str], row: pd.Series) -> str:
        '''Format the input fields and values for the prompt.'''
        return '\n\n'.join([f"{field}:\n```\n{row[field]}\n```" for field in fields])

    def _add_example(self, prompt: str, row: pd.Series, input_fields: list[str], output_fields: list[str]):
        '''
            Create a few-shot example where the user message is the prompt and input fields form the given row,
            and the model response is the output fields of the row.
        '''
        # The input of the example is the full prompt as it would be sent to the model
        full_prompt = self._create_prompt(prompt, input_fields, output_fields, row)
        # The output of the example is a json object with the output fields of the row
        response = {field: row[field] for field in output_fields}
        self._examples.append({"role": "user", "content": full_prompt})
        self._examples.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})

    async def _writer(self,
                      queue: asyncio.Queue,
                      write_output_rows: Callable[[pd.DataFrame, list[int]], None],
                      data: pd.DataFrame,
                      job_stats: JobStats = None,
                      row_index_offset: int = 0):
        '''
            Worker that writes all outputs currently available in the queue to the dataframe and calls `write_output_rows` to save the progress.
        '''
        while True:
            batch = []
            # Wait until there's something in the queue
            first_row, response, input_tokens, output_tokens = await queue.get()
            batch.append((first_row, response))
            # self.logger.info(f"WRITER triggered on row {first_row}. Output queue size: {queue.qsize()}")

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
                    self.logger.error(f"The model failed to generate a valid response for row: {i + row_index_offset}. Try again later?")
                else:
                    indices_to_write.append(i)
                    for field in response:
                        data.at[i, field] = response[field]

            # Write valid rows persistent storage
            if indices_to_write:
                indices_to_write.sort()  # Sort indices to avoid unneeded reordering
                await asyncio.to_thread(write_output_rows, data, indices_to_write)

            # Log the number of rows processed in this batch
            if job_stats:
                # We count unsuccessful rows as well, because they still consume tokens, but we don't count the sentinel row
                rows_processed = len([i for i, _ in batch if i is not None])
                job_stats.log_rows(rows_processed, input_tokens, output_tokens)

            # Mark all dequeued items as done
            for _ in batch:
                queue.task_done()

            # If last row was a sentinel, we are done
            if batch[-1][0] is None:
                break

    async def _analyze_row_worker(self,
                                  data: pd.DataFrame,
                                  prompt: str,
                                  input_fields: list[str],
                                  output_fields: list[str],
                                  row_queue: asyncio.Queue,
                                  output_queue: asyncio.Queue):
        '''
            Worker that processes a single row from the dataframe, sends it to the model, and puts the response in the output queue.
        '''
        while True:
            i = await row_queue.get()
            if i is None:
                break
            row = data.loc[i]
            full_prompt = self._create_prompt(prompt, input_fields, output_fields, row)
            response, input_tokens, output_tokens = await self.get_response(full_prompt, output_fields)
            await output_queue.put((i, response, input_tokens, output_tokens))
            row_queue.task_done()

    async def analyze_data(self,
                     data: pd.DataFrame,
                     prompt: str,
                     input_fields: list[str],
                     output_fields: list[str],
                     write_output_rows: Callable[[pd.DataFrame, list[int]], None],
                     rows: Iterable[int],
                     examples: Iterable[int],
                     overwrite: bool,
                     row_index_offset: int = 0):
        '''
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
            This function is asynchronous and uses `self.parallel_rows` workers to process this many rows in parallel,
            and a single writer to write the output rows.
        '''

        # Check if all input fields are present in the dataframe
        for field in input_fields:
            if field not in data.columns:
                self.logger.error(f"Input field {field} not found.")
                return
        # If no input fields are specified, use all columns except the output fields
        if not input_fields:
            input_fields = [field for field in data.columns if field not in output_fields]

        # Create missing output fields
        for field in output_fields:
            if field not in data.columns:
                # If the output field is not in the dataframe, add it
                data[field] = ''
            else:
                # Otherise, convert the field to string because the model will be returning strings
                # TODO: in the future, we may want to specify the type of the output fields
                data[field] = data[field].fillna('').astype(str)

        # Create a tokenizer for the model
        try:
            self._tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # fallback for new or unknown models
            self.logger.warning(f"Not sure how to compute the token count for {self.model}. Using default tokenizer; cost might not be accurate.")
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        if self.model not in self.pricing:
            self.logger.warning(f"No pricing available for {self.model}; cost will be reported as 0.")

        # Prepare the few-shot examples
        self._examples = []
        for i in examples:
            if i < 0 or i >= len(data):
                self.logger.error(f"Skipping example {i + row_index_offset} (no such row)")
                continue
            row = data.loc[i]
            self.logger.info(f"Adding example row {i + row_index_offset}")
            self._add_example(prompt, row, input_fields, output_fields)

        # Create a task queue
        # TODO: We might want to limit the parallelism by the number of rows to process
        # but it's kinda annoying to count how many rows we actually have.
        row_queue = asyncio.Queue(2 * self.parallel_rows)  # Double the size to avoid blocking
        output_queue = asyncio.Queue()

        # Start workers
        for _ in range(self.parallel_rows):
            asyncio.create_task(self._analyze_row_worker(data, prompt, input_fields, output_fields, row_queue, output_queue))

        # Start writer
        stats = JobStats(self.pricing.get(self.model, {'input': 0, 'output': 0}), self.logger)
        writer_task = asyncio.create_task(self._writer(output_queue, write_output_rows, data, stats, row_index_offset))

        # Add rows to be processed by the workers
        for i in rows:
            if i < 0 or i >= len(data):
                self.logger.error(f"Skipping row {i + row_index_offset} (no such row)")
                continue
            row = data.loc[i]
            if not overwrite and any(row[field] for field in output_fields):
                # If any of the output fields is already filled, skip the row
                self.logger.info(f"Skipping row {i + row_index_offset} (already filled)")
                continue
            await row_queue.put(i)

        # Wait for input processing to finish
        await row_queue.join()

        # Tell workers and writer to shut down
        for _ in range(self.parallel_rows):
            await row_queue.put(None)
        await output_queue.put((None, None, 0, 0))
        await writer_task
        stats.report_cost()

    def analyze_csv(self,
                    path: str,
                    prompt: str,
                    input_fields: list[str] = [],
                    output_fields: list[str] = ['gpt_output'],
                    rows: Iterable[int] | None = None,
                    examples: Iterable[int] = [],
                    in_place: bool = True,
                    overwrite: bool = False):
        '''
            Analyze a CSV file.
            If in_place is True, save the results to the input file, otherwise create a unique output file.
        '''
        # Create a unique output file name based on current time;
        # when in_place is True, this file only serves as a backup, in case the finally block fails to run
        out_file_name = os.path.splitext(path)[0] + f'_output_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S")}.csv'

        def write_output_rows(data, indices):
            # Append the rows to the output file
            data.loc[indices].to_csv(out_file_name, mode='a', header=False, index=True)

        data = pd.read_csv(path, dtype=str, na_filter=False)
        # Write headers once at the top
        data.iloc[[]].to_csv(out_file_name, mode='w', index=True)
        if rows is None:
            rows = range(len(data))
        try:
            _run_async(self.analyze_data(data, prompt, input_fields, output_fields, write_output_rows, rows, examples, overwrite))
        except Exception as e:
            raise RuntimeError(f"Error analyzing CSV: {e}")
        finally:
            if in_place and os.path.exists(out_file_name):
                data.to_csv(path, index=False)
                os.remove(out_file_name)

    def _read_spreadsheet(self,
                          key: str,
                          worksheet_index: int,
                          input_fields: list[str],
                          input_range: str,
                          in_place: bool = True):
        '''
            Open a worksheet in a Google Sheet and return a pair of the worksheet and a pandas dataframe with the data.
            If in_place is False, create a copy of the worksheet.
            In the data, replace URLs to Google Docs with the content of the documents.
        '''
        if not IN_COLAB:
            self.logger.error("This method is only available in Google Colab.")
            return
        creds, _ = default()
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(key)
        worksheet = spreadsheet.get_worksheet(worksheet_index)
        # If in_place is False, create a copy of the worksheet
        if not in_place:
            worksheet = worksheet.duplicate(new_sheet_name=self._output_sheet_name(spreadsheet), insert_sheet_index=worksheet_index+1)

        header = worksheet.row_values(1)

        duplicate_headers = [col for col in header if header.count(col) > 1]
        if duplicate_headers:
            self.logger.error(f"Cannot analyze your spreadsheet because it contains duplicate headers: {set(duplicate_headers)}")
            return (worksheet, None)

        data = worksheet.get_all_records()
        data = pd.DataFrame(data)
        rows = self._parse_row_ranges(input_range, len(data))

        # For those input fields that are URLs to Google Docs, follow the links and get the content as markdown
        for field in input_fields:
            for i in rows:
                data.at[i, field] = self._follow_google_doc_url(data.at[i, field])

        return (worksheet, data)


    def _parse_row_ranges(self, range_str: str, n_rows: int) -> list[int]:
        '''
            Parse a g-sheet-style row range string (e.g., "2:10,12,15:") into a list of row indexes.
            Note that g-sheet ranges are effectively 2-based, because the first row is the header,
            and the result is 0-based.
        '''
        row_indexes = []
        ranges = range_str.split(',')

        def parse_int(s):
            try:
                return int(s)
            except ValueError:
                self.logger.error(f"Invalid row range: {range_str}")
                return GSHEET_FIRST_ROW

        for r in ranges:
            if ':' in r:  # Range like 1:10, 2:, or :
                parts = r.split(':')
                if len(parts[0]) == 0:
                    start = 0
                else:
                    start = parse_int(parts[0]) - GSHEET_FIRST_ROW
                if len(parts[1]) == 0:
                    end = n_rows
                else:
                    end = parse_int(parts[1]) - GSHEET_FIRST_ROW + 1
                row_indexes.extend(range(start, end))
            elif r:  # Single row like 1
                row_indexes.append(parse_int(r) - GSHEET_FIRST_ROW)

        return row_indexes

    def _output_sheet_name(self, spreadsheet) -> str:
        '''Create a new worksheet in the spreadsheet to save the output, avoiding name conflicts.'''
        worksheet_list = spreadsheet.worksheets()
        worksheet_names = [worksheet.title for worksheet in worksheet_list]
        if self.output_sheet in worksheet_names:
            i = 1
            while f"{self.output_sheet}_{i}" in worksheet_names:
                i += 1
            return f"{self.output_sheet}_{i}"
        else:
            return self.output_sheet

    def _convert_value_for_gsheet(self, val):
        '''Convert complex types to strings for Google Sheets.'''
        if isinstance(val, list):
            return ', '.join(map(str, val))  # Convert list to comma-separated string
        elif isinstance(val, dict):
            return str(val)  # Convert dictionary to string
        else:
            return val  # Leave supported types as-is

    def _follow_google_doc_url(self, url: str) -> str:
        '''If URL is a Google Doc link, return the content of the document as markdown; otherwise return the input unchanged.'''
        match = GOOGLE_DOC_URL_PATTERN.match(url)
        if match:
            self.logger.info(f"Opening Google Doc {url}")
            return self._get_gdoc_content(match.group('doc_id'))
        else:
            return url

    def analyze_google_sheet(self,
                             sheet_key: str,
                             prompt: str,
                             input_fields: list[str] = [],
                             output_fields: list[str] = ['gpt_output'],
                             rows: str = ':',
                             examples: str = '',
                             in_place: bool = True,
                             overwrite: bool = False,
                             worksheet_index: int = 0):
        '''
            When in Colab: analyze data in the Google Sheet with key `sheet_key`; the user must have write access to the sheet.
            Use `worksheet_index` to specify a sheet other than the first one.
            If `in_place` is True, the input sheet will be extended with the output data; otherwise a new sheet will be created.
            If `n_rows` is provided, only the first n_rows are processed (useful for testing).
        '''
        # Open the spreadsheet and the worksheet, and read the data
        worksheet, data = self._read_spreadsheet(sheet_key, worksheet_index, input_fields, f'{rows},{examples}', in_place)
        if data is None:
            return

        input_range = self._parse_row_ranges(rows, len(data))
        example_range = self._parse_row_ranges(examples, len(data))

        # Prepare the worksheet for output and get output column indices
        output_column_indices = []
        header = worksheet.row_values(1)
        for field in output_fields:
            if field in header:
                # If the column exists, get its index (1-based)
                output_column_indices.append(header.index(field) + 1)
            else:
                if len(header) + 1 > worksheet.col_count:
                    # Add more columns if necessary
                    worksheet.add_cols(1)
                # If the column doesn't exist, append it to the header
                worksheet.update_cell(1, len(header) + 1, field)  # Add to the next available column
                output_column_indices.append(len(header) + 1)
                header.append(field)  # Update the header list

        # Now we have the column indices, prepare the function that outputs a list of rows
        @retry(
            wait=wait_exponential(min=10, max=60),  # Exponential back-off, 10 to 60 seconds
            stop=stop_after_attempt(10),  # Max 10 retries
            retry=retry_if_exception_type(Exception)  # Retry on any exception
        )
        def write_output_rows(data, indices):
            cells = []
            for i in indices:
                gsheet_row = i + GSHEET_FIRST_ROW
                for j, field in enumerate(output_fields):
                    gsheet_col = output_column_indices[j]
                    value = self._convert_value_for_gsheet(data.at[i, field])
                    cells.append(gspread.Cell(row=gsheet_row, col=gsheet_col, value=value))
            worksheet.update_cells(cells)

        _run_async(self.analyze_data(data,
                                      prompt,
                                      input_fields,
                                      output_fields,
                                      write_output_rows,
                                      input_range,
                                      example_range,
                                      overwrite,
                                      row_index_offset=GSHEET_FIRST_ROW))

    def _verified_field_name(self, output_field: str) -> str:
        return f'{output_field}_verified'

    def check_citations(self,
                        data: pd.DataFrame,
                        output_field: str,
                        input_fields: list[str],
                        rows: Iterable[int]):
        '''
            For each row in the rows range, check that the citations from the output field actually exist in one of the input fields.
            We assume that the values in output_field are strings that contain citations in quotes,
            and the values in all input fields are strings.
            Record the results in a new column called {output_field}_verified.
        '''
        if not (self._verified_field_name(output_field) in data.columns):
            data[self._verified_field_name(output_field)] = ''
        for row in rows:
            output = data.loc[row, output_field]
            citations = extract_citations(output)
            input_text = '\n\n'.join(data.loc[row, input_fields])
            verified = output
            for citation in citations:
                self.logger.info(f'Checking citation: "{citation[:50]}..."')
                matched = fuzzy_find_in_text(citation, input_text, self.max_fuzzy_distance)

                if matched:
                    (res, dist) = matched
                    verified = verified.replace(citation, res)
                    if dist == 0:
                      self.logger.info("Found exact match")
                    else:
                      self.logger.info(f"Found a match {dist} character(s) apart")
                else:
                    verified = verified.replace(citation, 'CITATION NOT FOUND')
                    self.logger.info(f"CITATION NOT FOUND")

            data.loc[row, self._verified_field_name(output_field)] = verified

    def check_citations_csv(self,
                            path: str,
                            output_field: str,
                            input_fields: list[str] = [],
                            rows: Iterable[int] | None = None,
                            in_place: bool = True):
        '''
            The same as check_citations, but for a CSV file.
            If in_place is True, save the results to the input file, otherwise create a unique output file.
        '''
        # Create a unique output file name based on current time;
        # when in_place is True, this file only serves as a backup, in case the finally block fails to run
        out_file_name = os.path.splitext(path)[0] + f'_verified_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S")}.csv'

        data = pd.read_csv(path)
        if rows is None:
            rows = range(len(data))

        # Perform citation checks
        self.check_citations(data, output_field, input_fields, rows)

        # Save the results
        if in_place:
            data.to_csv(path, index=False)
        else:
            data.to_csv(out_file_name, index=False)


    def check_citations_google_sheet(self,
                                      sheet_key: str,
                                      output_field: str,
                                      input_fields: list[str] = [],
                                      rows: str = ':',
                                      worksheet_index: int = 0):
        '''The same as check_citations, but for a Google Sheet.'''
        # Open the spreadsheet and the worksheet, and read the data
        worksheet, data = self._read_spreadsheet(sheet_key, worksheet_index, input_fields, rows)
        if data is None:
            return

        rows = self._parse_row_ranges(rows, len(data))
        # Find the verified column or create one if it doesn't exist
        verified_column = self._verified_field_name(output_field)
        header = worksheet.row_values(1)
        if verified_column in header:
            verified_column_index = header.index(verified_column) + 1
        else:
            output_column_index = header.index(output_field) + 1
            verified_column_index = output_column_index + 1
            if verified_column_index > worksheet.col_count:
                # Add more columns if necessary
                worksheet.add_cols(1)
            new_col_data = [verified_column] + [''] * (worksheet.row_count - 1)
            worksheet.insert_cols([new_col_data], verified_column_index)

        self.check_citations(data, output_field, input_fields, rows)
        verified_column = [self._convert_value_for_gsheet(val) for val in data[verified_column].tolist()]
        verified_column_range = rowcol_to_a1(GSHEET_FIRST_ROW, verified_column_index) + ':' + rowcol_to_a1(GSHEET_FIRST_ROW + len(data) - 1, verified_column_index)
        worksheet.update([verified_column], verified_column_range, major_dimension='COLUMNS')

T = TypeVar("T")

def _run_async(coro: Awaitable[T]) -> T:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop is running, safe to use asyncio.run
        return asyncio.run(coro)

    # A loop is already running (e.g., notebook)
    import nest_asyncio
    nest_asyncio.apply()
    return loop.run_until_complete(coro)
