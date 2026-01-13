"""Google Sheets processing - only available in Colab."""

import asyncio
import logging
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from gpt_scientist.llm.client import LLMClient
from gpt_scientist.processors.core import analyze_data
from gpt_scientist.config import GSHEET_FIRST_ROW, GOOGLE_DOC_URL_PATTERN
from gpt_scientist.stats import JobStats
from gpt_scientist.verification.quotes import check_quotes, verified_field_name

logger = logging.getLogger(__name__)

# Check if we are in Google Colab
try:
    from google.colab import auth
    IN_COLAB = True
    import gspread
    from google.auth import default
    from googleapiclient.discovery import build
    auth.authenticate_user()
except ImportError:
    IN_COLAB = False


async def get_gdoc_content(doc_id: str) -> str:
    """Get the content of a Google Doc as text."""
    if not IN_COLAB:
        logger.error("This method is only available in Google Colab.")
        return ""

    def _fetch_doc():
        from gpt_scientist.google_doc_parser import convert_to_text
        creds, _ = default()
        service = build('docs', 'v1', credentials=creds)
        doc = service.documents().get(documentId=doc_id).execute()
        return convert_to_text(doc['body']['content'])
    return await asyncio.to_thread(_fetch_doc)


async def follow_google_doc_url(url: str) -> str:
    """If URL is a Google Doc link, return the content of the document; otherwise return the input unchanged."""
    match = GOOGLE_DOC_URL_PATTERN.match(url)
    if match:
        logger.info(f"Opening Google Doc {url}")
        return await get_gdoc_content(match.group('doc_id'))
    else:
        return url


def parse_row_ranges(range_str: str, n_rows: int) -> list[int]:
    """
    Parse a g-sheet-style row range string (e.g., "2:10,12,15:") into a list of row indexes.
    Note that g-sheet ranges are effectively 2-based, because the first row is the header,
    and the result is 0-based.
    """
    row_indexes = []
    ranges = range_str.split(',')

    def parse_int(s):
        try:
            return int(s)
        except ValueError:
            logger.error(f"Invalid row range: {range_str}")
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


def convert_value_for_gsheet(val):
    """Convert complex types to strings for Google Sheets."""
    if isinstance(val, list):
        return ', '.join(map(str, val))  # Convert list to comma-separated string
    elif isinstance(val, dict):
        return str(val)  # Convert dictionary to string
    else:
        return val  # Leave supported types as-is


async def read_spreadsheet(
    key: str,
    worksheet_index: int,
    input_fields: list[str],
    input_range: str
):
    """
    Open a worksheet in a Google Sheet and return a pair of the worksheet and a pandas dataframe with the data.
    In the data, replace URLs to Google Docs with the content of the documents.
    """
    if not IN_COLAB:
        logger.error("This method is only available in Google Colab.")
        return None

    # Wrap all gspread I/O operations in to_thread
    def _open_and_read_sheet():
        creds, _ = default()
        gc = gspread.authorize(creds)
        if "docs.google.com" in key:
            spreadsheet = gc.open_by_url(key)
        else:
            spreadsheet = gc.open_by_key(key)
        worksheet = spreadsheet.get_worksheet(worksheet_index)
        header = worksheet.row_values(1)

        duplicate_headers = [col for col in header if header.count(col) > 1]
        if duplicate_headers:
            logger.error(f"Cannot analyze your spreadsheet because it contains duplicate headers: {set(duplicate_headers)}")
            return (worksheet, None, None)

        data = worksheet.get_all_records()
        return (worksheet, header, data)

    worksheet, header, data = await asyncio.to_thread(_open_and_read_sheet)

    if data is None:
        return (worksheet, None)

    data = pd.DataFrame(data)
    rows = parse_row_ranges(input_range, len(data))

    # For those input fields that are URLs to Google Docs, follow the links and get the content
    for field in input_fields:
        for i in rows:
            value = data.at[i, field]
            if isinstance(value, str):
                data.at[i, field] = await follow_google_doc_url(value)

    return (worksheet, data)


async def analyze_google_sheet(
    sheet_key: str,
    prompt: str,
    similarity_queries: list[str],
    input_fields: list[str],
    output_fields: list[str],
    rows: str,
    examples: str,
    overwrite: bool,
    worksheet_index: int,
    llm_client: LLMClient,
    similarity_mode: str,
    parallel_rows: int,
    stats: JobStats
):
    """
    When in Colab: analyze data in the Google Sheet with key `sheet_key`; the user must have write access to the sheet.
    Use `worksheet_index` to specify a sheet other than the first one.
    Async version.
    """
    # Open the spreadsheet and the worksheet, and read the data
    result = await read_spreadsheet(sheet_key, worksheet_index, input_fields, f'{rows},{examples}')
    if result is None:
        return
    worksheet, data = result
    if data is None:
        return

    input_range = parse_row_ranges(rows, len(data))
    example_range = parse_row_ranges(examples, len(data))

    # Prepare the worksheet for output and get output column indices
    def _prepare_output_columns():
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
        return output_column_indices

    output_column_indices = await asyncio.to_thread(_prepare_output_columns)

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
                value = convert_value_for_gsheet(data.at[i, field])
                cells.append(gspread.Cell(row=gsheet_row, col=gsheet_col, value=value))
        worksheet.update_cells(cells)

    await analyze_data(
        data,
        prompt,
        similarity_queries,
        input_fields,
        output_fields,
        write_output_rows,
        input_range,
        example_range,
        overwrite,
        llm_client,
        similarity_mode,
        parallel_rows,
        stats,
        row_index_offset=GSHEET_FIRST_ROW
    )


async def check_quotes_google_sheet(
    sheet_key: str,
    output_field: str,
    input_fields: list[str] = [],
    rows: str = ':',
    worksheet_index: int = 0,
    fuzzy_threshold: float = 0.25
):
    """Check quotes in a Google Sheet. Async version."""
    if not IN_COLAB:
        logger.error("This method is only available in Google Colab.")
        return

    # Import here since it's only available in Colab
    from gspread.utils import rowcol_to_a1

    # Open the spreadsheet and the worksheet, and read the data
    result = await read_spreadsheet(sheet_key, worksheet_index, input_fields, rows)
    if result is None:
        return
    worksheet, data = result
    if data is None:
        return

    rows_to_check = parse_row_ranges(rows, len(data))

    # Find the verified column or create one if it doesn't exist
    def _prepare_verified_column():
        verified_column_name = verified_field_name(output_field)
        header = worksheet.row_values(1)
        if verified_column_name in header:
            verified_column_index = header.index(verified_column_name) + 1
        else:
            output_column_index = header.index(output_field) + 1
            verified_column_index = output_column_index + 1
            if verified_column_index > worksheet.col_count:
                # Add more columns if necessary
                worksheet.add_cols(1)
            new_col_data = [verified_column_name] + [''] * (worksheet.row_count - 1)
            worksheet.insert_cols([new_col_data], verified_column_index)
        return verified_column_name, verified_column_index

    verified_column_name, verified_column_index = await asyncio.to_thread(_prepare_verified_column)

    # Perform quote checks (this is CPU-bound, not I/O)
    check_quotes(data, output_field, input_fields, rows_to_check, fuzzy_threshold)

    # Write results back to sheet
    def _write_verified_column():
        verified_column_data = [convert_value_for_gsheet(val) for val in data[verified_column_name].tolist()]
        verified_column_range = rowcol_to_a1(GSHEET_FIRST_ROW, verified_column_index) + ':' + rowcol_to_a1(GSHEET_FIRST_ROW + len(data) - 1, verified_column_index)
        worksheet.update([verified_column_data], verified_column_range, major_dimension='COLUMNS')

    await asyncio.to_thread(_write_verified_column)
