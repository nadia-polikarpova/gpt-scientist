"""Quote verification functionality."""

import asyncio
import logging
import pandas as pd
from typing import Iterable
from gpt_scientist.quote_checker import extract_quotes, fuzzy_find_in_text
from gpt_scientist.config import GSHEET_FIRST_ROW

logger = logging.getLogger(__name__)

# Check if we are in Google Colab
try:
    from google.colab import auth
    IN_COLAB = True
    import gspread
    from gspread.utils import rowcol_to_a1
    auth.authenticate_user()
except ImportError:
    IN_COLAB = False


def verified_field_name(output_field: str) -> str:
    """Return the name of the verified field for a given output field."""
    return f'{output_field}_verified'


def check_quotes(
    data: pd.DataFrame,
    output_field: str,
    input_fields: list[str],
    rows: Iterable[int],
    max_fuzzy_distance: int
):
    """
    For each row in the rows range, check that the quotes from the output field actually exist in one of the input fields.
    We assume that the values in output_field are strings that contain quotes in quotes,
    and the values in all input fields are strings.
    Record the results in a new column called {output_field}_verified.
    """
    verified_field = verified_field_name(output_field)
    if not (verified_field in data.columns):
        data[verified_field] = ''
    for row in rows:
        output = data.loc[row, output_field]
        quotes = extract_quotes(output)
        input_text = '\n\n'.join(data.loc[row, input_fields])
        verified = output
        for quote in quotes:
            logger.info(f'Checking quote: "{quote[:50]}..."')
            matched = fuzzy_find_in_text(quote, input_text, max_fuzzy_distance)

            if matched:
                (res, dist) = matched
                verified = verified.replace(quote, res)
                if dist == 0:
                    logger.info("Found exact match")
                else:
                    logger.info(f"Found a match {dist} character(s) apart")
            else:
                verified = verified.replace(quote, 'QUOTE NOT FOUND')
                logger.info(f"QUOTE NOT FOUND")

        data.loc[row, verified_field] = verified


def check_quotes_csv(
    path: str,
    output_field: str,
    input_fields: list[str] = [],
    rows: Iterable[int] | None = None,
    max_fuzzy_distance: int = 30
):
    """Check quotes in a CSV file."""
    data = pd.read_csv(path)
    if rows is None:
        rows = range(len(data))

    # Perform quote checks
    check_quotes(data, output_field, input_fields, rows, max_fuzzy_distance)

    # Save the results
    data.to_csv(path, index=False)


async def check_quotes_google_sheet_async(
    sheet_key: str,
    output_field: str,
    input_fields: list[str] = [],
    rows: str = ':',
    worksheet_index: int = 0,
    max_fuzzy_distance: int = 30
):
    """Check quotes in a Google Sheet. Async version."""
    if not IN_COLAB:
        logger.error("This method is only available in Google Colab.")
        return

    from gpt_scientist.processors.sheets import read_spreadsheet, parse_row_ranges, convert_value_for_gsheet

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
    check_quotes(data, output_field, input_fields, rows_to_check, max_fuzzy_distance)

    # Write results back to sheet
    def _write_verified_column():
        verified_column_data = [convert_value_for_gsheet(val) for val in data[verified_column_name].tolist()]
        verified_column_range = rowcol_to_a1(GSHEET_FIRST_ROW, verified_column_index) + ':' + rowcol_to_a1(GSHEET_FIRST_ROW + len(data) - 1, verified_column_index)
        worksheet.update([verified_column_data], verified_column_range, major_dimension='COLUMNS')

    await asyncio.to_thread(_write_verified_column)
