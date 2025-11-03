"""Core quote verification functionality."""

import logging
import re
import pandas as pd
from typing import Iterable
from fuzzysearch import find_near_matches

logger = logging.getLogger(__name__)

# Quote extraction utilities
QUOTE_PAIRS = {
    '"': '"',
    '«': '»',
    '„': '"',
    '‚': '\u2019',  # Right single quotation mark
    '\u2018': '\u2019',  # Left and right single quotation marks
    '"': '"',
    '‹': '›',
    "'": "'"
}


def extract_quotes(text: str) -> list[str]:
    """
    If text contains only properly quoted strings separated by whitespace,
    extract all substrings between the quotes. Otherwise, return the whole text.
    """
    # Build patterns that disallow closing quotes within the quoted string
    quoted_patterns = []
    for opening, closing in QUOTE_PAIRS.items():
        # Use negative character class to disallow the closing quote
        quoted_patterns.append(f'{re.escape(opening)}[^{re.escape(closing)}]*{re.escape(closing)}')

    quoted_string = '|'.join(quoted_patterns)
    quoted_sequence = rf'^(?:\s*(?:{quoted_string}))*\s*$'

    # Check if the entire text matches our pattern
    if not re.match(quoted_sequence, text):
        return [text]

    # If it does match, extract all the quoted content
    all_matches = []
    for opening, closing in QUOTE_PAIRS.items():
        # For extraction, we use a capturing group but still disallow closing quotes
        pattern = rf'{re.escape(opening)}([^{re.escape(closing)}]*){re.escape(closing)}'
        matches = re.findall(pattern, text)
        all_matches.extend(matches)

    return all_matches


def fuzzy_find_in_text(quote: str, text: str, max_distance: int) -> tuple[str, int] | None:
    """
    Find a quote in text using fuzzy matching.
    Returns (matched_text, distance) or None if not found.
    """
    # Clean the text and quote by collapsing multiple spaces and normalizing newlines
    text = re.sub(r'\s+', ' ', text)
    quote = re.sub(r'\s+', ' ', quote)

    # First check if the quote is an exact match, ignoring case
    # (because this is common and faster)
    exact_match = re.search(re.escape(quote), text, re.IGNORECASE)
    if exact_match:
        return (exact_match.group(), 0)

    # Otherwise, use fuzzy search to find the closest
    matches = find_near_matches(quote, text, max_l_dist=min(len(quote)//4, max_distance))
    if not matches:
        return None
    else:
        # Find the match with the smallest distance
        best_match = min(matches, key=lambda match: match.dist)
        return (best_match.matched, best_match.dist)


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
        output = str(data.loc[row, output_field])
        quotes = extract_quotes(output)
        input_text = '\n\n'.join(data.loc[row, input_fields])
        verified = output
        for quote in quotes:
            matched = fuzzy_find_in_text(quote, input_text, max_fuzzy_distance)

            if matched:
                (res, dist) = matched
                verified = verified.replace(quote, res)
                if dist == 0:
                    logger.debug(f'Quote "{quote[:50]}...": exact match')
                else:
                    logger.info(f'Quote "{quote[:50]}...": fuzzy match {dist} character(s) apart')
            else:
                verified = verified.replace(quote, 'QUOTE NOT FOUND')
                logger.info(f'Quote "{quote[:50]}...": NOT FOUND')

        data.loc[row, verified_field] = verified
