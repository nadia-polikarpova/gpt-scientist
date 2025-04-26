import re
from fuzzysearch import find_near_matches

QUOTE_PAIRS = {
    '“': '”',
    '«': '»',
    '„': '“',
    '‚': '‘',
    '‘': '’',
    '"': '"',
    '‹': '›',
    "'": "'"
}

def extract_citations(text: str) -> list[str]:
    '''
    If text contains only properly quoted strings separated by whitespace,
    extract all substrings between the quotes. Otherwise, return the whole text.
    '''
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

def fuzzy_find_in_text(citation: str, text: str, max_distance: int) -> str:
    # Clean the text and citation by collapsing multiple spaces and normalizing newlines
    text = re.sub(r'\s+', ' ', text)
    citation = re.sub(r'\s+', ' ', citation)

    # First check if the citation is an exact match, ignoring case
    # (because this is common and faster)
    exact_match = re.search(re.escape(citation), text, re.IGNORECASE)
    if exact_match:
        return (exact_match.group(), 0)

    # Otherwise, use fuzzy search to find the closest
    matches = find_near_matches(citation, text, max_l_dist=min(len(citation)//4, max_distance))
    if not matches:
        return None
    else:
        # Find the match with the smallest distance
        best_match = min(matches, key=lambda match: match.dist)
        return (best_match.matched, best_match.dist)
