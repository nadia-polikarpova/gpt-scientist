"""Configuration management for gpt_scientist."""

import json
import logging
import re
import requests
import importlib.resources

logger = logging.getLogger(__name__)

# Github URL for the default pricing table
PRICING_URL = "https://raw.githubusercontent.com/nadia-polikarpova/gpt-scientist/main/src/gpt_scientist/model_pricing.json"

# Index of the first non-header row in google-sheet indexing
GSHEET_FIRST_ROW = 2

# Regular expression pattern for Google doc URL
GOOGLE_DOC_URL_PATTERN = re.compile(r'https://docs.google.com/document/d/(?P<doc_id>[^/]+)/.*')

# Default model
DEFAULT_MODEL = 'gpt-4o-mini'

# Default embedding model
DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'


def fetch_pricing() -> dict:
    """
    Fetch the pricing table from GitHub or fall back to the local file.
    Returns a dictionary mapping model names to pricing info.
    """
    try:
        # Try to fetch the pricing table from github
        resp = requests.get(PRICING_URL, timeout=2)
        if resp.ok:
            logger.info(f"Fetched pricing table from {PRICING_URL}")
            return resp.json()
    except requests.RequestException:
        pass

    # Otherwise: read the pricing table from the local file
    try:
        with importlib.resources.files("gpt_scientist").joinpath("model_pricing.json").open("r") as f:
            pricing = json.load(f)
            logger.info("Loaded pricing table from the local file.")
            return pricing
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        logger.warning(f"Could not load the pricing table: {e}.")
        return {}


def is_embedding_model(model: str, pricing: dict) -> bool:
    """Return True if the given model is an embedding model."""
    return model in pricing and 'embedding' in pricing[model] and pricing[model]['embedding']
