# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gpt_scientist is a Python library for processing tabular data (Google Sheets or CSV files) using OpenAI models. It's designed for social science researchers to run AI-based textual analysis with minimal code, primarily in Google Colab.

## Build & Development Commands

```bash
# Build the package
python -m build

# Upload to PyPI (requires .env with credentials)
make upload

# Clean build artifacts
make clean

# Install for development
pip install -e .
```

## Architecture

### Core Components

- **`Scientist`** (`scientist.py`): Main orchestrator class that users interact with. Holds all configuration (model, prompts, parallelism settings, pricing) and provides sync/async methods for analyzing data.

- **`LLMClient`** (`llm/client.py`): Wrapper around OpenAI's AsyncOpenAI client. Handles prompting, response parsing, retries, and embedding generation. Supports both regular JSON mode and structured outputs.

- **`analyze_data`** (`processors/core.py`): Core async function that orchestrates parallel processing of dataframe rows using asyncio TaskGroups with producer-consumer pattern (row_queue → workers → output_queue → writer).

### Data Flow

1. User calls `analyze_google_sheet()` or `analyze_csv()` on a `Scientist` instance
2. Data is loaded into a pandas DataFrame
3. `analyze_data()` spawns parallel worker coroutines that:
   - Pull row indices from `row_queue`
   - Call `LLMClient.get_response()` to process each row
   - Push results to `output_queue`
4. A single `writer` coroutine batches results and writes back to the source

### Key Patterns

- **Async-first with sync wrappers**: All core functionality is async. Sync methods use `run_async()` utility to handle event loop management (including `nest_asyncio` for Colab).
- **Parallel row processing**: Configurable via `parallel_rows` (default 100). Workers process rows concurrently while a single writer batches output.
- **Google Sheets I/O**: Uses `asyncio.to_thread()` to wrap blocking gspread calls. Automatically follows Google Doc URLs in input cells.
- **Quote verification**: `check_quotes()` uses fuzzy matching (fuzzysearch library) to verify extracted quotes exist in source documents.

### Module Structure

- `processors/`: Data source handlers (sheets.py for Google Sheets, csv.py for CSV files)
- `llm/`: OpenAI client and prompt construction
- `verification/`: Quote verification with fuzzy matching
- `config.py`: Model pricing, defaults, regex patterns
- `stats.py`: Token/cost tracking and reporting

## Important Implementation Details

- Google Sheets functionality only works in Google Colab (detected via `IN_COLAB` flag)
- Row indexing: Google Sheets uses 2-based indexing (row 2 = first data row), internally converted to 0-based
- Pricing table is fetched from GitHub, with local fallback in `model_pricing.json`
- Output fields are automatically created in the dataframe/sheet if they don't exist
