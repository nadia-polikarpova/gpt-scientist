# gpt_scientist

`gpt_scientist` is a lightweight Python library for processing tabular data stored in Google Sheets (or CSV files) using [OpenAI](https://openai.com/) models (like GPT-4o, GPT-4o-mini, etc).

The library is designed primarily for social science researchers, human rights analysts, and anyone who wants to run AI-based textual analysis over tabular data with just a few lines of Python code.

The library is best used in [Google Colab](https://colab.research.google.com/) for processing Google Sheets.
However, it can also be used locally with CSV files.

## Installation

```bash
pip install gpt-scientist
```

## Quick Example

```python
from gpt_scientist import Scientist

# Create a Scientist
sc = Scientist(api_key='YOUR_OPENAI_API_KEY')
# (or set it via the OPENAI_API_KEY environment variable)

# Load a system prompt from a Google Doc
sc.load_system_prompt_from_google_doc('your-google-doc-id')
# or define it inline using sc.set_system_prompt('Your system prompt here')

# Define the task prompt
prompt = "Analyze this customer review and return a sentiment score (positive/neutral/negative) and a short explanation.\n\nReview:"

# Analyze a Google Sheet
sc.analyze_google_sheet(
    sheet_key='your-google-sheet-id', # a sheet key is the part of the URL after /d/ and before the next /
    prompt=prompt,
    input_fields=['review_text'],
    output_fields=['sentiment', 'explanation'],
    rows='2:12',  # optional: analyze only rows 2 to 12 in the sheet
)
```

This will:
- Read the first worksheet from your Google Sheet
- Create the `sentiment` and `explanation` columns in that sheet if they don't exist
- For each row in the specified range (2 to 12):
  - Read the content of the `review_text` column
  - Call the OpenAI model with the prompt and the review text
  - Write the results (sentiment and explanation) back into the sheet

**Notes**
- Google Sheets can *only* be accessed from Google Colab, so you need to run this code in a Colab notebook.
- To use the library locally with CSV files, call `sc.analyze_csv(...)` instead of `sc.analyze_google_sheet(..)`.
- The library will write to the sheet as it goes, so even if you stop the execution, you will have the results for the rows that were already processed.
- The library will also show you the cost of the API calls so far, so you can keep track of your spending (only for those models whose price it knows).
- If the output columns already exist, the library will skip those rows where the outputs are already filled in (unless you specify `overwrite=True`).


**Advanced Features**

- The default model is `gpt-4o-mini`: it is cheap and good enough for most tasks. But you can use any [model](https://platform.openai.com/docs/models) that is enabled for your OpenAI API key by calling e.g. `sc.set_model('gpt-4o')`.
- *Document processing:* if a spreadsheet cell contains a link to a Google Doc, the document’s full text will be automatically loaded and analyzed.
- *Quote verification:* you can validate whether extracted quotes from documents actually match their sources by calling `sc.check_citations_google_sheet(...)`.