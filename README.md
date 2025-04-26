# gpt_scientist

[![PyPI version](https://badge.fury.io/py/gpt-scientist.svg)](https://badge.fury.io/py/gpt-scientist)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🔵 [Читать на русском (Read in Russian)](README_ru.md)

`gpt_scientist` is a lightweight Python library for processing tabular data stored in Google Sheets (or CSV files) using [OpenAI](https://openai.com/) models (like GPT-4o, GPT-4o-mini, etc).

The library is designed primarily for **social science researchers** and other users without extensive programming experience who wants to run AI-based textual analysis over tabular data with just a few lines of Python code.

The library is best used in [Google Colab](https://colab.research.google.com/) for processing Google Sheets.
However, it can also be used locally with CSV files.

**Feedback and Collaboration**

If you use `gpt_scientist` for your project, we would love to hear about it!
Your feedback helps us improve the library and better support real-world research and activist work.

Feel free to open an [issue](https://github.com/nadia-polikarpova/gpt-scientist/issues) on GitHub, or reach out by [email](mailto:npolikarpova@ucsd.edu).


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

# Set the system prompt that describes the general capabilities you need:
sc.set_system_prompt("You are an assistant helping to analyze customer reviews.")
# Or, if the system prompt is long (e.g. contains the theoretical frame of your research study), you can load it from a google doc:
# sc.load_system_prompt_from_google_doc('your-google-doc-id')


# Define the task prompt
prompt = "Analyze the review and provide the overall sentiment from 1 (very negative) to 5 (very positive), together with a short explanation."

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

> *Important:*
> Google Sheets can *only* be accessed from Google Colab, so you need to run this code in a Colab notebook.
> To use the library locally with CSV files, call `sc.analyze_csv(...)` instead of `sc.analyze_google_sheet(..)` (see [example](https://github.com/nadia-polikarpova/gpt-scientist/blob/main/examples/review_sentiment/example.py)).

**Notes**
- The library will write to the sheet as it goes, so even if you stop the execution, you will have the results for the rows that were already processed.
- The library will also show you the cost of the API calls so far, so you can keep track of your spending (only for those models whose price it knows).
- If the output columns already exist, the library will skip those rows where the outputs are already filled in (unless you specify `overwrite=True`).

## Advanced Features

**Document Processing**

Often, you may want to analyze not just short text fields, but longer documents — for example, interview transcripts.
If your input cell in the Google Sheet contains a link to a Google Doc, the library will automatically open the document and feed its full content to the language model.

> *Important:*
> If Google Sheets automatically converted your link into a "smart chip" (those clickable document previews), the library will not recognize it.
> You must ensure the spreadsheet cell contains a plain hyperlink, not a chip.

**Quote Verification**

One of the useful applications of GPT-based analysis is extracting quotes on specific topics from documents.
The library includes a helper function to verify extracted quotes against the original source text.

You can call this *after* your call to `analyze_google_sheet`:

```python
sc.check_citations_google_sheet(
    sheet_key='your-google-sheet-id',
    input_fields=['transcript'],
    output_field='gpt_extracted_quote',
    rows='4:5'  # optional: specify which rows to process
)
```

This function will:
- Create a new column 'gpt_extracted_quote_verified' (if it doesn't exist yet).
- For each row, search for the extracted quote in any of the input fields (in this case, in the transcript).
- If it finds an exact or approximate match, it will write the exact version of the quote into 'gpt_extracted_quote_verified'.
- Otherwise, it will insert 'CITATION NOT FOUND'.

This helps verify that the quotes generated by the model actually correspond to the original document, improving the reliability of automated extraction.

## Other Settings

**Write results to a new sheet**

If you don't want to modify the input sheet, add `in_place=False` to the parameters of your `analyze_google_sheet`. This will create a new worksheet for the output.

**Select a different worksheet**

If your input data is not on the first sheet, add `worksheet_index=n` (e.g. `worksheet_index=1`) to the parameters of your `analyze_google_sheet`.
Indexing starts from 0, so 1 is the second sheet.

**Change the model**

The default model is `gpt-4o-mini`: it is cheap and good enough for most tasks.
But you can use any [model](https://platform.openai.com/docs/models) that is enabled for your OpenAI API key.
Just make this call before your call to `analyze_google_sheet`:

```python
sc.set_model('gpt-4o')
```

**Load a system prompt from a file**

```python
sc.load_system_prompt_from_file('content/system.txt')
```

**Limit the maximum length of model responses**

```python
sc.set_max_tokens(100)
```

This protects you from excessive costs if the model starts generating very long outputs.
However, if you need longer answers, setting this too low may cut them off.
By default, the response length is unlimited.

**Control response diversity**

```python
sc.set_top_p(0.5)
```

- Lower values (e.g., 0.1) make the model's responses more deterministic.
- Higher values (e.g., 1.0) make them more varied (and possibly less coherent).

The default is 0.3, balancing predictability and creativity.

**Adjust retries and batch sizes**

```python
sc.set_num_results(5)
sc.set_num_retries(20)
```

- `set_num_retries` controls how many times the library retries after a bad response (default: 10).
- `set_num_results` controls how many completions are requested at once — useful if input size is much bigger than output size, and the reponses are often bad.

**Customize token pricing**

```python
sc.set_pricing({'gpt-3.5-turbo': {'input': 1.5, 'output': 2}})
```

If you are using a model not included in the built-in pricing table, or if token prices have changed, you can define your own (in dollars per million tokens)

## Acknowledgements

This library has been created as a result of my collaboration with the [Hannah Arendt Research Center](https://www.tharesearch.center/en), and the idea is due to the Center's founder, Mariia Vasilevskaia.
