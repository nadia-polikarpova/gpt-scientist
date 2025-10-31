# %%

from gpt_scientist import Scientist
from dotenv import load_dotenv
import logging

# Load environment variables from .env file (including OPENAI_API_KEY)
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("gpt_scientist").setLevel(logging.INFO)

sc = Scientist()  # Reads OPENAI_API_KEY from environment, or pass api_key parameter

sc.set_system_prompt("You are an assistant helping to analyze customer reviews.")
prompt = f'''
Analyze the review and provide:
1. The overall sentiment from 1 (very negative) to 5 (very positive),
2. A direct quote from the review that best illustrates the sentiment (3-5 words)
'''

# %%

# We're calling the synchronous version from a notebook:
# this should work by patching the event loop.

# Analyze the reviews in the CSV file and add output fields to the same file
sc.analyze_csv('reviews.csv',
               prompt,
               input_fields=['review_text'],
               output_fields=['sentiment', 'quote'])

# %%

# Check quote accuracy
sc.check_quotes_csv('reviews.csv',
                       input_fields=['review_text'],
                       output_field='quote')
