from gpt_scientist import Scientist
import logging

from gpt_scientist import *

sc = Scientist() # Either add your OpenAI API key here or create a .env file with OPENAI_API_KEY
sc.logger.addHandler(logging.StreamHandler()) # We want to see the logs from the library

sc.set_system_prompt("You are an assistant helping to analyze customer reviews.")
prompt = f'''
Analyze the review and provide:
1. The overall sentiment from 1 (very negative) to 5 (very positive),
2. A direct quote from the review that best illustrates the sentiment (3-5 words)
'''

# Analyze the reviews in the CSV file and add output fields to the same file
sc.analyze_csv('reviews.csv',
               prompt,
               input_fields=['review_text'],
               output_fields=['sentiment', 'quote'])

# Check quote accuracy
sc.check_quotes_csv('reviews.csv',
                       input_fields=['review_text'],
                       output_field='quote')

