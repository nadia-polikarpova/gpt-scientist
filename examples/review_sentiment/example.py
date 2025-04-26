from gpt_scientist import Scientist
import logging

from gpt_scientist import *

sc = Scientist() # Either add your OpenAI API key here or create a .env file with OPENAI_API_KEY
sc.logger.addHandler(logging.StreamHandler()) # We want to see the logs from the library

# Test analyzing a csv

sc.set_system_prompt("You are an assistant helping to analyze customer reviews.")
prompt = f'''
Analyze the review and provide:
1. The overall sentiment from 1 (very negative) to 5 (very positive),
2. A short justification of your assessment,
3. A direct quote from the review that best illustrates the sentiment.
'''
sc.analyze_csv('reviews.csv',
               prompt,
               input_fields=['review_text'],
               output_fields=['sentiment', 'explanation', 'quote'])

sc.check_citations_csv('reviews.csv',
                       input_fields=['review_text'],
                       output_field='quote')

