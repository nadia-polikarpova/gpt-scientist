"""Async example showing how to use the async API."""
import asyncio
import logging
from gpt_scientist import Scientist
from dotenv import load_dotenv

# Load environment variables from .env file (including OPENAI_API_KEY)
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("gpt_scientist").setLevel(logging.INFO)


async def main():
    """Async main function to demonstrate the async API."""
    sc = Scientist()  # Reads OPENAI_API_KEY from environment, or pass api_key parameter

    sc.set_system_prompt("You are an assistant helping to analyze customer reviews.")
    prompt = '''
Analyze the review and provide:
1. The overall sentiment from 1 (very negative) to 5 (very positive),
2. A direct quote from the review that best illustrates the sentiment (3-5 words)
'''

    # Use the async version directly for better control in async contexts
    await sc.analyze_csv_async('reviews.csv',
                               prompt,
                               input_fields=['review_text'],
                               output_fields=['sentiment', 'quote'])

    # Check quote accuracy
    # Note: check_quotes_csv is still sync as it's a helper function
    sc.check_quotes_csv('reviews.csv',
                        input_fields=['review_text'],
                        output_field='quote')


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
