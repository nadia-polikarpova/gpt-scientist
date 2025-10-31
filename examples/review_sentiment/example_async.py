"""Async example showing how to use the async API."""
import asyncio
import logging
from gpt_scientist import Scientist
from dotenv import load_dotenv

# Load environment variables from .env file (including OPENAI_API_KEY)
load_dotenv()

# logging.basicConfig(level=logging.WARNING)
# logging.getLogger("gpt_scientist").setLevel(logging.INFO)


async def main():
    """Async main function to demonstrate the async API."""
    sc = Scientist()  # Reads OPENAI_API_KEY from environment, or pass api_key parameter

    # sc.set_model("text-embedding-3-small")

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

    assert sc.stats is not None, "Stats should be populated after analysis"
    # Print final cost report
    print(f"Input tokens: {sc.stats.input_tokens}, Output tokens: {sc.stats.output_tokens}, Model: {sc.stats.model}")
    print(f"Input cost: ${sc.stats.current_cost()['input']:.2f}, Output cost: ${sc.stats.current_cost()['output']:.2f}")

    # Check quote accuracy using async version
    await sc.check_quotes_csv_async('reviews.csv',
                                    input_fields=['review_text'],
                                    output_field='quote')

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
