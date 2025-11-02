"""Data models for gpt_scientist."""

import logging
logger = logging.getLogger(__name__)


class JobStats:
    '''Statistics for a table processing job.'''

    def __init__(self, model: str, pricing: dict):
        '''Initialize JobStats with optional pricing information.'''
        self.model = model
        self.pricing = pricing
        self.rows_processed = 0
        self.errors = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def current_cost(self) -> dict:
        '''Return the cost corresponding to the current number of input and output tokens.'''
        current_pricing = self.pricing.get(self.model, {})
        input_cost = current_pricing.get('input', 0) * self.input_tokens / 1e6
        output_cost = current_pricing.get('output', 0) * self.output_tokens / 1e6
        return {'input': input_cost, 'output': output_cost}

    def report_cost(self):
        cost = self.current_cost()
        logger.info(f"PROCESSED {self.rows_processed} ROWS. TOTAL_COST: ${cost['input']:.4f} + ${cost['output']:.4f} = ${cost['input'] + cost['output']:.4f}")

    def log_rows(self, rows: int, input_tokens: int, output_tokens: int):
        '''Add the tokens used in the current row to the total and log the cost.'''
        self.rows_processed += rows
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self.rows_processed % 10 == 0:
            self.report_cost()

    def log_error(self):
        '''Increment the error counter.'''
        self.errors += 1
