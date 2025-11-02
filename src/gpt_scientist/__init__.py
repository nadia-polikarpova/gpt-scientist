"""Top-level package for GPT Scientist."""

__author__ = """Nadia Polikarpova"""
__email__ = 'nadia.polikarpova@gmail.com'
__version__ = '0.1.0'

from .scientist import Scientist
from .stats import JobStats

__all__ = ['Scientist', 'JobStats']
