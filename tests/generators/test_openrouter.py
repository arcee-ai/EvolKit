import unittest
import pytest
from src.generators.openrouter import OpenRouterGenerator

def test_generate():
    generator = OpenRouterGenerator()
    result = generator.generate("Test prompt")
    assert result is not None
    assert len(result) > 0