import pytest
from src.generators.openai import OpenAIGenerator

def test_generate():
    generator = OpenAIGenerator()
    result = generator.generate("Test prompt")
    assert result is not None
    assert len(result) > 0