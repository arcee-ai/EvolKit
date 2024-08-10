import pytest
from src.generators.openai import OpenAIGenerator
from src.generators.openrouter import OpenRouterGenerator
from src.analyzers.trajectory_analyzer import TrajectoryAnalyzer

# class TestTrajectoryAnalyzer(unittest.TestCase):
def test_analyzer():
    generator = OpenRouterGenerator(model='deepseek/deepseek-chat')
    analyzer = TrajectoryAnalyzer(generator=generator)
    result = analyzer.analyze("x + 2 = 12, what is x?", ["What is x in the case of 40x^2 - 5 = 40?"])
    assert result is not None
    assert len(result) > 0