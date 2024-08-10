import pytest
from src.generators import OpenAIGenerator, OpenRouterGenerator
from src.evolvers.recurrent_evolver import RecurrentEvolver
from src.utils import parse_steps

def test_recurrent_evolver():
    generator = OpenRouterGenerator(model='deepseek/deepseek-chat')
    evolver = RecurrentEvolver(generator=generator)
    results = evolver.evolve(instruction="What is the sum of 2 and 2, and please provide a detailed explanation of how you arrived at the answer in the form of a mathematical equation? Furthermore, imagine you have 2 apples and receive 2 more, how many apples do you have now?", n=2)
    parsed_results = []
    for result in results:
        parsed_results.append(parse_steps(result))
    
    # print(parsed_results)
        
    assert len(parsed_results) == 2
    assert parsed_results[0][-1]['step_name'] == "Finally Rewritten Instruction", f"Unexpected final step name for result 1: {parsed_results[0][-1]['step_name']}"
    assert parsed_results[1][-1]['step_name'] == "Finally Rewritten Instruction", f"Unexpected final step name for result 2: {parsed_results[1][-1]['step_name']}"
    