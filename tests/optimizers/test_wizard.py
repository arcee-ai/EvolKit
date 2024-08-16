import pytest
from src.generators import OpenRouterGenerator
from src.optimizers.wizard_optimizer import WizardOptimizer
from src.analyzers.trajectory_analyzer import TrajectoryAnalyzer
from src.evolvers.recurrent_evolver import RecurrentEvolver, INITIAL_EVOLVE_METHOD
from src.evaluator.failure_detector_evaluator import FailureDetectorEvaluator
from src.utils import parse_steps

dev_set = ['Write a python function to perform bubble sort', 'Write a letter to my headmaster asking for a day off.']
    
@pytest.mark.asyncio
async def test_wizard_optimizer():
    generator = OpenRouterGenerator(model='openai/chatgpt-4o-latest')
    evolver = RecurrentEvolver(generator)
    analyzer = TrajectoryAnalyzer(generator)
    detector = FailureDetectorEvaluator()
    optimizer = WizardOptimizer(generator, detector)
    
    init_instruction = "What is the sum of 2 and 2, and please provide a detailed explanation of how you arrived at the answer in the form of a mathematical equation? Furthermore, imagine you have 2 apples and receive 2 more, how many apples do you have now?"
    
    # Assuming evolver.evolve and analyzer.analyze are also async
    evolved_instructions = evolver.evolve(init_instruction, INITIAL_EVOLVE_METHOD, n=5)
    

    feedbacks = analyzer.analyze(INITIAL_EVOLVE_METHOD, evolved_instructions)
    
    optimized_method, methods = await optimizer.optimize(INITIAL_EVOLVE_METHOD.format(instruction=init_instruction), feedback=feedbacks, evolver=evolver, development_set=dev_set)