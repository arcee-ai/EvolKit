import pytest
from src.generators import OpenAIGenerator, OpenRouterGenerator
from src.optimizers.wizard_optimizer import WizardOptimizer
from src.analyzers.trajectory_analyzer import TrajectoryAnalyzer
from src.evolvers.recurrent_evolver import RecurrentEvolver, INITIAL_EVOLVE_METHOD
from src.detectors.evolution_failure_detector import EvolutionFailureDetector
from src.utils import parse_steps

dev_set = ['Write a python function to perform bubble sort', 'Write a letter to my headmaster asking for a day off.']

# def test_wizard_optimizer():
#     generator = OpenRouterGenerator(model='openai/gpt-4o-2024-08-06')
#     evolver = RecurrentEvolver(generator)
#     analyzer = TrajectoryAnalyzer(generator)
#     detector = EvolutionFailureDetector()
#     optmizer = WizardOptimizer(generator, detector)
    
#     init_instruction = "What is the sum of 2 and 2, and please provide a detailed explanation of how you arrived at the answer in the form of a mathematical equation? Furthermore, imagine you have 2 apples and receive 2 more, how many apples do you have now?"
    
#     evolved_instructions = evolver.evolve(init_instruction, INITIAL_EVOLVE_METHOD, n=2)
#     feedbacks = analyzer.analyze(INITIAL_EVOLVE_METHOD, evolved_instructions)
    
#     optimized_method, methods = optmizer.optimize(INITIAL_EVOLVE_METHOD.format(instruction=init_instruction), feedback=feedbacks, evolver=evolver, development_set=dev_set)
    
#     # for i, method in enumerate(methods):
#     #     print(f"\033[93mMethod {i+1}:\n\033[0m")
#     #     steps = parse_steps(method)
#     #     for step in steps:
#     #         print(step)
            
#     # print("The best method is: ")
#     # print(optimized_method)
    
@pytest.mark.asyncio
async def test_wizard_optimizer():
    generator = OpenRouterGenerator(model='meta-llama/llama-3.1-405b-instruct')
    evolver = RecurrentEvolver(generator)
    analyzer = TrajectoryAnalyzer(generator)
    detector = EvolutionFailureDetector()
    optimizer = WizardOptimizer(generator, detector)
    
    init_instruction = "What is the sum of 2 and 2, and please provide a detailed explanation of how you arrived at the answer in the form of a mathematical equation? Furthermore, imagine you have 2 apples and receive 2 more, how many apples do you have now?"
    
    # Assuming evolver.evolve and analyzer.analyze are also async
    evolved_instructions = evolver.evolve(init_instruction, INITIAL_EVOLVE_METHOD, n=5)
    

    feedbacks = analyzer.analyze(INITIAL_EVOLVE_METHOD, evolved_instructions)
    
    optimized_method, methods = await optimizer.optimize(INITIAL_EVOLVE_METHOD.format(instruction=init_instruction), feedback=feedbacks, evolver=evolver, development_set=dev_set)

    
    # for method in methods:
    #     print(f'method: {method}')
    
    # print(f"Optmized Method: {optimized_method}")