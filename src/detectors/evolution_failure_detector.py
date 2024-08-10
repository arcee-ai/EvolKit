class EvolutionFailureDetector:
    def __init__(self):
        self.stagnant_keywords = ["understood", "thank you", "noted", "got it", "okay", "alright"]
        self.insufficient_keywords = ["sure", "certainly", "of course", "happy to help"]
        self.loss_keywords = ["please provide", "need more information", "could you clarify", "what exactly"]

    def is_failure(self, response):
        """
        Detect if the evolution has failed based on the original instruction,
        evolved instruction, and the response.
        """
        response = response.lower()
        
        # 1. Check for stagnant complexity
        if self._is_stagnant_complexity(response):
            return True
        
        # 2. Check for insufficient qualification
        if self._is_insufficient_qualification(response):
            return True
        
        # 3. Check for loss of key information
        if self._is_loss_of_information(response):
            return True
        
        return False

    def _is_stagnant_complexity(self, response,):
        # Check for stagnant complexity keywords
        if any(keyword in response for keyword in self.stagnant_keywords) and response.endswith("?"):
            return True
        
        return False

    def _is_insufficient_qualification(self, response):
        # Check for insufficient qualification keywords
        if any(keyword in response.lower() for keyword in self.insufficient_keywords) and response.endswith("?"):
            return True
        
        # Check if the response is asking for clarification
        if "what do you mean" in response.lower() or "could you explain" in response.lower():
            return True
        
        return False

    def _is_loss_of_information(self, response):
        # Check for loss of information keywords
        if any(keyword in response.lower() for keyword in self.loss_keywords):
            return True
        
        return False

    def calculate_failure_rate(self, responses):
        """
        Calculate the failure rate for a set of evolved instructions and their responses.
        """
        total_failures = sum(self.is_failure(response) 
                             for response in responses)
        return total_failures / len(responses)

    def detect(self, methods, development_set):
        """
        Compare different evolving methods and return the one with the lowest failure rate.
        """
        best_method = None
        lowest_failure_rate = float('inf')

        for method in methods:
            evolved_instructions, responses = self.simulate_evolution(method, development_set)
            failure_rate = self.calculate_failure_rate(development_set, evolved_instructions, responses)
            
            if failure_rate < lowest_failure_rate:
                lowest_failure_rate = failure_rate
                best_method = method

        return best_method, lowest_failure_rate

    def simulate_evolution(self, method, instructions):
        """
        Simulate the evolution of instructions using a given method.
        This is a placeholder and should be implemented based on your specific evolution logic.
        """
        # Placeholder implementation
        evolved_instructions = [f"Evolved: {instruction}" for instruction in instructions]
        responses = [f"Response to: {evolved}" for evolved in evolved_instructions]
        return evolved_instructions, responses