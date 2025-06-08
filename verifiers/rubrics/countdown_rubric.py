from typing import List, Dict
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
import re

class CountdownRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.parser = XMLParser(fields=["think", "answer"])
        self.reward_funcs = [
            self.countdown_reward_func,
            # self.parser.get_xml_reward_func(),
            # self.parser.get_format_reward_func()
        ]
        
    def extract_solution(self, solution_str: str) -> str | None:
        """Extract the equation from the solution string."""
        if "Assistant:" in solution_str:
            solution_str = solution_str.split("Assistant:", 1)[1]
        elif "<|im_start|>assistant" in solution_str:
            solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
        else:
            pass
            # return None
            
        solution_str = solution_str.split('\n')[-1]
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.finditer(answer_pattern, solution_str)
        matches = list(match)
        if matches:
            return matches[-1].group(1).strip()
        return None

    def validate_equation(self, equation_str: str, available_numbers: List[int]) -> bool:
        """Validate that equation only uses available numbers and each number once."""
        try:
            numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)
            return numbers_in_eq == available_numbers
        except:
            return False

    def evaluate_equation(self, equation_str: str) -> float | None:
        """Safely evaluate the arithmetic equation."""
        try:
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation_str):
                return None
            result = eval(equation_str, {"__builtins__": None}, {})
            return result
        except:
            return None

    def format_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function for formatting the answer."""
        rewards = []
        format_score = 0.1
        for completion, ans in zip(completions, answer):
            messages = self.get_assistant_messages(completion)
            if not messages:
                rewards.append(0.0)
                continue
                
            solution = messages[-1]['content']
            equation = self.extract_solution(solution)
            
            if equation is None:
                rewards.append(0.0)
                continue
                
            target = ans['target']
            numbers = ans['numbers']
            
            if not self.validate_equation(equation, numbers):
                rewards.append(format_score)
                continue
                
            result = self.evaluate_equation(equation)
            if result is None:
                rewards.append(format_score)
                continue
    
    def countdown_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function for countdown arithmetic problems."""
        format_score = 0.1
        full_score = 1.0
        
        rewards = []
        for completion, ans in zip(completions, answer):
            messages = self.get_assistant_messages(completion)
            if not messages:
                rewards.append(0.0)
                continue
                
            solution = messages[-1]['content']
            equation = self.extract_solution(solution)
            
            if equation is None:
                rewards.append(0.0)
                continue
                
            target = ans['target']
            numbers = ans['numbers']
            
            if not self.validate_equation(equation, numbers):
                rewards.append(format_score)
                continue
                
            result = self.evaluate_equation(equation)
            if result is None:
                rewards.append(format_score)
                continue
                
            rewards.append(full_score if abs(result - target) < 1e-5 else format_score)
            
        return rewards