from typing import List, Callable, Dict, Tuple
import re
from collections import defaultdict

from verifiers import (
    ChatMessage,
    Parser,
)


EDITS_PATTERN = re.compile(
    r"```.*?\n"
    r"### (.*)\n"
    r"<<<<<<< SEARCH\n"
    r"([\s\S]*?)\n"
    r"=======\n"
    r"([\s\S]*?)\n"
    r">>>>>>> REPLACE\n"
    r"```"
)

# TODO: consider subclassing ThinkParser
class SweRlParser(Parser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_edits(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """Parse SEARCH/REPLACE edits from input text."""
        edits = defaultdict(list)
        matches = EDITS_PATTERN.finditer(text)
        for match in matches:
            file_path = match.group(1)
            search_content = match.group(2)
            replace_content = match.group(3)
            edits[file_path].append((search_content, replace_content))
        return edits

    def parse(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return self.parse_edits(text)

    def get_format_reward_func(self) -> Callable:
        def format_reward_func(completion: List[ChatMessage], **kwargs) -> float:
            messages = self.get_assistant_messages(completion)
            scores = []
            for msg in messages:
                text = msg["content"]
                try:
                    parsed_edits = self.parse(text)
                    if len(parsed_edits) > 0:
                        scores.append(0.0)
                    else:
                        scores.append(-1.0)
                except Exception as e:
                    scores.append(-1.0)
            return sum(scores) / len(scores)
        return format_reward_func