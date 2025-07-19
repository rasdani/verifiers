import logging
import sys
from typing import Optional

from .types import *

try:
    import torch._dynamo # type: ignore
    torch._dynamo.config.suppress_errors = True # type: ignore
except ImportError:
    pass

try:
    from .utils.logging_utils import setup_logging
    from .utils.logging_utils import print_prompt_completions_sample
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

from .utils.data_utils import extract_boxed_answer, extract_hash_answer, load_example_dataset

from .parsers.parser import Parser
from .parsers.think_parser import ThinkParser
from .parsers.xml_parser import XMLParser

from .rubrics.rubric import Rubric
from .rubrics.judge_rubric import JudgeRubric
from .rubrics.rubric_group import RubricGroup
from .rubrics.tool_rubric import ToolRubric
from .rubrics.swe_rl_rubric import swe_rl_reward_func

from .envs.environment import Environment
from .envs.multiturn_env import MultiTurnEnv
from .envs.singleturn_env import SingleTurnEnv
from .envs.tool_env import ToolEnv
from .envs.env_group import EnvGroup

# Conditional import based on trl availability
try:
    import trl # type: ignore
    from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer
    from .trainers import GRPOTrainer, GRPOConfig, grpo_defaults, lora_defaults
    _HAS_TRL = True
except ImportError:
    _HAS_TRL = False

__version__ = "0.1.0"

# Setup default logging configuration
def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Setup basic logging configuration for the verifiers package.
    
    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Create a StreamHandler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    # Get the root logger for the verifiers package
    logger = logging.getLogger("verifiers")
    logger.setLevel(level.upper())
    logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False 

setup_logging()

__all__ = [
    "Parser",
    "ThinkParser",
    "XMLParser",
    "Rubric",
    "JudgeRubric",
    "RubricGroup",
    "ToolRubric",
    "swe_rl_reward_func",
    "Environment",
    "MultiTurnEnv",
    "SingleTurnEnv",
    "ToolEnv",
    "EnvGroup",
    "extract_boxed_answer",
    "extract_hash_answer",
    "load_example_dataset",
    "setup_logging",
]

# Add trainer exports only if trl is available
if _HAS_TRL:
    __all__.extend([
        "get_model",
        "get_tokenizer",
        "get_model_and_tokenizer",
        "GRPOTrainer",
        "GRPOConfig",
        "grpo_defaults",
        "lora_defaults",
    ])

if _HAS_RICH:
    __all__.extend([
        "print_prompt_completions_sample",
    ])