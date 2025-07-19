from datasets import load_dataset
from typing import Dict, List, Tuple, Union
import json
import cydifflib
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

import verifiers as vf
from verifiers.parsers.swe_rl_parser import SweRlParser
from verifiers.types import ChatMessage, Info
from verifiers.rubrics.utils.swe_rl_utils import FileDiff, extract_minimal_patch


"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager
# CUDA_VISIBLE_DEVICES=0 vf-vllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --enforce-eager

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/swe_rl_example.py
"""


# model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
model_name = 'willcb/Qwen3-0.6B'
dataset = load_dataset('rasdani/SkyRL-v0-293-data-oracle-8k-context', split='train').map(lambda x: {'question': x['text'], 'answer': x['text'][::-1]})
TRAIN_SIZE = 100
EVAL_SIZE = 10
train_dataset = dataset.select(range(TRAIN_SIZE)) # type: ignore
eval_dataset = dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)) # type: ignore

# parser = vf.XMLParser(['think', 'answer'], answer_field='answer')
parser = SweRlParser()

def create_patched_file_context(
    edited_file_context: Dict[str, str],
    file_diffs: List[FileDiff],
) -> Dict[str, str]:
    """Create patched file context from edited file context and file diffs."""
    patch_dict = {}
    for file_diff in file_diffs:
        file_path = file_diff.header.file.path
        # Update the file content with the edited file context
        file_diff.new_file_content = edited_file_context.get(file_path, "")
        file_diff.generate_hunks_from_content()
        predicted_patch = file_diff.get_patch()
        if predicted_patch.strip():
            patch_dict[file_path] = predicted_patch
    return patch_dict


def get_unidiff_from_patched_file_context(
    patched_file_context: Dict[str, str]
) -> str:
    """Convert patched file context to unified diff format."""
    try:
        patches = list(patched_file_context.values())
        if not patches:
            return ""
        first_patch = patches.pop(0)
        patch_set = PatchSet(first_patch)
        for patch in patches:
            patch_set.extend(PatchSet(patch))
        return str(patch_set)
    except UnidiffParseError:
        return ""


def apply_edits(
    file_context: Dict[str, str], 
    edits: Dict[str, List[Tuple[str, str]]]
) -> Dict[str, str] | None:
    """Apply search/replace edits to file context."""
    edited_file_context = {}
    for file_path, file_edits in edits.items():
        edited_file_content = f"\n{file_context.get(file_path, '')}"
        for search_str, replace_str in file_edits:
            if search_str not in edited_file_content:
                return None
            edited_file_content = edited_file_content.replace(f"\n{search_str}", f"\n{replace_str}")
        edited_file_context[file_path] = edited_file_content.lstrip("\n")
    return edited_file_context


def score_patch(
    pred_patch: str, 
    oracle_patch: str
) -> float:
    """Score predicted patch against oracle patch using sequence matching."""
    try:
        score = cydifflib.SequenceMatcher(
            None,
            a=pred_patch,
            b=oracle_patch,
            autojunk=False,
        ).ratio()
        return score
    except Exception:
        return 0.0


def score_swe_rl(
    solution_str: str, 
    file_context: Dict[str, str], 
    file_diffs: List[FileDiff], 
    oracle_patch: str
) -> float:
    """Score a solution string against oracle patch using SWE-RL methodology."""
    edits = SweRlParser().parse(solution_str)
    if len(edits) == 0:
        return 0.0
    edited_file_context = apply_edits(file_context, edits)
    if edited_file_context is None:
        return 0.0
    patched_file_context = create_patched_file_context(edited_file_context, file_diffs)
    pred_patch = get_unidiff_from_patched_file_context(patched_file_context)
    min_pred_patch = extract_minimal_patch(pred_patch)
    min_oracle_patch = extract_minimal_patch(oracle_patch)
    return score_patch(min_pred_patch, min_oracle_patch)


def compute_swe_rl_score(
    solution_str: str, 
    ground_truth: str, 
    extra_info: Dict[str, str] = None
) -> float:
    """Compute SWE-RL score for a solution."""
    try:
        parsed_commit_content = json.loads(extra_info["parsed_commit_content"])
        file_diffs = parsed_commit_content.get("file_diffs")
        file_diffs = [FileDiff(**file_diff) for file_diff in file_diffs]
        
        file_context = json.loads(extra_info["file_context"])
        
        return score_swe_rl(
            solution_str=solution_str, 
            file_context=file_context, 
            file_diffs=file_diffs, 
            oracle_patch=ground_truth
        )
    except Exception:
        return 0.0


def swe_rl_reward_func(
    completion: Union[str, List[ChatMessage]],
    answer: str,
    info: Info = {},
    **kwargs
) -> float:
    """
    Args:
        completion: The model's completion/solution
        answer: The ground truth oracle patch
        info: Additional information including file context and commit data
        **kwargs: Additional keyword arguments
    
    Returns:
        float: Score between -1.0 and 1.0, where 1.0 is perfect match
    """
    # Convert completion to string if it's a list of chat messages
    if isinstance(completion, list):
        completion_str = completion[-1]["content"]
    else:
        completion_str = completion
    
    return compute_swe_rl_score(
        solution_str=completion_str,
        ground_truth=answer,
        extra_info=info
    )
rubric = vf.Rubric(
    funcs=[swe_rl_reward_func, parser.get_format_reward_func()],
    weights=[1.0, 1.0]
)

vf_env = vf.SingleTurnEnv(
    dataset=train_dataset, # type: ignore
    eval_dataset=eval_dataset, # type: ignore
    parser=parser,
    rubric=rubric
)
args = vf.grpo_defaults(run_name='swe_rl_example')
args.per_device_train_batch_size = 12
args.num_generations = 12
args.gradient_accumulation_steps = 8
args.max_steps = 100
args.eval_strategy = 'steps'
args.eval_steps = 2

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    #peft_config=vf.lora_defaults(),
    args=args,
)
trainer.train()