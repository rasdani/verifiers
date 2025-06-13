from datasets import load_dataset
import verifiers as vf
from verifiers.trainers.grpo_config import GRPOConfig

def grpo_defaults(run_name: str) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=1,          # ignored once max_steps is set
        bf16=True,
        max_grad_norm=0.001,
        save_strategy="no",
        logging_steps=1,
        log_on_each_node=True,
        log_completions=False,       # keep stdout clean
    )

model_name = 'willcb/Qwen2.5-0.5B-Reverse-SFT'

dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train') \
            .map(lambda x: {'question': x['text'], 'answer': x['text'][::-1]})

# ── tiny train split: exactly one batch ────────────────────────────────
train_dataset = dataset.select(range(4))
eval_dataset  = dataset.select(range(8, 40))

parser = vf.XMLParser(['think', 'answer'], answer_field='answer')
system_prompt = f"""Reverse the given text.

Respond in the following format:
{parser.get_format_str()}"""

def lcs_reward_func(completion, answer, **kwargs):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, parser.parse_answer(completion) or '', answer).ratio()

rubric = vf.Rubric(
    funcs=[lcs_reward_func, parser.get_format_reward_func()],
    weights=[1.0, 0.2]
)

vf_env = vf.SingleTurnEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric
)

args = grpo_defaults('reverse_text_bug')
args.per_device_train_batch_size   = 4   # one GPU batch
args.generation_batch_size         = 4   # ensure sampler has one batch
args.gradient_accumulation_steps   = 1
args.num_iterations                = 1
args.num_generations               = 1
args.max_steps                     = 3   # second optimisation step will hit IndexError

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args
)

trainer.train()   # -> IndexError in _gather_batch_data at step 2