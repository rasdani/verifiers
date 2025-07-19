from datasets import load_dataset
import verifiers as vf
from verifiers.parsers.swe_rl_parser import SweRlParser
from verifiers.rubrics.swe_rl_rubric import SweRlRubric

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

rubric = SweRlRubric(parser=parser)

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