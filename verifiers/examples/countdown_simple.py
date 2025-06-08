import verifiers as vf
from verifiers.rubrics import CountdownRubric

# model_name = "Qwen/Qwen2.5-Math-1.5B"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "Qwen/Qwen2.5-0.5B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.MathEnv(dataset="countdown", system_prompt=None, few_shot=[], fields=["think", "answer"])
vf_env.rubric = CountdownRubric()
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

run_name = "countdown_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=2)
training_args.max_steps = 1000
training_args.beta = 0.001
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
