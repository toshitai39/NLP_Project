from unsloth import FastLanguageModel
import torch
max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    fix_tokenizer=False
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    # titles = examples["title"]
    texts  = examples["context"]
    outputs = []
    for text in texts:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = text + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }
# pass
from datasets import load_dataset

dataset = load_dataset('csv', data_files='context_new.csv')

# We select 1% of the data to make training faster!
dataset = dataset["train"]

dataset = dataset.map(formatting_prompts_func, batched = True,)
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 1024,
        gradient_accumulation_steps = 8,

        # Use warmup_ratio and num_train_epochs for longer runs!
        # max_steps = 120,
        # warmup_steps = 10,
        warmup_ratio = 0.1,
        num_train_epochs = 50,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_pretrain",
        report_to = "none", # Use this for WandB etc
    ),
)
trainer_stats = trainer.train()