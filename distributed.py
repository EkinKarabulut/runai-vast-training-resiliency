import os
import re
import logging
import torch
import transformers
import warnings
from trl import SFTConfig, SFTTrainer
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import load_dataset
from peft import LoraConfig, TaskType

# -------------------- LOGGING CONFIGURATION --------------------

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message=".*`evaluation_strategy` is deprecated.*",
    category=FutureWarning,
    module="transformers.training_args"
)

warnings.filterwarnings(
    "ignore",
    message=".*torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    category=UserWarning,
    module="torch._dynamo"
)

warnings.filterwarnings(
    "ignore",
    message=".*MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization.*",
    category=UserWarning,
    module="bitsandbytes.autograd._functions"
)

# ----------------------------------------------------------------------------

class CustomCheckpointCallback(TrainerCallback):
    """
    A TrainerCallback that prints a single line whenever a checkpoint is saved.
    """
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        print(f"Checkpointing to the directory {ckpt_dir}")

def find_latest_checkpoint(output_dir):
    ckpts = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full):
            m = re.match(r"^checkpoint-(\d+)$", name)
            if m:
                ckpts.append((int(m.group(1)), full))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0], reverse=True)
    return ckpts[0][1]


def initialize_distributed():

    print("Initializing the DDP environment") # DDP will be configured only if there are multiple GPUs. Otherwise it will fallback to single GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))  
    torch.cuda.set_device(local_rank)
    print(f"Initialized distributed training on local rank: {local_rank}")
    return local_rank


def load_model(model_name: str, local_rank):

    print("Starting to load the model (8-bit)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        trust_remote_code=True,
        device_map={'':local_rank}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # ensure pad token is defined
    
    return model, tokenizer


def load_and_prepare_data(tokenizer, local_rank):

    raw = load_dataset("poornima9348/finance-alpaca-1k-test")
    split = raw["test"].train_test_split(test_size=0.2, shuffle=True)
    train_dataset = split["train"]
    test_dataset = split["test"]

    # Drop any unused columns
    for ds in (train_dataset, test_dataset):
        for col in ["input", "text"]:
            if col in ds.column_names:
                ds = ds.remove_columns(col)

    # Build prompt as "instruction + output"
    def prompt_builder(example):
        return {"text": example["instruction"] + example["output"]}

    # Tokenize the combined text (truncate/pad to max_length=512)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    train_dataset = train_dataset.map(prompt_builder)
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(prompt_builder)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Drop the original instruction/output columns now that we have tokenized text
    train_dataset = train_dataset.remove_columns(["instruction", "output"])
    test_dataset = test_dataset.remove_columns(["instruction", "output"])

    print("Datasets are tokenized and formatted")
    return train_dataset, test_dataset


def configure_lora(model, ckpt_dir):
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
    )
    return lora_config


def train_model(model, lora_config, train_dataset, test_dataset, tokenizer, local_rank, checkpoint_dir):

    #print("LoRA model is ready. Starting training...")

    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=1,            # log every single step
        logging_first_step=True,     # ensure first step is logged
        report_to="none",            # no TensorBoard / WandB / etc.
        remove_unused_columns=False,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        metric_for_best_model="eval_runtime",
        save_total_limit=1,         # keep only the last checkpoint
        resume_from_checkpoint=True,
        disable_tqdm=True,          # turn off all tqdm bars
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[CustomCheckpointCallback()],
    )


    latest_ckpt = find_latest_checkpoint(training_args.output_dir)
    if latest_ckpt:
        print(f"Resuming training from latest checkpoint: {latest_ckpt}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No existing checkpoint found; starting from scratch.")
        trainer.train()

    return trainer


if __name__ == "__main__":

    local_rank = initialize_distributed()

    model_name = "/model/Meta-Llama-3.1-8B-Instruct"
    model, tokenizer = load_model(model_name, local_rank)

    train_dataset, test_dataset = load_and_prepare_data(tokenizer, local_rank)

    checkpoint_dir = "/model/checkpoints/Meta-Llama-3.1-8B-Instruct-finetuned"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    lora_config = configure_lora(model, checkpoint_dir)

    trainer = train_model(model, lora_config, train_dataset, test_dataset, tokenizer, local_rank, checkpoint_dir)

    dist.destroy_process_group()
