"""
Fine-Tune SantaCoder on code/text dataset
"""

import argparse
import os

import numpy as np
import json
import wandb
import torch
import random
from datasets.load import load_dataset
from torch.utils.data import IterableDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainerState,
    TrainerControl,
    TrainerCallback,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from accelerate import Accelerator


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(
            checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigcode/santacoder")
    parser.add_argument("--model_revision", type=str, default="main")
    parser.add_argument("--dataset_name", type=str,
                        default="bigcode/santacoder")
    parser.add_argument("--dataset_revision", type=str, default="main")
    parser.add_argument("--subset", type=str, default="data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--perc_valid_set", type=float, default=0.005)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--data_column", type=str, default="content")

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lang", type=str, default="lua")

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default="checkpoint")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--no_custom_tokenizer", action="store_true")
    parser.add_argument("--humaneval_eval_loss", action="store_true")
    parser.add_argument("--eval_reruns", type=int, default=1)
    parser.add_argument("--no_shuffle_train", action="store_true")

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(model):
    import bitsandbytes as bnb
    cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        reruns=1,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else args.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.reruns = reruns

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        reruns = self.reruns
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite or reruns > 0:
                        iterator = iter(self.dataset)
                        reruns -= 1
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(
                buffer, truncation=False)["input_ids"]
            all_token_ids = []
            examples = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)

            random.shuffle(examples)
            for input_ids in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(input_ids),
                    "labels": torch.LongTensor(input_ids),
                }


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        revision=args.dataset_revision,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )

    eval_dataset = None
    if args.humaneval_eval_loss:
        eval_dataset = load_dataset("nuprl/MultiPL-E-synthetic-solutions", split="train") \
            .filter(lambda example: example["language"] == args.lang) \
            .map(lambda example: {"content": example["prompt"] + example["solution"]})

    if args.streaming:
        print("Loading the dataset in streaming mode")
        if args.humaneval_eval_loss:
            raise ValueError(
                "TODO Streaming mode is not supported for humaneval_eval_loss"
            )
        else:
            valid_data = dataset.take(args.size_valid_set)  # type: ignore
            train_data = dataset.skip(args.size_valid_set)  # type: ignore
        train_data = train_data.shuffle(
            buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        if args.humaneval_eval_loss:
            valid_data = eval_dataset
            train_data = dataset if args.no_shuffle_train else dataset.shuffle(
                seed=args.seed)
        else:
            dataset = dataset.train_test_split(  # type: ignore
                test_size=args.perc_valid_set, seed=args.seed)
            train_data = dataset["train"]
            valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )
    chars_per_token = chars_token_ratio(
        train_data, tokenizer, args.data_column)
    print(
        f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        reruns=args.eval_reruns,
    )

    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    print(f"Loading the model.")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        trust_remote_code=True,
        load_in_8bit=args.lora or args.load_in_8bit,
        use_cache=not args.no_gradient_checkpointing,
        device_map={
            "": Accelerator().process_index} if args.lora or args.load_in_8bit else None,
    )

    train_data.start_iteration = 0

    if args.lora:
        print("!!! Using LoRA")
        prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=not args.no_gradient_checkpointing)
        all_linear_layers = find_all_linear_names(model)
        print(f"Found {len(all_linear_layers)} linear layers")
        print(all_linear_layers)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_proj", "c_attn", "q_attn", "q_proj",
                            "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        )

        model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.no_gradient_checkpointing,
        hub_strategy=args.hub_strategy,
        save_total_limit=99999 if args.lora else args.save_total_limit,
        hub_model_id=args.hub_model_id,
        save_strategy=args.save_strategy,
        push_to_hub=args.push_to_hub,
        fp16=args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=f"{args.model_path.replace('/', '_')}",
        report_to=["wandb"],
        ddp_find_unused_parameters=False,
    )

    if (args.local_rank == 0 or args.local_rank == -1):
        wandb.init(project="roblox")

    callbacks = []
    if args.lora:
        callbacks = [SavePeftModelCallback]

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data, callbacks=callbacks
    )

    print("Training...")
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        trainer.train(args.checkpoint)
    else:
        trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def load_special_tokens(tokenizer):
    thisFolder = os.path.dirname(os.path.abspath(__file__))
    file = open(os.path.join(thisFolder, "special_tokens_map.json"))
    special_tokens_map = json.load(file)
    tokenizer.add_special_tokens(special_tokens_map)


def main(args):
    if args.no_custom_tokenizer:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            revision=args.model_revision,
        )
    else:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            "./tokenizer_files"
        )
        load_special_tokens(tokenizer)

    print(tokenizer.special_tokens_map)

    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
