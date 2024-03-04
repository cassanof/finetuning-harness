"""
Code adapted from: https://github.com/loubnabnl/santacoder-finetuning
"""

import argparse
from functools import wraps
import os

import json
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import time
from push_checkpoints import push_checkpoints
from datasets.load import load_dataset, load_from_disk
from datasets import DatasetDict
from number_of_tokens import get_total_tokens, get_total_tokens_from_iterable
from dataset_loader import ConstantLengthDataset, PaddedDataset, TQDMWraper
from lora import hacky_model_convert, find_all_linear_names, SavePeftModelCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        self.tokenizer.save_pretrained(checkpoint_folder)


def neftune_post_forward_hook(module, input, output):  # NOTE: copypasted from TRL
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```

    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set
            `module.neftune_noise_alpha` to the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + \
            torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output


def unwrap_model(model: nn.Module) -> nn.Module:  # NOTE: copypasted from TRL
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class BetterTrainer(Trainer):
    def __init__(
        self,
        neftune_noise_alpha: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.neftune_noise_alpha = neftune_noise_alpha

    @wraps(Trainer.train)
    def train(self, *args, **kwargs):  # NOTE: copypasted from TRL
        # Activate neftune right before training.
        if self.neftune_noise_alpha is not None:
            self.model = self._trl_activate_neftune(self.model)

        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            unwrapped_model = unwrap_model(self.model)
            embeddings = unwrapped_model.get_input_embeddings()

            self.neftune_hook_handle.remove()
            if hasattr(embeddings, "neftune_noise_alpha"):
                del embeddings.neftune_noise_alpha

        return output

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids)[0]

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss


    def _trl_activate_neftune(self, model):  # NOTE: copypasted from TRL
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
        """
        print("*** Activating NEFTune ***")
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(
            neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="bigcode/starcoderbase")
    parser.add_argument("--model_revision", type=str, default="main")
    parser.add_argument("--dataset_name", type=str,
                        default="bigcode/starcoderdata")
    parser.add_argument("--dataset_revision", type=str, default="main")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--perc_valid_set", type=float, default=0.005)
    parser.add_argument("--data_column", type=str, default="content")
    parser.add_argument("--min_edu_score", type=float, default=0.0)
    parser.add_argument("--edu_score_column", type=str)
    parser.add_argument("--no_shuffle_train", action="store_true")

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bits", type=int, default=8)
    parser.add_argument("--lora_extreme", action="store_true")

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--total_tokens", type=int,
                        help="Total number of tokens in the dataset. If not provided, will be computed.")
    parser.add_argument("--no_approx_tokens", action="store_true")
    parser.add_argument("--dataset_loader", type=str,
                        default="constant", choices=["constant", "padded"])
    parser.add_argument("--pad_token_id", type=int, default=None)
    parser.add_argument("--trim_longer", action="store_true")
    parser.add_argument("--mask_loss_till_token_id", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--attention_dropout", type=float, default=None)
    parser.add_argument("--neft_alpha", type=float, default=None)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--no_gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--eval_freq", default=1.0, type=float,
                        help="Evaluate X times per epoch, can be < 1")
    parser.add_argument("--save_freq", default=1.0, type=float,
                        help="Save X times per epoch, can be < 1")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--custom_tokenizer", type=str, default=None)

    parser.add_argument("--humaneval_eval_loss", action="store_true")
    parser.add_argument("--save_best_model", action="store_true")
    parser.add_argument("--lang", type=str, default="lua")

    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--fa2", action="store_true")
    return parser


def is_main(args):
    return args.local_rank in [-1, 0]


def get_rank(args):
    return args.local_rank if args.local_rank != -1 else 0


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


def get_num_gpus(args):
    # NOTE: using torch.cuda.device_count() isn't bulletproof, but it's good enough for our purposes
    return 1 if args.local_rank == -1 else torch.cuda.device_count()


def load_source_dataset(args):
    num_gpus = get_num_gpus(args)
    # if dataset is a path, load it from the path
    if os.path.isdir(args.dataset_name):
        dataset = load_from_disk(args.dataset_name)
        # if DatasetDict, select the split
        if isinstance(dataset, DatasetDict):
            dataset = dataset[args.split]

    else:
        kwargs = {}
        if args.subset:
            kwargs["data_dir"] = args.subset
        dataset = load_dataset(
            args.dataset_name,
            revision=args.dataset_revision,
            split=args.split,
            num_proc=args.num_workers // num_gpus,
            **kwargs,
        )

    return dataset


def create_dataloaders(tokenizer, args, tqdm=True):
    # TODO: for multi-node, this won't work
    num_gpus = get_num_gpus(args)

    dataset = load_source_dataset(args)

    eval_dataset = None
    if args.humaneval_eval_loss:
        eval_dataset = load_dataset("nuprl/MultiPL-E-synthetic-solutions", split="train") \
            .filter(lambda example: example["language"] == args.lang) \
            .map(lambda example: {"content": example["prompt"] + example["solution"]})

    if args.humaneval_eval_loss:
        valid_data = eval_dataset
        train_data = dataset
    elif args.perc_valid_set == 0:
        train_data = dataset
        valid_data = None
    else:
        dataset = dataset.train_test_split(  # type: ignore
            test_size=args.perc_valid_set, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
    if args.edu_score_column:
        train_data = train_data.filter(
            lambda example: example[args.edu_score_column] >= args.min_edu_score
        )
        if not args.humaneval_eval_loss:
            assert valid_data is not None
            valid_data = valid_data.filter(
                lambda example: example[args.edu_score_column] >= args.min_edu_score
            )

    if not args.no_shuffle_train:
        train_data = train_data.shuffle(seed=args.seed)

    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data) if valid_data else None}"
    )
    chars_per_token = chars_token_ratio(
        train_data, tokenizer, args.data_column)
    print(
        f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    if args.dataset_loader == "constant":
        def ds_constructor(data, infinite): return ConstantLengthDataset(
            tokenizer,
            data,
            infinite=infinite,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
            content_field=args.data_column,
        )
    elif args.dataset_loader == "padded":
        def ds_constructor(data, infinite): return PaddedDataset(
            tokenizer,
            data,
            infinite=infinite,
            seq_length=args.seq_length,
            content_field=args.data_column,
            pad_token_id=args.pad_token_id,
            trim_longer=args.trim_longer,
            mask_loss_till_token_id=args.mask_loss_till_token_id,
        )
    else:
        raise ValueError(
            f"Invalid dataset loader: {args.dataset_loader}. Must be 'constant' or 'padded'.")

    total_tokens = args.total_tokens
    if total_tokens is None:
        # approximate if dataset is too large (greater than 50k examples)
        if len(train_data) > 50000 and not args.no_approx_tokens:
            print(
                f"Dataset is too large ({len(train_data)} examples). Approximating the number of tokens. Disable with --no_approx_tokens.")
            total_tokens_50k = get_total_tokens(
                train_data, tokenizer, args.data_column, 50000)
            total_tokens = total_tokens_50k * (len(train_data) // 50000)
        else:
            total_tokens = get_total_tokens_from_iterable(
                ds_constructor(train_data, infinite=False))

    training_examples = total_tokens // args.seq_length

    effective_batch_size = args.batch_size * \
        args.gradient_accumulation_steps * num_gpus
    max_steps = max(1, int(training_examples /
                    effective_batch_size * args.epochs))

    if is_main(args):
        print(f" #### SCALING LAWS ####")
        print(f" ###### Examples ######")
        print(f"Total tokens: {total_tokens}")
        print(f"Seq length: {args.seq_length}")
        print(f"Training examples: {training_examples}")
        print(f" ####### Batch #######")
        print(f"Batch size: {args.batch_size}")
        print(
            f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Number of GPUs: {num_gpus}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Epoch: {args.epochs}")
        print(f"####### RESULT ###########")
        print(f"# Max steps: {max_steps} #")
        print(f"##########################")

    train_dataset = ds_constructor(train_data, infinite=True)
    valid_dataset = ds_constructor(
        valid_data, infinite=False) if valid_data else None

    if tqdm and is_main(args):
        train_dataset = TQDMWraper(
            train_dataset, num_iters=training_examples * args.epochs, desc="Training")
        if valid_dataset:
            valid_dataset = TQDMWraper(
                valid_dataset, desc="Evaluating")

    return max_steps, train_dataset, valid_dataset


def dtype_from_str(dtype_str):
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {dtype_str}")


def run_training(args, max_steps, train_data, val_data):
    os.makedirs(args.output_dir, exist_ok=True)
    model_extra_kwargs = {}
    if args.lora:
        config = {}
        if args.lora_bits == 8:
            config["load_in_8bit"] = True
        elif args.lora_bits == 4:
            config["load_in_4bit"] = True
        else:
            assert False, f"Invalid lora_bits: {args.lora_bits}"

        if args.lora_extreme:  # extreme quantization
            print("LOADING EXTREME QUANTIZATION!!!!!!!")
            config["load_in_8bit"] = False  # disable if set by user
            config["load_in_4bit"] = True
            config["llm_int8_threshold"] = 6.0
            config["llm_int8_has_fp16_weight"] = False
            config["bnb_4bit_quant_type"] = "nf4"
            config["bnb_4bit_use_double_quant"] = True
            dtype = None
            if args.bf16:
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
            config["bnb_4bit_compute_dtype"] = dtype

        model_extra_kwargs["device_map"] = {
            "": args.local_rank if args.local_rank != -1 else 0
        }
        model_extra_kwargs["quantization_config"] = BitsAndBytesConfig(
            **config)

    if args.fa2:
        # need to set dtype to either float16 or bfloat16
        if args.bf16:
            model_extra_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_extra_kwargs["torch_dtype"] = torch.float16

    if args.torch_dtype:  # overrides everything else
        model_extra_kwargs["torch_dtype"] = dtype_from_str(args.torch_dtype)

    if args.attention_dropout is not None:  # some models dont support this
        model_extra_kwargs["attention_dropout"] = args.attention_dropout

    train_data.start_iteration = 0

    # calculate eval and save steps from max steps
    steps_per_epoch = max_steps // args.epochs
    eval_steps = int(steps_per_epoch * args.eval_freq)
    eval_steps = None if eval_steps == 0 else eval_steps  # disable if 0
    save_steps = int(steps_per_epoch * args.save_freq)
    print(f"Eval steps: {eval_steps} -- Save steps: {save_steps}")

    extra_training_args = {}
    if args.deepspeed:
        extra_training_args["deepspeed"] = args.deepspeed

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        torch_compile=True,
        dataloader_drop_last=True,
        evaluation_strategy="steps" if eval_steps else "no",
        max_steps=max_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.no_gradient_checkpointing,
        save_total_limit=99999 if args.lora else args.save_total_limit,
        save_strategy=args.save_strategy,
        fp16=args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        report_to=["wandb"] if not args.no_wandb else [],
        load_best_model_at_end=args.save_best_model,
        ddp_find_unused_parameters=False,
        **extra_training_args,
    )

    print(f"*** [{get_rank(args)}] Loading the model. ***")

    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        trust_remote_code=True,
        **model_extra_kwargs,
    )

    if args.lora:
        print("Preparing model for LoRA training")
        prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=not args.no_gradient_checkpointing)
        all_linear_layers = find_all_linear_names(model)
        added_modules = set(["c_proj", "c_attn", "q_attn"])
        modules = list(added_modules.union(all_linear_layers))
        print(f"Target modules: {modules}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        )

        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        hacky_model_convert(args, model)

    print_trainable_parameters(model) if not args.deepspeed else None

    if is_main(args) and not args.no_wandb:
        import wandb
        date = time.strftime("%Y-%m-%d-%H-%M")
        lora_str = "_lora" if args.lora else ""
        model_name = args.model_path.rstrip("/").split("/")[-1]
        dataset_name = args.dataset_name.rstrip("/").split("/")[-1]
        wandb_name = f"{model_name}_{dataset_name}_{date}_{lora_str}"
        try:
            wandb.init(name=wandb_name)
        except Exception as e:
            print(
                f"Failed to initialize wandb -- Can disable it with the `--no_wandb` option.\nError: {e}")
            raise e

    trainer_extra_kwargs: Dict[str, Any] = {
        "callbacks": [SaveTokenizerCallback(train_data.get_tokenizer())],
    }
    if args.lora:
        trainer_extra_kwargs["callbacks"] += [SavePeftModelCallback]

    trainer = BetterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        neftune_noise_alpha=args.neft_alpha,
        **trainer_extra_kwargs
    )

    print(f"*** [{get_rank(args)}] Training... ***")

    if args.checkpoint:
        print(f"***** Loading checkpoint from {args.checkpoint} *****")
        trainer.train(args.checkpoint)
    else:
        # find latest checkpoint
        chks = []
        for checkpoint in Path(args.output_dir).glob("checkpoint-*"):
            try:
                num = int(checkpoint.name.split("-")[-1])
                chks.append(num)
            except ValueError:
                continue
        if len(chks) > 0:
            chks.sort()
            last_chk = chks[-1]
            print(
                f"***** Automatically detected checkpoint. Loading checkpoint from {last_chk} *****")
            trainer.train(f"{args.output_dir}/checkpoint-{last_chk}")
        else:
            trainer.train()

    if args.push_to_hub:
        push_checkpoints(args.output_dir, args.push_to_hub)

    if args.save_best_model:
        print("Saving best model...")
        model.save_pretrained(os.path.join(args.output_dir, "best/"))


def load_special_tokens(tokenizer):
    thisFolder = os.path.dirname(os.path.abspath(__file__))
    file = open(os.path.join(thisFolder, "special_tokens_map.json"))
    special_tokens_map = json.load(file)
    tokenizer.add_special_tokens(special_tokens_map)
