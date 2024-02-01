# A Pipeline for Fine-Tuning HuggingFace Models

This repo provides the whole pizza for fine-tuning HuggingFace models (e.g. Llama2, DeepSeek, StarCoder, or Code Llama) on any task.
_It has been built primarily for code generation tasks._
The pipeline includes:

1. Both Constant Length Dataset Loader and Padded Dataset Loader. The constant length one is good for code generation (e.g. Copilot) or "further pre-training", while the padded one is typically better for instruction-tuning.
2. Scaling laws for computing the correct number of steps, given number of gpus, effective batch size, and number of epochs
3. LoRA, with 8, 4 bits and QLoRA (double quant) support
4. FlashAttention2 for super-duper fast long sequence training
5. DeepSpeed support for fine-tuning large models by offloading to multiple GPUs and the CPU
6. Edu-score filtering to remove non-educational data
7. Multi-programming-language loss evaluation (using MultiPL-E evaluation datasets)
8. Custom tokenizer injection
9. Automatic mixed precision

## Generic Usage

This repo is driven by the `main.py` script. It supports supports a wide range of arguments, which can be listed using `python main.py --help`.
It may be simpler to look at the scripts in the `run_scripts` directory, which are used to run training on the different models with different settings.

### LoRA

There is built-in support for LoRA, which can be enabled by passing the `--lora` flag. See `run_scripts/run_starcoder_lora.sh` for an example.
There is additional support for some "lora hacks", like double quant, which can be enabled by passing the `--lora_extreme` flag.

### DeepSpeed

We support DeepSpeed and we recommend using it for training large models, instead of using LoRA.
See `run_starcoder.sh` or `run_codellama_34b.sh` for an example. There are various deepspeed
configs in this repo that can be used right away.

### FlashAttention2

If you need to train on long sequences, you can use FlashAttention2. This can be enabled by passing the `--fa2` flag.
However, this will require you to install the [FlashAttention2](https://github.com/Dao-AILab/flash-attention)
package, which is not included in the requirements.

### Evaluation

The evaluation for the models is done via the `multipl_e_eval.sh` script, and it requires an installation
of the [MultiPL-E](https://github.com/nuprl/MultiPL-E) repo. This is an evaluation for code generation only.
Through this script, you can evaluate
different checkpoints at the same time using different GPUs on multiple languages and datasets (HumanEval or MBPP).

### Pushing Checkpoints

There are two scripts that can be used as helpers for pushing checkpoints to HuggingFace:

1. `./scripts/load_and_push_to_hub.py` can be used to push a single checkpoint
2. `./scripts/push_checkpoints.py` can be used to push multiple checkpoints in the given directory

# Citation

If you use this code in your research, please cite it as follows:

```
@software{cassano2023finetuning,
    author = {Cassano, Federico},
    month = jun,
    title = {{A Pipeline for Fine-Tuning HuggingFace Models}},
    url = {https://github.com/cassanof/finetuning-harness},
    version = {1.0.0},
    year = {2023}
}
```
