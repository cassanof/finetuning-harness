# Pipeline for Fine-Tuning HuggingFace Models On Code Generation Tasks

This repo provides the whole pizza for fine-tuning HuggingFace models (e.g. StarCoder, or Code LLama) on code generation tasks.
It includes:

1. Constant Length Dataset Loader
2. Scaling laws for computing the correct number of steps, given number of gpus, effective batch size, and number of epochs
3. LoRA, with 8, 4 bits and QLoRA (double quant) support
4. DeepSpeed support for fine-tuning large models
5. Edu-score filtering to remove non-educational data
6. Multi-language loss evaluation (using MultiPL-E evaluation datasets)
7. Custom tokenizer injection
8. Automatic mixed precision quantization

## Generic Usage

This repo is driven by the `main.py` script. It supports supports a wide range of arguments, which can be listed using `python main.py --help`.
It may be simpler to look at the scripts in the `run_scripts` directory, which are used to run training on the different models with different settings.

### LoRA

There is built-in support for LoRA, which can be enabled by passing the `--lora` flag. See `run_scripts/run_starcoder_lora.sh` for an example.
There is additional support for some "lora hacks", like double quant, which can be enabled by passing the `--lora_extreme` flag.

### DeepSpeed

We support DeepSpeed and we recommend using it for training large models, instead of using LoRA.
See `./run_starcoder.sh` or `./run_codellama_34b.sh` for an example. There are various deepspeed
configs in this repo that can be used right away.

### Evaluation

The evaluation for the models is done via the `multipl_e_eval.sh` script, and it requires an installation
of the [MultiPL-E](https://github.com/nuprl/MultiPL-E) repo. Through this script, you can evaluate
different checkpoints at the same time using different GPUs on multiple languages and datasets (HumanEval or MBPP).
