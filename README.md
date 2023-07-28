# Scripts to finetune GPTBigCode architecture models

This repo provides the whole pizza for fine-tuning GPTBigCode models (e.g. StarCoder) on code generation tasks.
It includes:

1. Constant Length Dataset Loader
2. Scaling laws for computing the correct number of steps, given number of gpus, effective batch size, and number of epochs
3. LoRA, with 8, 4 bits and QLoRA (double quant) support
4. DeepSpeed support for fine-tuning large models
5. Edu-score filtering to remove non-educational data
6. Multi-language loss evaluation (using MultiPL-E evaluation datasets)
7. Custom tokenizer injection
8. Automatic mixed precision quantization
