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
