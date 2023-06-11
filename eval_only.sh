accelerate launch  bigcode-evaluation-harness/main.py \
  --model Salesforce/codegen2-7B \
  --tasks multiple-lua \
  --allow_code_execution \
  --load_generations_path generations.json
