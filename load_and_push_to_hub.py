import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--push", type=str, required=True)
parser.add_argument("--tokenizer")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--peft", type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer if args.tokenizer else args.model,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(args.model)
if args.peft:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.peft)
    model = model.merge_and_unload()

model.push_to_hub(args.push, private=True)
tokenizer.push_to_hub(args.push, private=True)
