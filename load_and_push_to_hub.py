import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--push", type=str, required=True)
parser.add_argument("--tokenizer")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--peft", type=str)
args = parser.parse_args()

print(f"{args.model} ---> {args.push}")

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer if args.tokenizer else args.model,
)
print("Loading model")
model = AutoModelForCausalLM.from_pretrained(args.model)
if args.peft:
    from peft import PeftModel
    print("Loading PEFT")
    model = PeftModel.from_pretrained(model, args.peft)
    print("Merging and unloading")
    model = model.merge_and_unload()

print("Pushing to hub")

model.push_to_hub(args.push, private=True)
tokenizer.push_to_hub(args.push, private=True)
