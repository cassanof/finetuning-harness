from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pathlib


def push_checkpoints(chk_dir, base_repo, tokenizer=None):
    checkpoints = []
    for path in pathlib.Path(chk_dir).rglob(f"checkpoint-*"):
        checkpoints.append(path.name)

    dir_name = pathlib.Path(chk_dir)

    if tokenizer:
        print(f"Pushing tokenizer from {tokenizer} to {base_repo}")
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tok.push_to_hub(base_repo, private=True)
    else:
        try:
            chk0 = checkpoints[0]
            print(f"Pushing tokenizer from {chk0} to {base_repo}")
            tok = AutoTokenizer.from_pretrained(dir_name / chk0)
            tok.push_to_hub(base_repo, private=True)
        except Exception as e:
            print(e)
            print("Failed to push tokenizer. Not going to push one.")

    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    for epoch, checkpoint in enumerate(checkpoints):
        epoch += 1  # 1-indexed
        commit = f"{dir_name}-epoch{epoch}"
        print(
            f"Pushing {checkpoint} (epoch {epoch}) to {base_repo} - {commit}")
        m = AutoModelForCausalLM.from_pretrained(
            dir_name / checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        while True:
            try:
                m.push_to_hub(base_repo, private=True,
                              commit_message=commit)
            except RuntimeError as e:
                print(e)
                print("Retrying...")
                continue

            break


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dir", type=str, required=True)
    arg_parser.add_argument("--tokenizer", type=str,
                            default="", help="Optional, tokenizer to push to hub. Otherwise it will try to load the tokenizer from the checkpoint")
    arg_parser.add_argument("--base_repo", type=str, required=True)
    args = arg_parser.parse_args()
    push_checkpoints(args.dir, args.base_repo, args.tokenizer)
