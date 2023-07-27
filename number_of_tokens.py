# script to get the total number of tokens in a dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
from tqdm import tqdm


def get_total_tokens(dataset, tokenizer, data_column, nb_examples):
    """
    Estimate the total number of tokens in the dataset.
    """
    total_tokens = 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = example[data_column]
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str,
                        default="./starcoder_tokenizer_files")
    parser.add_argument("--dataset", type=str,
                        default="nuprl/stack_dedup_lua_codegen")
    args = parser.parse_args()

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer
    )

    print("Loading dataset")
    dataset = load_dataset(
        args.dataset,
        split="train")

    print("Tokenizing dataset")
    data_column = "content"
    total_tokens = get_total_tokens(
        dataset, tokenizer, data_column, nb_examples=len(dataset))
    print(f"Total number of tokens in dataset: {total_tokens}")
