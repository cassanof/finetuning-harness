from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    logging,
    set_seed,
)
from train import load_special_tokens, create_datasets, run_training, get_arg_parser


def get_args_from_cli():
    return get_arg_parser().parse_args()


def main(args):
    set_seed(args.seed)

    logging.set_verbosity_error()

    if args.custom_tokenizer:
        print("Loading custom tokenizer ...")
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.custom_tokenizer,
            padding_side="right",
        )
        load_special_tokens(tokenizer)
        print("Special tokens:")
        print(tokenizer.special_tokens_map)
    else:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            revision=args.model_revision,
            padding_side="right",
        )

    max_steps, train_dataset, eval_dataset = create_datasets(tokenizer, args)
    # suppress all dynamo errors
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


    run_training(args, max_steps, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args_from_cli()
    main(args)
