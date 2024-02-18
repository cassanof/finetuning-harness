from typing import List
from torch.utils.data import IterableDataset
from tqdm import tqdm
import random
import torch


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=2048,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        concat_token_id=None,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = concat_token_id if concat_token_id is not None else tokenizer.eos_token_id
        print(f"Concat token id (EOS token): {self.concat_token_id}")
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(
                buffer, truncation=False)["input_ids"]
            all_token_ids = []
            examples = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)

            random.shuffle(examples)
            for input_ids in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids)),
                }

    def get_tokenizer(self):
        return self.tokenizer


class PaddedDataset(IterableDataset):
    """
    Unlike ConstantLengthDataset this dataset returns padded sequences of tokens concatenated together,
    which all have a fixed length of seq_length. The dataset will panic if a sequence is longer
    than seq_length, except if trim_longer is set to True, in which case the sequence
    will be trimmed to seq_length. It is important to set pad_token_id to the id of the
    padding token in the tokenizer, otherwise the model will be trained on a wrong padding token.
    By default, we set the pad_token_id to the pad_token_id of the tokenizer, if it exists,
    otherwise we set it to 0. The padding is done at the end of the concatenated sequence (right padding).
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=2048,
        content_field="content",
        concat_token_id=None,
        pad_token_id=None,
        trim_longer=False,
        # niche option for some instruct tasks. removes loss calculation for tokens before a certain token id (example-wise and inclusive)
        # this token is a marker and gets removed from the input_ids and labels
        mask_loss_till_token_id=None,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = concat_token_id if concat_token_id is not None else tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.content_field = content_field
        self.trim_longer = trim_longer
        self.mask_loss_till_token_id = mask_loss_till_token_id
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif self.tokenizer.pad_token_id is None:
            # default to 0
            self.pad_token_id = 0
        else:
            # we good, we have a pad token id preset
            self.pad_token_id = self.tokenizer.pad_token_id

        print(f"Concat token id (EOS token): {self.concat_token_id}")
        print(f"Pad token id: {self.pad_token_id}")

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        prev_iter_skipped = None
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                try:
                    if prev_iter_skipped is not None:
                        encoded = prev_iter_skipped
                        prev_iter_skipped = None
                    else:
                        new = next(iterator)[self.content_field]
                        encoded = self.tokenizer.encode(
                            new) + [self.concat_token_id]

                    if len(encoded) > self.seq_length:
                        if self.trim_longer:
                            encoded = encoded[:self.seq_length -
                                              1] + [self.concat_token_id]

                        else:
                            raise ValueError(
                                f"Sequence of length {len(encoded)} is longer than seq_length {self.seq_length}."
                            )

                    if len(encoded) + buffer_len > self.seq_length:
                        prev_iter_skipped = encoded
                        break

                    buffer.append(encoded)
                    buffer_len += len(encoded)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            assert buffer_len <= self.seq_length

            # concatenate all sequences
            token_ids = []
            for tokenized_input in buffer:
                token_ids.extend(tokenized_input)

            # pad to seq_length
            token_ids.extend([self.pad_token_id] *
                             (self.seq_length - len(token_ids)))

            labels = token_ids
            if self.mask_loss_till_token_id is not None:
                labels = apply_mask_till_token_id(
                    labels,
                    mask_till_token_id=self.mask_loss_till_token_id,
                    concat_token_id=self.concat_token_id,
                    pad_token_id=self.pad_token_id,
                )
                # remove the mask_till_token_id from the input_ids and labels
                token_ids = [t for t in token_ids if t !=
                             self.mask_loss_till_token_id]
                assert len(token_ids) == len(labels)

                # pad to seq_length
                token_ids.extend([self.pad_token_id] *
                                 (self.seq_length - len(token_ids)))
                labels.extend([self.pad_token_id] *
                              (self.seq_length - len(labels)))
                assert len(token_ids) == len(labels)

            yield {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    def get_tokenizer(self):
        return self.tokenizer


def apply_mask_till_token_id(
        token_ids: List[int],
        mask_till_token_id: int,
        concat_token_id: int,
        pad_token_id: int,
        mask=-100,
):
    # notes:
    #   - stop condition: we reach the end of the sequence or we find pad_token_id and evey token after it is pad_token_id
    #   - concat_token_id delimiates examples. we need to start re-masking from this token_id
    #   - mask_till_token_id is the token that dictates the end of the masking (we also mask this token)

    masked = []
    masking = True
    masking_off_forever = False
    for i, token in enumerate(token_ids):
        if token == pad_token_id and all([t == pad_token_id or t == concat_token_id for t in token_ids[i:]]):
            masking_off_forever = True

        if token == mask_till_token_id:
            masking = False
        else:  # do not add if mask token
            if masking and not masking_off_forever:
                masked.append(mask)
            else:
                masked.append(token)

        if token == concat_token_id:
            masking = True

    return masked


class TQDMWraper(IterableDataset):
    def __init__(self, dataset, num_iters=None, desc=""):
        self.dataset = dataset
        self.num_iters = num_iters
        self.desc = desc

    def __iter__(self):
        for example in tqdm(self.dataset, total=self.num_iters, desc=self.desc):
            yield example

    def get_tokenizer(self):
        return self.dataset.get_tokenizer()


if __name__ == "__main__":
    # testing out the padded dataset
    import datasets
    from transformers import AutoTokenizer
    ds = datasets.load_dataset(
        "nuprl-staging/multiplt-python-instrs-5k-train-codellama", split="train")
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-hf")
    dataset = PaddedDataset(tokenizer, ds, seq_length=2048)
    num_exs = 0
    for i, example in enumerate(dataset):
        decoded = tokenizer.decode(example["input_ids"])
        if i < 4:
            print("#" * 80)
            print(decoded)
        num_exs += decoded.count("### Instruction:")

    print(ds)
    print(f"Total number of examples: {num_exs}")
