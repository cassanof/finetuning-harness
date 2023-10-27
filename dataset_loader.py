from torch.utils.data import IterableDataset
import random
from tqdm import tqdm
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
            reruns (int): Number of times to rerun the dataset.
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
        reruns=1,
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
        self.reruns = reruns

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        reruns = self.reruns
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite or reruns > 0:
                        iterator = iter(self.dataset)
                        reruns -= 1
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
                    "input_ids": torch.LongTensor(input_ids),
                    "labels": torch.LongTensor(input_ids),
                }


class PaddedDataset(IterableDataset):
    """
    Unlike ConstantLengthDataset this dataset returns padded sequences of tokens, which
    have a fixed length of seq_length. The dataset will panic if a sequence is longer
    than seq_length, except if trim_longer is set to True, in which case the sequence
    will be trimmed to seq_length. It is important to set pad_token_id to the id of the
    padding token in the tokenizer, otherwise the padding will not work.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=2048,
        trim_longer=False,
        content_field="content",
        pad_token_id=None,
        use_tqdm=True,
    ):
        self.tokenizer = tokenizer
        if pad_token_id is not None:
            self.tokenizer.pad_token_id = pad_token_id
        elif self.tokenizer.pad_token_id is None:
            # default to 0
            self.tokenizer.pad_token_id = 0
        else:
            # we good, we have a pad token id preset
            pass

        print(f"Padded token id (PAD token): {self.tokenizer.pad_token_id}")
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.content_field = content_field
        self.trim_longer = trim_longer
        self.use_tqdm = use_tqdm

    def __iter__(self):
        def init_iter():
            iterator = iter(self.dataset)
            if self.use_tqdm:
                iterator = tqdm(iterator, desc="PaddedDataset")
            return iterator
        iterator = init_iter()
        more_examples = True

        while more_examples:
            try:
                content = next(iterator)[self.content_field]
                tokenized_input = self.tokenizer(
                    content,
                    max_length=self.seq_length,
                    truncation=self.trim_longer,
                    padding='max_length',
                    return_tensors="pt"
                )["input_ids"].squeeze(0)

                # Check the length if truncation is not allowed
                if not self.trim_longer and len(tokenized_input) > self.seq_length:
                    raise ValueError(
                        f"A sequence is longer than {self.seq_length} tokens and trim_longer is set to False")

                self.current_size += 1
                yield {
                    "input_ids": tokenized_input,
                    "labels": tokenized_input.clone(),
                }
            except StopIteration:
                if self.infinite:
                    iterator = init_iter()
                else:
                    more_examples = False
