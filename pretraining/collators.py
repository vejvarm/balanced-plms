import random
import numpy as np
from transformers import PreTrainedTokenizerBase, BatchEncoding

class DataCollatorForT5MLM:
    """
    Data collator for T5 span-masked language modeling.
    For a given batch of `input_ids`, random spans are chosen (with given noise density
    and mean span length) to be replaced with sentinel tokens. The target is the concatenated masked
    spans, each prefixed by its corresponding sentinel.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, noise_density: float = 0.15,
                 mean_noise_span_length: float = 3.0, input_length: int = 512, target_length: int = None):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length if target_length is not None else input_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples: list[dict]) -> BatchEncoding:
        batch_input_ids = [ex["input_ids"] for ex in examples]
        # Pad the batch using the tokenizer pad method.
        batch = self.tokenizer.pad({"input_ids": batch_input_ids}, padding="max_length", return_tensors="pt", max_length=self.input_length)
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape

        masked_inputs = []
        targets = []

        for i in range(batch_size):
            tokens = input_ids[i].tolist()
            noise_mask = self.random_spans_noise_mask(seq_len)
            corrupted_input, target = self.create_sentinel_inputs_and_labels(tokens, noise_mask)
            masked_inputs.append(corrupted_input)
            targets.append(target)

        batch_inputs = self.tokenizer.pad({"input_ids": masked_inputs}, padding="max_length", return_tensors="pt", max_length=self.input_length)
        batch_labels = self.tokenizer.pad({"input_ids": targets}, padding="max_length", return_tensors="pt", max_length=self.input_length)
        batch["input_ids"] = batch_inputs["input_ids"]
        batch["labels"] = batch_labels["input_ids"]
        return batch

    def random_spans_noise_mask(self, length: int) -> list[bool]:
        num_noise_tokens = int(np.round(length * self.noise_density))
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        
        lengths = np.random.poisson(self.mean_noise_span_length, num_noise_spans)
        lengths = lengths[lengths > 0]  # Ensure no zero-length spans
        
        # Randomly place spans
        mask = np.zeros(length, dtype=bool)
        span_starts = np.random.choice(length - lengths.sum(), num_noise_spans, replace=False)
        
        for start, span_length in zip(span_starts, lengths):
            mask[start:start + span_length] = True
        
        return mask.tolist()


    def create_sentinel_inputs_and_labels(self, tokens: list[int], noise_mask: list[bool]):
        input_tokens = []
        label_tokens = []
        current_sentinel = 0
        i = 0
        while i < len(tokens):
            if noise_mask[i]:
                # Insert sentinel token in input and target.
                sentinel = self.tokenizer.convert_tokens_to_ids(f"<extra_id_{current_sentinel}>")
                input_tokens.append(sentinel)
                label_tokens.append(sentinel)
                while i < len(tokens) and noise_mask[i]:
                    label_tokens.append(tokens[i])
                    i += 1
                current_sentinel += 1
            else:
                input_tokens.append(tokens[i])
                i += 1
        label_tokens.append(self.tokenizer.eos_token_id)
        # Optionally, truncate to fixed lengths.
        input_tokens = input_tokens[:self.input_length]
        label_tokens = label_tokens[:self.target_length]
        return input_tokens, label_tokens
