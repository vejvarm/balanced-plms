import torch
from pretraining.collators import DataCollatorForT5MLM

class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def pad(self, inputs, padding="max_length", return_tensors=None, max_length=None):
        padded = [ids + [self.pad_token_id] * (max_length - len(ids)) for ids in inputs["input_ids"]]
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}

    def convert_tokens_to_ids(self, token):
        return int(token[10:-1]) + 100

    def __call__(self, texts, add_special_tokens=False, truncation=False):
        return {"input_ids": [[i + 2 for i in range(len(text.split()))] for text in texts]}

def test_collator_shapes():
    tokenizer = DummyTokenizer()
    collator = DataCollatorForT5MLM(tokenizer=tokenizer, input_length=8, target_length=8, noise_density=0.5, mean_noise_span_length=2.0)
    examples = [{"input_ids": list(range(8))}, {"input_ids": list(range(8))}]
    batch = collator(examples)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)

