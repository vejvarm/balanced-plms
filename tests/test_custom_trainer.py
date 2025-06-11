import torch
from pretraining.custom_trainer import CustomTrainer
from transformers import TrainingArguments

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])

def test_prediction_step_generates_tokens(tmp_path):
    model = DummyModel()
    trainer = CustomTrainer(model=model, args=TrainingArguments(output_dir=tmp_path.as_posix()))
    _, tokens, _ = trainer.prediction_step(model, {"input_ids": torch.tensor([[0, 1]])}, False)
    assert tokens.tolist() == [[1, 2, 3]]

