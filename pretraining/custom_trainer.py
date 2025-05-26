from transformers import Trainer
import torch

class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Remove the labels so that the generation does not attempt to compute loss or use them.
        if "labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}
        # Move the inputs to the same device as the model if not already.
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        # Call model.generate to obtain predictions.
        generated_tokens = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
        # Return None for loss and labels (since these are not computed during prediction)
        return (None, generated_tokens, None)
