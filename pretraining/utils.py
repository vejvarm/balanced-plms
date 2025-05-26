# Credit : gpt-4o , Claude-3.5-Sonnet-200k , Gemini-Pro-1.5

# Reference :
# [Protein Discovery with Discrete Walk-Jump Sampling](http://arxiv.org/abs/2306.12360)
# [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](http://arxiv.org/abs/2407.01392)


import torch
import numpy

import random

from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

# the models have been trained / finetuned, run inference code only
INFERENCE_ONLY = 0

# Just for code development / debugging purpose
TEST_OVERFIT = 0

# for the denoiser module, choose only ONE of the following options :
USE_PRETRAINED_BERT = 0
USE_PRETRAINED_BERT_MLM = 0
USE_PRETRAINED_T5 = 0
USE_CUSTOM_TRANSFORMER_ENCODER = 0  # the most RAM memory efficient option
USE_CUSTOM_TRANSFORMER_ENCODER_DECODER = 0

# Early-stopping for the models training
USE_EARLY_STOP = 0
EARLY_STOP_THRESHOLD = 2.175  #1.91

# for sentence completion downstream task
ENABLE_MASK_LEARNING = 1

# google colab T4 GPU does not have a lot of RAM for computation
# custom transformer module can now handle multiple masked tokens
if torch.cuda.is_available():  #or USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER:
    MASK_RATIO = 0.15   # use 0.15 for 15% masking probability, use the value of -1 to indicate only a single masked token
else:
    MASK_RATIO = 0.15   # use 0.15 for 15% masking probability, use the value of -1 to indicate only a single masked token

# allows the denoiser model to train on [batch_size, sequence_length, vocab_size]
USE_LOGITS_FOR_THE_ENTIRE_SENTENCE = 1
USE_LOGITS_FOR_THE_ENTIRE_SENTENCE = USE_LOGITS_FOR_THE_ENTIRE_SENTENCE or (MASK_RATIO != -1)  # if masking more than 1 token, then it makes sense to train on [batch_size, sequence_length, vocab_size]
# custom transformer module can now handle multiple masked tokens
#USE_LOGITS_FOR_THE_ENTIRE_SENTENCE = USE_LOGITS_FOR_THE_ENTIRE_SENTENCE and not (USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER)

# Analyze walk-jump's output samples for debugging purpose
ENABLE_SAMPLE_ANALYSIS = 0  # turns off for reducing memory consumption

if torch.cuda.is_available():
    device_str = "cuda"
else:
    device_str = "mps"

device = torch.device(device_str)

# Automatic Mixed Precision for training
from torch import autocast
if torch.cuda.is_available():
    from torch.amp import GradScaler
    USE_MIXED_PRECISION_TRAINING = 0  # optional, turns off for this code since it hurts model performance
else:
    USE_MIXED_PRECISION_TRAINING = 0  # not implemented


# for saving RAM memory during training : https://github.com/zyushun/Adam-mini
USE_ADAM_MINI = 0

# 0: Sinusoidal Positional Embedding , 1: Rotary Positional Embedding
USE_ROPE = 0

# Just for code development / debugging purpose
USE_DUMMY_TRAINING_DATA = 0

# for adjusting the generation process due to fixed output length
GENERATES_OUTPUT_OF_VARYING_LENGTH = 0

# for more difficult denoising task
ADD_EXTRA_GAUSSIAN_NOISE = 0  # turns off for now

# Select between diffusion forcing and walk-jump
# if the following two variables are turned off, it would be walk-jump (single constant noise level)
USE_DIFFUSION_FORCING = 1 & ADD_EXTRA_GAUSSIAN_NOISE
USE_PRECOMPUTE_NOISE_SCHEDULE = 0  # testing only, do not recommend to use due to expensive storage

# Regarding two different approaches for Langevin MCMC sampling
USE_MCMC = 1
USE_ALGORITHM_1_OR_4 = 0  # value of 1 means Algorithm 1, value of 0 means Algorithm 4, see walk-jump paper
USE_OBABO = 0  # Using KIPLMC2 is slow because of the need to compute gradients of U with respect to both theta and X

# sequential monte-carlo (SMC)
USE_SMC = 0  # if use SMC, then ignore USE_ALGORITHM_1_OR_4 which is related to Langevin MCMC

# Markov-approximate fractional Brownian motion (MA-fBM)
USE_MAFBM = 0  # if use MAFBM, then ignore USE_ALGORITHM_1_OR_4 which is related to Langevin MCMC

# Once turned on, it will be different from the walk-jump denoise update equation
USE_LOGITS_FOR_DENOISING = 0  # consumes much more RAM memory
USE_LOGITS_FOR_DENOISING = USE_LOGITS_FOR_DENOISING and (USE_SMC or USE_MAFBM or USE_MCMC)

# kl_div method (requires extra run of denoiser model) to improve sampling based on prior distribution
# Only turn on USE_GRAD_KL if USE_PRETRAINED_T5 is disabled, because USE_GRAD_KL uses
# "tokenizer.vocab_size"-rounds of denoiser module execution, hence extremely long execution time.
# Using large pretrained T5 model as denoiser module will only worsen the runtime issue.
USE_GRAD_KL = 0

# Choose only one of the following training receipes for walk-jump sampling
USE_dWJS_ENERGY = 1
USE_dWJS_SCORE = ~USE_dWJS_ENERGY

# Define parameters
input_dim = 512
model_dim = input_dim
model_dim_ebm = model_dim >> 2  # specific only to EBM model
hidden_dim = 256
num_layers = 4
num_layers_ebm = num_layers >> 1  # specific only to EBM model
num_heads = 8
num_heads_ebm = num_heads >> 2  # specific only to EBM model
num_smc_steps = 5  # sequential monte-carlo (SMC)
N_particles = 10  # sequential monte-carlo (SMC)
hurst = 0.7  # Markov-approximate fractional Brownian motion (MA-fBM)
T_fbm = 1.0  # Markov-approximate fractional Brownian motion (MA-fBM)
n_steps = 1000  # Markov-approximate fractional Brownian motion (MA-fBM)
K_fbm = 3  # Markov-approximate fractional Brownian motion (MA-fBM)
num_walk_steps = 5  # for langevin dynamics MCMC sampling process
num_jump_steps = 20  #num_walk_steps
walk_step_size = 0.6  # for langevin dynamics MCMC sampling process
sigma_max = 1.1
sigma_min = 0.1
num_epochs = 500
batch_size = 512


if USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:
    # BERT model is larger than TransformerDenoiser() module
    batch_size = batch_size >> 6

elif USE_PRETRAINED_T5:
    # T5 models are way larger than both BERT model and TransformerDenoiser() module
    batch_size = 1

elif USE_CUSTOM_TRANSFORMER_ENCODER_DECODER:
    # we have extra decoder layers inside the TransformerDenoiser() module
    batch_size = batch_size >> 4

else:  # USE_CUSTOM_TRANSFORMER_ENCODER
    # we do not have extra decoder layers inside the TransformerDenoiser() module
    batch_size = batch_size >> 3


#if torch.cuda.is_available():  # so far colab run session has some extra unused GPU RAM on T4 GPU
#    batch_size = batch_size << 2  # increasing batch_size worsens the validation loss convergence rate


# Monitors the quality of the generated samples throughout the training and validation
# processes to assess the model's performance and identify potential issues
def analyze_samples(generated_samples, tokenizer, skip_special_tokens=False, num_samples=1):
    decoded_samples = []
    if num_samples != 1:
        num_samples = generated_samples.size(0)

    for i in range(num_samples):
        sample = generated_samples[i]
        sample = sample.long()  # Convert the sample to integer tensor
        decoded_sample = tokenizer.decode(sample, skip_special_tokens=skip_special_tokens)
        print(f"Sample {i+1}: {decoded_sample}")
        decoded_samples.append(decoded_sample)

    return decoded_samples

def assert_sample_range_compliance(sample, tokenizer):
    # Assert that all token IDs are within the valid range
    assert sample.min() >= 0, f"Token ID is less than 0!  sample = {sample}, sample.min() = {sample.min()}"
    assert sample.max() < tokenizer.vocab_size, f"Token ID exceeds valid range! Max ID: {sample.max()}, Vocab Size: {tokenizer.vocab_size}"

    # Assert that the tokens input to the model are not all zeros
    assert not torch.all(sample == 0), "Error: sample contains all zeros!"
    return True

def check_for_vanishing_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            if grad_norm < 1e-5:  # Threshold for detecting vanishing gradients
                print(f"Warning: Vanishing gradient detected in {name} with norm {grad_norm.item():.6f}")

if USE_PRETRAINED_T5: #or USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER:
    tokenizer = AutoTokenizer.from_pretrained("pnawrot/nanoT5-base")
    #tokenizer = T5Tokenizer.from_pretrained('google/t5-efficient-tiny')
else:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenizer_function(raw_sequence_input, tokenizer, max_length=input_dim):
    tokenized_sequence = tokenizer(
                            raw_sequence_input,
                            padding='max_length',
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                         )

    return tokenized_sequence.to(device)


#print(f"tokenizer.pad_token_id = {tokenizer.pad_token_id}")
# for initializing target_label for denoiser module
CONSTANTS_VALUE_IGNORE = tokenizer.pad_token_id  # -100

# for creating data loader for span-masking task
class DataCollatorForSpanCorruption:
    def __init__(self, tokenizer, mlm_probability=0.15, mean_noise_span_length=3, input_length=input_dim):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length

    def __call__(self, examples):
        # If examples are tensors, convert them to lists
        if isinstance(examples[0], torch.Tensor):
            input_ids = [example.tolist() for example in examples]
            attention_mask = None  # No attention mask for tensor inputs
        else:
            # Assuming examples are dicts with 'input_ids' keys
            input_ids = [example['input_ids'] for example in examples]
            attention_mask = [example['attention_mask'] for example in examples] if 'attention_mask' in examples[0] else None

        batch = self._collate_batch(input_ids)

        # Add attention mask if it exists
        if attention_mask is not None:
            batch['attention_mask'] = pad_sequence(
                [mask.clone().detach() for mask in attention_mask],
                batch_first=True,
                padding_value=0
            )

        return batch

    def _collate_batch(self, input_ids_list):
        # Pad input_ids to the same length
        batch_input_ids = pad_sequence(
            [ids.clone().detach() for ids in input_ids_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        # Create masked inputs and labels
        if USE_PRETRAINED_T5:
            masked_input_ids, labels, mlm_mask = self._mask_tokens_span(batch_input_ids)
            return {'input_ids': masked_input_ids, 'labels': labels, 'mask_indices': mlm_mask}

        elif USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:
            #labels, mlm_mask = self._mask_tokens_span(batch_input_ids)
            labels, mlm_mask = self._mask_tokens_standard(batch_input_ids)
            return {'input_ids': batch_input_ids, 'labels': labels, 'mask_indices': mlm_mask}

        else:  # USE_CUSTOM_TRANSFORMER_ENCODER or USE_CUSTOM_TRANSFORMER_ENCODER_DECODER
            #labels, mlm_mask = self._mask_tokens_span(batch_input_ids)
            labels, mlm_mask = self._mask_tokens_standard(batch_input_ids)
            return {'input_ids': batch_input_ids, 'labels': labels, 'mask_indices': mlm_mask}

    # span-masking strategy
    def _mask_tokens_span(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked span language modeling according to T5's objective.
        """
        inputs = inputs.clone()
        labels = torch.full(inputs.shape, self.tokenizer.pad_token_id)
        special_tokens = {self.tokenizer.pad_token_id}

        batch_size, seq_len = inputs.shape
        mask_indices = []

        # Track masking locations
        mask_indices_tensor = torch.zeros_like(inputs, dtype=torch.bool)

        for i in range(batch_size):
            input_ids = inputs[i].tolist()
            num_to_mask = max(1, int(round(seq_len * self.mlm_probability)))

            # Get candidate indices to mask
            candidate_indices = [
                idx for idx in range(len(input_ids)) if input_ids[idx] not in special_tokens
            ]

            # Shuffle candidate indices
            random.shuffle(candidate_indices)

            masked_indices = set()
            current_idx = 0
            spans = []
            while len(masked_indices) < num_to_mask and current_idx < len(candidate_indices):
                span_length = max(1, int(numpy.random.poisson(lam=self.mean_noise_span_length)))
                start = candidate_indices[current_idx]
                end = min(start + span_length, seq_len)
                span_indices = list(range(start, end))

                # Avoid overlapping spans
                if any(idx in masked_indices for idx in span_indices):
                    current_idx += 1
                    continue

                masked_indices.update(span_indices)
                spans.append((start, end))
                current_idx += 1

            # Sort spans in reverse order to avoid index shifting issues
            spans = sorted(spans, key=lambda x: x[0], reverse=True)

            target_tokens = []
            prev_end = seq_len
            for idx, (start, end) in enumerate(spans):
                # Replace span with sentinel token in inputs
                if USE_PRETRAINED_T5:
                    sentinel_token_id = self.tokenizer.convert_tokens_to_ids(f'<extra_id_{idx}>')
                else:
                    sentinel_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                inputs[i, start:end] = sentinel_token_id
                # Build labels
                target_tokens = [sentinel_token_id] + input_ids[start:end] + target_tokens

            # Record the masked positions
            for start, end in spans:
                mask_indices_tensor[i, start:end] = True

            # Handle unmasked positions in labels
            #labels[~mask_indices_tensor] = CONSTANTS_VALUE_IGNORE
            #labels[i, :len(target_tokens)] = torch.tensor(target_tokens, dtype=torch.long)

            # debug prints
            if len(spans) > 0:
                total_masked = sum(end - start for start, end in spans)
                #print(f"Sequence {i}: Created {len(spans)} spans, masking {total_masked} tokens")
                #print(f"Spans: {spans}")

        if USE_PRETRAINED_T5:
            return inputs, labels, mask_indices_tensor  # T5 masking tokens are not unique, so need to return masked "inputs"
        else:
            return labels, mask_indices_tensor  # Return the mask information

    # standard BERT masking strategy without any span-masking
    def _mask_tokens_standard(self, inputs):
        """
        Prepare masked tokens inputs/labels for standard masked language modeling (e.g., BERT).
        """
        labels = inputs.clone()

        # Create a mask for tokens to mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels for masked tokens, set CONSTANTS_VALUE_IGNORE for others
        #labels[~masked_indices] = CONSTANTS_VALUE_IGNORE  # We only compute loss on masked tokens

        # Replace masked input tokens according to BERT's strategy
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, device=device, dtype=torch.long)
        indices_random = indices_random.to(device)
        inputs[indices_random] = random_words[indices_random]

        # The rest 10% of the time, keep the original token (do nothing)

        return labels, masked_indices