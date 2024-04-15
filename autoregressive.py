import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from model import LanguageModel
from utils import *

class AutoregressiveWrapper(torch.nn.Module):
    """
    Pytorch module that wraps a GPT model and makes it autoregressive.
    """

    def __init__(self, decoder_model):
        super().__init__()
        self.decoder_model = decoder_model
        
    def forward(self, x, padding_mask):
        """
        Autoregressive forward pass
        """
        inp, target = x[:, :-1], x[:, 1:]
        padding_mask = padding_mask[:, :-1]

        output = self.decoder_model(inp, padding_mask)
        return output, target

    def next_token_probabilities(self, x, padding_mask, temperature=1.0):
        """
        Calculate the token probabilities for the next token in the sequence.
        """
        logits = self.decoder_model(x, padding_mask)[:, -1]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply the softmax
        probabilities = torch.softmax(logits, dim=-1)

        return probabilities

    def save_checkpoint(self, path):
        self.decoder_model.save_checkpoint(path)

    @staticmethod
    def load_checkpoint(path) -> 'AutoregressiveWrapper':
        decoder_model = LanguageModel.load_checkpoint(path)
        return AutoregressiveWrapper(decoder_model).to(get_device())




class Generator:

    def __init__(
            self,
            generator_model,
            tokenizer):
        self.generator_model = generator_model
        self.tokenizer = tokenizer
        
    def generate(
            self,
            max_tokens_to_generate: int,
            prompt: str = None,
            temperature: float = 1.0,
            eos_token: int = None,
            padding_token: int = 0):

        self.generator_model.eval()

        if prompt is None:
            start_tokens = [self.tokenizer.character_to_token(padding_token)]
        else:
            start_tokens = self.tokenizer.tokenize(prompt)

        input_tensor = torch.tensor(
            pad_left(
                sequence=start_tokens,
                final_length=self.generator_model.decoder_model.max_sequence_length,
                padding_token=padding_token
            ),
            dtype=torch.long
        ).to(get_device())

        num_dims = len(input_tensor.shape)

        if num_dims == 1:
            input_tensor = input_tensor[None, :] # add batch dimension

        out = input_tensor
        for _ in range(max_tokens_to_generate):

            x = out[:, -self.generator_model.decoder_model.max_sequence_length:]
            padding_mask = torch.ones_like(x)
            padding_mask[x == padding_token] = 0

            # Compute the next token probabilities
            next_token_probabilities = self.generator_model.next_token_probabilities(
                x = x,
                temperature = temperature,
                padding_mask = padding_mask
            )

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(next_token_probabilities, num_samples=1)

            # Append the next token to the output
            out = torch.cat([out, next_token], dim=1)

            # If the end of sequence token is reached, stop generating tokens
            if eos_token is not None and next_token == eos_token:
                break

        generated_tokens = out[0].tolist()
        return ''.join([self.tokenizer.token_to_character(token) for token in generated_tokens])








