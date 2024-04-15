import random
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from typing import *


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_left(sequence, final_length, padding_token):
    return [padding_token] * (final_length - len(sequence)) + sequence



def create_training_sequences(max_sequence_length, tokenized_training_data):
    # Create sequences of length max_sequence_length + 1
    # The last token of each sequence is the target token
    # 输入输出放到一起, 用 1~n 预测 2~n+1
    sequences = []
    for i in range(0, len(tokenized_training_data) - max_sequence_length - 1):
        sequences.append(tokenized_training_data[i: i + max_sequence_length + 1])
    return sequences


def tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data):
    # Tokenize the training data
    tokenized_training_data = tokenizer.tokenize(training_data)
    
    # 这里在整个序列最开始 insert 0 的作用是, 想把前 max_len个 token也用上, 比如第一个token, 就是用 max_len个0 预测这第一个token
    # 不过这里, 我理解其实也可以直接用 1 - max_len 个 预测 max_len + 1个, 不管最一开始的 max_len个
    for _ in range(max_sequence_length):
        # Prepend padding tokens
        tokenized_training_data.insert(0, tokenizer.character_to_token('<pad>'))
    return tokenized_training_data


class Tokenizer:
    r'''
        0-9 (10 个 token) , a-z (26 个 token) , ' ' , ',' '.' ,  '<pad>' 共40个token
    '''
    def __init__(self):
        self.dictionary = {}
        self.reverse_dictionary = {}

        # Add the padding token
        self.__add_to_dict('<pad>')

        # Add characters and numbers to the dictionary
        for i in range(10):
            self.__add_to_dict(str(i))
        for i in range(26):
            self.__add_to_dict(chr(ord('a') + i))

        # Add space and punctuation to the dictionary
        self.__add_to_dict(',')
        self.__add_to_dict('.')
        self.__add_to_dict(' ')

    def __add_to_dict(self, character):
        if character not in self.dictionary:
            self.dictionary[character] = len(self.dictionary)
            self.reverse_dictionary[self.dictionary[character]] = character

    def tokenize(self, text):
        return [self.dictionary[c] for c in text]

    def character_to_token(self, character):
        return self.dictionary[character]

    def token_to_character(self, token):
        return self.reverse_dictionary[token]

    def size(self):
        return len(self.dictionary)



def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self, model, tokenizer: Tokenizer, optimizer=None):
        super().__init__()
        self.model = model
        self.seq_len = self.model.decoder_model.max_sequence_length
        if optimizer is None:
            # 因为是引用, 所以更新 self.model 就是更新了 model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        else:
            self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.loss_function = torch.nn.CrossEntropyLoss()

    def train(self, data: List[str], epochs, batch_size):
        
        loss_per_epoch = []
        for epoch in range(epochs):
            losses = []

            # Shuffle the sequences
            random.shuffle(data)

            # Create batches of sequences and their respective mask.
            batches = []
            for i in range(0, len(data), batch_size):
                sequence_tensor = torch.tensor(data[i: i + batch_size], dtype=torch.long)

                # Create the mask tensor for the batch, where 1 means the token is not a padding token
                padding_mask = torch.ones_like(sequence_tensor)
                padding_mask[sequence_tensor == self.tokenizer.character_to_token('<pad>')] = 0

                batches.append((sequence_tensor, padding_mask))

            # Train the model on each batch
            for batch in batches:
                self.model.train()

                # Create the input and mask tensors
                # 这里为什么是 seq_len + 1 呢, 是把输入输出放到一起了, 然后用的时候, 输入就是 input_tensor[:seq_len], 输出就是 input_tensor[1:seq_len+1]
                input_tensor = torch.zeros((batch_size, self.seq_len + 1), dtype=torch.long)
                padding_mask = torch.zeros((batch_size, self.seq_len + 1), dtype=torch.long)

                for i, input_entry in enumerate(batch[0]):
                    input_tensor[i] = input_entry

                for i, mask_entry in enumerate(batch[1]):
                    padding_mask[i] = mask_entry

                # Compute the model output
                model_output, target = self.model.forward(
                    x = input_tensor.to(get_device()),
                    padding_mask = padding_mask.to(get_device())
                )

                # Compute the losses
                # The loss is computed on the model output and the target
                # model_output shape with (batch_size , seq_len , vocabulary_size)
                # for high dimension , need transpose input to (batch_size, vocabulary_size , ... )
                loss = self.loss_function(model_output.transpose(1, 2), target)
                # loss = self.loss_function(model_output[:, -1, :], target[:, -1])

                # Backpropagate the loss.
                loss.backward()

                # Clip the gradients. This is used to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                # Update the model parameters. This is done by taking a step in the direction of the gradient.
                self.optimizer.step()

                # Reset the gradients. This is done so that the gradients from the previous batch
                # are not used in the next step.
                self.optimizer.zero_grad()

                # Append the loss to the list of losses, so that the average loss can be computed for this epoch.
                losses.append(loss.item())

            # Print the loss
            epoch_loss = np.average(losses)
            loss_per_epoch.append(epoch_loss)
            print('Epoch:', epoch, 'Loss:', epoch_loss)

        return loss_per_epoch
