import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from model import LanguageModel
from utils import *
from autoregressive import *
import matplotlib.pyplot as plt



class Runner(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def run(self):
        # Create the tokenizer
        tokenizer = Tokenizer()

        embedding_dimension = 256
        max_sequence_length = 20
        number_of_tokens = tokenizer.size()

        # Create the generator_model
        generator_model = AutoregressiveWrapper(LanguageModel(
            embedding_dimension=embedding_dimension,
            number_of_tokens=number_of_tokens,
            number_of_heads=4,
            number_of_layers=3,
            dropout_rate=0.1,
            max_sequence_length=max_sequence_length
        )).to(get_device())

        # Create the training data
        training_data = '. '.join([
            'cats rule the world',
            'dogs are the best',
            'elephants have long trunks',
            'monkeys like bananas',
            'pandas eat bamboo',
            'tigers are dangerous',
            'zebras have stripes',
            'lions are the kings of the savannah',
            'giraffes have long necks',
            'hippos are big and scary',
            'rhinos have horns',
            'penguins live in the arctic',
            'polar bears are white'
        ])

        # 这里是直接把所有语句全部链接为一句话, 然后遍历将每个字符转换为 index
        tokenized_and_padded_training_data = tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data)
        # 这里是滑动窗口 append 序列, 只不过把输入和输出放在一起, [input[1] , 共用部分序列 , output[-1] ]
        sequences = create_training_sequences(max_sequence_length, tokenized_and_padded_training_data)

        # Train the generator_model
        optimizer = torch.optim.Adam(generator_model.parameters(), lr=0.0001)
        trainer = Trainer(generator_model, tokenizer, optimizer)
        loss_per_epoch = trainer.train(sequences, epochs=30, batch_size=16)

        # Plot the loss per epoch in log scale
        plt.plot(loss_per_epoch)
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        generator_model.save_checkpoint('./trained_model')

        # Generate text
        max_tokens_to_generate = 256
        generator = Generator(generator_model, tokenizer)
        generated_text = generator.generate(
            max_tokens_to_generate=max_tokens_to_generate,
            prompt="cats",
            padding_token=tokenizer.character_to_token('<pad>')
        )
        print(generated_text.replace('<pad>', ''))

Runner().run()


