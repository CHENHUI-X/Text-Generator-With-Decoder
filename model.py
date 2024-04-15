import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from utils import *

class TokenEmbedding(torch.nn.Module):
    """
    为 Vocabulary 中的所有 token 添加 embedding

    Input dimension is: (batch_size, sequence_length)
    Output dimension is: (batch_size, sequence_length, d_model)
    """

    def __init__(self, d_model, number_of_tokens):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=number_of_tokens,
            embedding_dim=d_model
        )

    def forward(self, x):
        return self.embedding_layer(x)


class PositionalEmbedding(nn.Module):

    # register buffer in Pytorch ->
    # If you have parameters in your model, which should be saved and restored in the state_dict,
    # but not trained by the optimizer, you should register them as buffers.

    def __init__(self, embed_model_dim, max_seq_len ):
        """
        Args:
            seq_len: max length of input sequence
            embed_model_dim: demension of embedding
        """
        self.embed_model_dim = embed_model_dim
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_seq_len, embed_model_dim)

        # notice the unsqueeze operation in this code ,
        # it add a dimension so that when position * div_term
        # it get a broadcast multiplication
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(embed_model_dim)) / embed_model_dim))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        self.register_buffer('pe', pe)
        # 位置信息不需要train

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_model_dim) # Maybe do not need this ?
        # add constant to embedding
        seq_len = x.size(1) # ( batch ,seq_len , embedding dim)
        x = x + self.pe[:seq_len].repeat(x.size(0),1,1)
        # " repeat(x.size(0),1,1) "for every sequence do the same thing
        return x




class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        self.single_head_dim = int(self.embed_dim / self.n_heads)
        self.query_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim,bias=False)
        self.key_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False)

        self.out = nn.Linear(
            self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, x, padding_mask = None , future_mask = None):

        """
        Args:
            x : input tensor with shape (B , L , D)
            
            padding_mask : with dimensions (batch_size, sequence_length). 
            It is used to mask padding tokens so that normal tokens do not attend to padding tokens.
            mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.

            future_mask : with dimensions ( sequence_length , sequence_length) 
            It is used to mask feature tokens so that current tokens do not attend to feature tokens.
            mask values are: 0 or 1. where 0 means the token is masked, 1 means the token is not masked , that's can be attented.

        Returns:
           output vector from multihead attention
        """

        batch_size = x.size(0)
        seq_length = x.size(1)
        qkv_x = x.view(
            batch_size, seq_length, self.n_heads,self.single_head_dim)

        k = self.key_matrix(qkv_x)  
        q = self.query_matrix(qkv_x)  
        v = self.value_matrix(qkv_x)  

        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)  
         
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1, -2)

        product = torch.matmul(q, k_adjusted)   # (batch_size, n_heads, seq_len, seq_len)

        if future_mask is not None:
            future_mask = future_mask.expand(batch_size, 1, seq_length, seq_length).to(x.device)
            mask = future_mask # (batch_size, n_heads, seq_
            
            if padding_mask is not None:
                padding_mask = padding_mask.reshape(batch_size, 1, 1, seq_length).to(x.device)
                mask = future_mask + padding_mask
                
            product = product.masked_fill(mask.gt(1), float("-1e20")) # 广播操作
        else:
            raise ValueError("In decoder, you must set feature mask")


        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        # that normalize the "weight" for a line
        scores = F.softmax(product, dim=-1)

        # mutiply with value matrix
        scores = torch.matmul(scores, v)

        # concatenated output
        concat = scores.transpose(1, 2).contiguous().view(
            batch_size, seq_length,self.single_head_dim * self.n_heads)

        output = self.out(concat)  
        return output


class FeedForward(torch.nn.Module):
    """
    Pytorch module for a feed forward layer.

    A feed forward layer is a fully connected layer with a ReLU activation function in between.
    """

    def __init__(self, embedding_dimension, feed_forward_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.feed_forward_dimension = feed_forward_dimension
        self.linear_1 = torch.nn.Linear(embedding_dimension, feed_forward_dimension)
        self.linear_2 = torch.nn.Linear(feed_forward_dimension, embedding_dimension)

    def forward(self, x):
        """
        Compute the feed forward layer.
        """
        return self.linear_2(torch.relu(self.linear_1(x)))


class DecoderLayer(torch.nn.Module):
    """
    Pytorch module for an encoder layer.

    An encoder layer consists of a multi-headed self attention layer, a feed forward layer and dropout.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate

        self.multi_headed_self_attention = MultiHeadAttention(embedding_dimension, number_of_heads)
        self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_normalization_1 = torch.nn.LayerNorm(embedding_dimension)
        self.layer_normalization_2 = torch.nn.LayerNorm(embedding_dimension)

    def forward(self, x,  padding_mask , future_mask ):
        """
        Compute the encoder layer.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        
        """

        # Layer normalization 1
        normalized_x = self.layer_normalization_1(x)

        # Multi headed self attention
        attention_output = self.multi_headed_self_attention(normalized_x, padding_mask , future_mask)

        # Residual output
        residual_output = x + attention_output

        # Layer normalization 2
        normalized_residual_output = self.layer_normalization_2(residual_output)

        # Feed forward
        feed_forward_output = self.feed_forward(normalized_residual_output)

        # Dropout
        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)

        # Residual output
        return residual_output + feed_forward_output



class DecoderStack(torch.nn.Module):
    """
    The decoder stack consists of multiple decoder layers in sequence.
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_layers,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate,
            max_sequence_length
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Create the encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [DecoderLayer(embedding_dimension, number_of_heads, feed_forward_dimension, dropout_rate) for _ in
             range(number_of_layers)])

    def forward(self, x,  padding_mask , future_mask ):
        r'''
            args :
                x dimensions are: (batch_size, sequence_length, embedding_dimension)
                padding_mask:   with dimensions (batch_size, sequence_length). 
                It is used to mask padding tokens so that normal tokens do not attend to padding tokens.
                mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.

                future_mask :   with dimensions ( sequence_length , sequence_length) 
                It is used to mask feature tokens so that current tokens do not attend to feature tokens.
                mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked , that's can be attented.
        '''
        decoder_outputs = x
        for decoder_layer in self.encoder_layers:
            decoder_outputs = decoder_layer(decoder_outputs,  padding_mask , future_mask )

        return decoder_outputs


class LMHead(torch.nn.Module):
    """
    Pytorch module for the language model head.
    The language model head is a linear layer that maps the embedding dimension to the vocabulary size.
    """

    def __init__(self, embedding_dimension, number_of_tokens):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.linear = torch.nn.Linear(embedding_dimension, number_of_tokens)

    def forward(self, x):
        """
        Compute the language model head.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        output dimensions are: (batch_size, sequence_length, number_of_tokens)
        """
        # Compute the linear layer
        # linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
        linear_output = self.linear(x)

        return linear_output



class LanguageModel(torch.nn.Module):
    """
    Pytorch module for a language model.
    """

    def __init__(
            self,
            number_of_tokens,  # The number of tokens in the vocabulary
            max_sequence_length=512,  # The maximum sequence length to use for attention
            embedding_dimension=512,  # The dimension of the token embeddings
            number_of_layers=6,  # The number of decoder layers to use
            number_of_heads=4,  # The number of attention heads to use
            feed_forward_dimension=None,  # The dimension of the feed forward layer
            dropout_rate=0.1  # The dropout rate to use
    ):
        super().__init__()
        self.number_of_tokens = number_of_tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.future_mask = torch.tril(torch.ones((max_sequence_length, max_sequence_length))) # for future mask

        if feed_forward_dimension is None:
            # GPT-2 paper uses 4 * embedding_dimension for the feed forward dimension
            self.feed_forward_dimension = embedding_dimension * 4
        else:
            self.feed_forward_dimension = feed_forward_dimension

        self.dropout_rate = dropout_rate

        # Create the token embedding layer
        self.token_embedding = TokenEmbedding(embedding_dimension, number_of_tokens)

        # Create the positional encoding layer
        self.positional_encoding = PositionalEmbedding(embedding_dimension, max_sequence_length)

        # Create the normalization layer
        self.layer_normalization = torch.nn.LayerNorm(embedding_dimension)

        # Create the decoder stack
        self.decoder = DecoderStack(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            feed_forward_dimension=self.feed_forward_dimension,
            dropout_rate=dropout_rate,
            max_sequence_length=max_sequence_length
        )

        # Create the language model head
        self.lm_head = LMHead(embedding_dimension, number_of_tokens)

    def forward(self, x, padding_mask):
        
        r'''
        args :
            x : with dimensions (batch_size , sequence_length) ,like [[1,34,32 is token index] ...[] * batch_size]
            padding_mask:   with dimensions (batch_size, sequence_length). 
            It is used to mask padding tokens so that normal tokens do not attend to padding tokens.
            mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.

            future_mask :   with dimensions ( sequence_length , sequence_length) 
            It is used to mask feature tokens so that current tokens do not attend to feature tokens.
            mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked , that's can be attented.
        '''
        
        # Compute the token embeddings
        # token_embeddings dimensions are: (batch_size, sequence_length, embedding_dimension)
        token_embeddings = self.token_embedding(x)

        # Compute the positional encoding
        # positional_encoding dimensions are: (batch_size, sequence_length, embedding_dimension)
        positional_encoding = self.positional_encoding(token_embeddings)

        # Post embedding layer normalization
        positional_encoding_normalized = self.layer_normalization(positional_encoding)

        decoder_outputs = self.decoder(positional_encoding_normalized,  padding_mask , self.future_mask )
        lm_head_outputs = self.lm_head(decoder_outputs)

        return lm_head_outputs

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'number_of_tokens': self.number_of_tokens,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dimension': self.embedding_dimension,
            'number_of_layers': self.number_of_layers,
            'number_of_heads': self.number_of_heads,
            'feed_forward_dimension': self.feed_forward_dimension,
            'dropout_rate': self.dropout_rate,
            'model_state_dict': self.state_dict()
        }, path)

    @staticmethod
    def load_checkpoint(path) -> 'LanguageModel':
        checkpoint = torch.load(path)
        model = LanguageModel(
            number_of_tokens=checkpoint['number_of_tokens'],
            max_sequence_length=checkpoint['max_sequence_length'],
            embedding_dimension=checkpoint['embedding_dimension'],
            number_of_layers=checkpoint['number_of_layers'],
            number_of_heads=checkpoint['number_of_heads'],
            feed_forward_dimension=checkpoint['feed_forward_dimension'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(get_device())














