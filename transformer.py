# Adapted from CS 182 Transformers Starter Code
from typing import Optional, List
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F

import transformer_utils
from transformer_utils import EmbeddingTranspose, get_device
from transformer_attention import MultiHeadAttention

class PositionEmbedding(nn.Module):
    """
    Adds positional embedding to an input embedding.
    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self, hidden_size):
        super(PositionEmbedding, self).__init__()

        assert hidden_size % 2 == 0 and 'Model vector size must be even for sinusoidal encoding'
        power = torch.arange(0, hidden_size, step=2, dtype=torch.float32)[:] / hidden_size
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def forward(self, inputs, start=1):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        assert inputs.shape[-1] == self.hidden_size and 'Input final dim must match model hidden size'

        sequence_length = inputs.shape[1]

        # obtain a sequence that starts at `start` and increments for `sequence_length `
        seq_pos = torch.arange(start, sequence_length + start, dtype=torch.float32)  # 1-index positions
        seq_pos_expanded = seq_pos[None,:,None]
        index = seq_pos_expanded.repeat(*[1,1,self.hidden_size//2])

        # create the position embedding as described in the paper
        # use the `divisor` attribute instantiated in __init__
        sin_embedding = torch.sin(index / self.divisor)
        cos_embedding = torch.cos(index / self.divisor)

        # interleave the sin and cos. For more info see:
        # https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/3
        position_shape = (1, sequence_length, self.hidden_size)
        position_embedding = torch.stack((sin_embedding,cos_embedding), dim=3).view(position_shape)

        pos_embed_deviced = position_embedding.to(get_device())
        return inputs + pos_embed_deviced # add the embedding to the input
     
class TransformerFeedForward(nn.Module):
    def __init__(self, input_size,
                 filter_size,
                 hidden_size,
                 dropout):
        super(TransformerFeedForward, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.feed_forward = nn.Sequential(
                                nn.Linear(input_size,filter_size),
                                nn.ReLU(),
                                nn.Linear(filter_size,hidden_size)
                            )
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.feed_forward.apply(weights_init)
        self.dropout = nn.Dropout(0 if dropout is None else dropout)

    def forward(self, inputs):
        norm_input = self.norm(inputs)
        dense_out = self.feed_forward(norm_input)
        dense_out = self.dropout(dense_out) # Add the dropout here
        return dense_out + inputs # Add the residual here

class TransformerEncoderBlock(nn.Module):
    """An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf). 
    :param inputs: Tensor with shape [batch_size, sequence_length, channels]
    :return: output: Tensor with same shape as input
    """

    def __init__(self, input_size, n_heads,
                 filter_size, hidden_size, dropout = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.self_attention = MultiHeadAttention(n_heads,[input_size,input_size])
        self.feed_forward = TransformerFeedForward(input_size, filter_size, hidden_size, dropout)

    def forward(self, inputs, self_attention_mask=None):
        norm_inputs = self.norm(inputs)
        attn = self.self_attention((norm_inputs, norm_inputs), mask=self_attention_mask)
        res_attn = attn + inputs # Residual connection of the attention block
        output = self.feed_forward(res_attn)
        return output

class TransformerEncoder(nn.Module):
    """
    Stack of TransformerEncoderBlocks. Performs repeated self-attention.
    """

    def __init__(self, seq_input_size, # size of the input tokens (obs_dim + input_dim)
                embed_size, output_size, 
                n_layers, n_heads, d_filter, 
                dropout=None) -> None:
        super().__init__()
        self.embedding_layer = nn.Linear(seq_input_size, embed_size)
        self.positional_encoding = PositionEmbedding(hidden_size=embed_size)
        self.encoding_stack = []
        for i in range(n_layers):
            encoder = TransformerEncoderBlock(embed_size, n_heads, d_filter, embed_size, dropout)
            setattr(self,f"encoder{i}",encoder)
            self.encoding_stack.append(encoder)
        self.output_layer = nn.Linear(embed_size, output_size)

    def forward(self, inputs, encoder_mask=None):
        """
            Args:
                inputs: Either a float32 or int32 Tensor with shape [batch_size, sequence_length, ndim]
                encoder_mask: a boolean Tensor with shape [batch_size, sequence_length, sequence_length]
            Returns:
                output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        output = self.embedding_layer(inputs)
        output = self.positional_encoding(output)
        for encoder in self.encoding_stack:
            output = encoder(output, self_attention_mask=encoder_mask)
        output = self.output_layer(output)
        return output


class TransformerDecoderBlock(nn.Module):
    """A decoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: two Tensors encoder_outputs, decoder_inputs
                    encoder_outputs -> a Tensor with shape [batch_size, sequence_length, channels]
                    decoder_inputs -> a Tensor with shape [batch_size, decoding_sequence_length, channels]

    :return: output: Tensor with same shape as decoder_inputs
    """

    def __init__(self, input_size, n_heads,
                 filter_size, hidden_size,
                 dropout = None) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(input_size)
        self.self_attention = MultiHeadAttention(n_heads,[input_size,input_size])
        self.feed_forward = TransformerFeedForward(input_size, filter_size, hidden_size, dropout)

    def forward(self, decoder_inputs, self_attention_mask=None):

        # Compute the self-attention over the decoder inputs. This uses the self-attention
        # mask to control for the future outputs.
        # This generates a tensor of size [batch_size x target_len x d_model]
        norm_decoder_inputs = self.self_norm(decoder_inputs)
        target_selfattn = self.self_attention((norm_decoder_inputs, norm_decoder_inputs), mask=self_attention_mask)
        res_target_self_attn = target_selfattn + decoder_inputs
        output = self.feed_forward(res_target_self_attn)

        return output

class TransformerDecoder(nn.Module): # Note: there is no cross-attention here. It is just next-token prediction, GPT style.
    """
        Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
    """

    def __init__(self,
                seq_input_size, # size of input tokens (in our case, is obs_dim + input_dim)
                embed_size, output_size,
                n_layers, n_heads, d_filter,
                dropout = None) -> None:
        super().__init__()
        
        self.embedding_layer = nn.Linear(seq_input_size, embed_size) # Project into the higher dimension embedding size
        self.positional_encoding = PositionEmbedding(hidden_size=embed_size) # add the positional encoding
        self.decoding_stack = []
        for i in range(n_layers):
            decoder = TransformerDecoderBlock(embed_size, n_heads, d_filter, embed_size, dropout)
            setattr(self,f"decoder{i}",decoder)
            self.decoding_stack.append(decoder)
        self.output_layer = nn.Linear(embed_size, output_size)

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def forward(self, target_input, decoder_mask=None, mask_future=True, shift_target_sequence_right=False):
        """
            Args:
                target_input: either a int32 or float32 Tensor with shape [batch_size, target_length, ndims]
                mask_future: a boolean for whether to mask future states in target self attention

            Returns a Tensor with shape [batch_size, sequence_length, d_model]
        """
        # print("target input shape", target_input.shape)
        if shift_target_sequence_right:
            target_input = self.shift_target_sequence_right(target_input)
        # print("after shifting right: target input shape", target_input.shape)
        target_embedding = self.embedding_layer(target_input)

        # Build the future-mask (upper triangular) to prevent the network from attending to later timesteps
        batch_size = target_embedding.shape[0]
        sequence_length = target_embedding.shape[1]
        self_attention_mask = self.get_self_attention_mask(batch_size, sequence_length, decoder_mask, mask_future)

        # Pass it through the decoder stack and the final output layer
        decoder_output = self.positional_encoding(target_embedding)
        for decoder in self.decoding_stack:
            decoder_output = decoder(decoder_output, self_attention_mask=self_attention_mask)
        output = self.output_layer(decoder_output)
        return output

    def shift_target_sequence_right(self, target_sequence):
        const_val = 0 if target_sequence.dtype in [torch.int32, torch.int64] else 1e-10
        # pad_array is strange. 
        # The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward.
        # to pad the last 3 dimensions, use (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)

        pad_array = [0,0,1,0] # add one zero-pad on the left of the second dimension.
        target_sequence = F.pad(target_sequence, pad_array, value=const_val)[:, :-1]
        return target_sequence

    def get_future_mask(self, batch_size, sequence_length):
        """Mask future targets and padding

            :param batch_size: a Tensor dimension
            :param sequence_length: a Tensor dimension
            :param padding_mask: None or bool Tensor with shape [batch_size, sequence_length]

            :return mask Tensor with shape [batch_size, sequence_length, sequence_length]
        """

        xind = torch.arange(sequence_length)[None,:].repeat(*(sequence_length, 1))
        yind = torch.arange(sequence_length)[:,None].repeat(*(1, sequence_length))
        mask = yind >= xind
        mask = mask[None,...].repeat(*(batch_size, 1, 1))
        return mask.to(get_device())

    def get_self_attention_mask(self, batch_size, sequence_length, decoder_mask, mask_future):
        if not mask_future:
            return decoder_mask
        elif decoder_mask is None:
            return self.get_future_mask(batch_size, sequence_length)
        else:
            return decoder_mask & self.get_future_mask(batch_size, sequence_length)

    # This is an upper left block matrix which masks the attention for things that don't
    # exist within the internals.
    def get_cross_attention_mask(self, encoder_output, decoder_input, encoder_mask, decoder_mask):
        if encoder_mask is None and decoder_mask is None:
            cross_attention_mask = None
        elif encoder_mask is None:
            # We need to not mask the encoding, but mask the decoding
            # The decoding mask should have shape [batch_size x target_len x target_len]
            # meaning all we have to do is pad the mask out properly
            cross_attention_mask = decoder_mask[:, 1, :][:, None, :].repeat(
                                    *(1, encoder_output.shape[1], 1)).permute((0, 2, 1))
        elif decoder_mask is None:
            cross_attention_mask = encoder_mask[:, 1, :][:, :, None].repeat(
                                    *(1, 1, decoder_input.shape[1])).permute((0, 2, 1))
        else:
            dec_attention_mask = decoder_mask[:, 1, :][:, None, :].repeat(
                                    *(1, encoder_output.shape[1], 1)).permute((0, 2, 1))
            enc_attention_mask = encoder_mask[:, 1, :][:, :, None].repeat(
                                    *(1, 1, decoder_input.shape[1])).permute((0, 2, 1))
            cross_attention_mask = torch.logical_and(enc_attention_mask, dec_attention_mask)

        return cross_attention_mask
