import torch
from linear import CustomLinear
from embedding import CustomEmbedding
from mha import CustomMHA

'''
Complete this module which handles a single "block" of our model
as described in our lecture. You should have two sections with
residual connections around them:

1) norm1, mha
2) norm2, a two-layer MLP, dropout

It is perfectly fine to use pytorch implementations of layer norm and dropout,
as well as activation functions (torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.ReLU).

For layer norm, you just need to pass in D-model: self.norm1 = torch.nn.LayerNorm((d_model,))

'''
class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm((d_model,))
        self.norm2 = torch.nn.LayerNorm((d_model,))
        self.mha = CustomMHA(d_model, n_heads)
        self.linear1 = CustomLinear(d_model, 4 * d_model)
        self.linear2 = CustomLinear(4 * d_model, d_model)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.activation = torch.nn.ReLU()

    
    def forward(self, x):
        '''
        param x : (tensor) a tensor of size (batch_size, sequence_length, d_model)
        returns the computed output of the block with the same size.
        '''
        residual = x
        h = self.norm1(x)
        h = self.mha(h)
        x = residual + h

        residual = x
        h = self.norm2(x)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.linear2(h)
        h = self.dropout(h)
        x = residual + h

        return x


'''
Create a full GPT model which has two embeddings (token and position),
and then has a series of transformer block instances (layers). Finally, the last 
layer should project outputs to size [vocab_size].
'''
class GPTModel(torch.nn.Module):

    
    def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len):
        '''
        param d_model : (int) the size of embedding vectors and throughout the model
        param n_heads : (int) the number of attention heads, evenly divides d_model
        param layers : (int) the number of transformer decoder blocks
        param vocab_size : (int) the final output vector size
        param max_seq_len : (int) the longest sequence the model can process.
            This is used to create the position embedding- i.e. the highest possible
            position to embed is max_seq_len
        '''

        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = CustomEmbedding(vocab_size, d_model)
        self.position_embedding = CustomEmbedding(max_seq_len, d_model)
        self.layers = torch.nn.ModuleList(
            [TransformerDecoderBlock(d_model, n_heads) for _ in range(layers)]
        )
        self.output_projection = CustomLinear(d_model, vocab_size)

    
    def forward(self, x):
        '''
        param x : (long tensor) an input of size (batch_size, sequence_length) which is
            filled with token ids

        returns a tensor of size (batch_size, sequence_length, vocab_size), the raw logits for the output
        '''
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError("seq_len cannot be larger than max_seq_len")

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = position_ids.expand(batch_size, seq_len)
        h = self.token_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            h = layer(h)

        logits = self.output_projection(h)
        return logits




if __name__ == "__main__":

    # example of building the model and doing a forward pass
    D = 128
    H = 8
    L = 4
    model = GPTModel(D, H, L, 1000, 512)
    B = 32
    S = 48 # this can be less than 512, it just cant be more than 512
    x = torch.randint(1000, (B, S))
    y = model(x) # this should give us logits over the vocab for all positions

    # should be size (B, S, 1000)
    print(y)
