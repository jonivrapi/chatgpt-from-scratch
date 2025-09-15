import torch

'''
Complete this class by instantiating a parameter called "self.weight", and
use it to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            # should be a random uniform distribution between -1 and 1
            torch.rand(num_embeddings, embedding_dim) * 2 - 1
        )

    def forward(self, x: torch.Tensor):
        return self.weight[x.long()]