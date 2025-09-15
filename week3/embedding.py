import torch

'''
Complete this class by instantiating a parameter called "self.weight", and
use it to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomEmbedding(torch.nn.Module):

	def __init__(self, num_embeddings, embedding_dim):
		super().__init__()
		# TODO

	def forward(self, x):
		# x is a tensor of integers
		# TODO
		