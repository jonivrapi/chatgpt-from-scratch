import torch

'''
Complete this class by instantiating parameters called "self.weight" and "self.bias", and
use them to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomLinear(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super().__init__()
		self.weight = torch.nn.Parameter(0.1*torch.randn(input_size, output_size))
		self.bias = torch.nn.Parameter(torch.randn(output_size))

	def forward(self, x):
		'''
		x is a tensor contain a batch of vectors, size (B, input_size).
		This should return a tensor of size (B, output_size).
		'''
		return x @ self.weight + self.bias