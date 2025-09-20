import torch
import math

'''
Complete this module such that it computes queries, keys, and values,
computes attention, and passes through a final linear operation W_o.

You do NOT need to apply a causal mask (we will do that next week).
If you don't know what that is, don't worry, we will cover it next lecture.

Be careful with your tensor shapes! Print them out and try feeding data through
your model. Make sure it behaves as you would expect.
'''
class CustomMHA(torch.nn.Module):

	'''
	param d_model : (int) the length of vectors used in this model
	param n_heads : (int) the number of attention heads. You can assume that
		this even divides d_model.
	'''
	def __init__(self, d_model, n_heads):
		super().__init__()
		# TODO
		# please name your parameters "self.W_qkv" and "self.W_o" to aid in grading
		# self.W_qkv should have shape (3D, D)
		# self.W_o should have shape (D,D)

	'''
	param x : (tensor) an input batch, with size (batch_size, sequence_length, d_model)
	returns : a tensor of the same size, which has had MHA computed for each batch entry.
	'''
	def forward(self, x):
		# TODO

		# SUGGESTED OUTLINE:
		#----------

		# use W_qkv to get queries Q, keys K, values V, each of shape (B,S,D)

		# reshape these into size (B, h, S, D/h)

		# compute QK^T and divide by sqrt(D/h)

		# apply softmax

		# matrix multiply against values

		# reshape (B,h,S,D/h) into (B,S,D)

		# matrix multiply against output projection W_o

		#---------




if __name__ == "__main__":

	# example of building and running this class
	mha = CustomMHA(128,8)

	# 32 samples of length 6 each, with d_model at 128
	x = torch.randn((32,6,128))
	y = mha(x)
	print(x.shape, y.shape) # should be the same