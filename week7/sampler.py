
import torch
import numpy as np

'''
Class implementing a sampler for inference on a model. Given the raw logits from
an LLM model, this will sample the next token id.
'''
class Sampler:

	def __init__(
		self,
		top_k=None,
		top_p=None,
		frequency_penalty=1.0,
		presence_penalty=1.0
	):
		'''
		param top_k : (None or int)
			If specified, only the top k logits should be used during sampling
			If this is specified, top_p should be None

		param top_p : (None or int)
			If specified, only the logits representing the probability mass p should be used during sampling.
			Or, if the top token has mass greater than p, the top token is returned.
			If this is specified, top_k should be None

		If top_k and top_p are both None, sample from the whole distribution (same as top_p=1.0)

		param frequency_penalty : (float)
			A penalty applied to tokens that have previously occured in the sequence. Along with
			presence_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.

		param presence_penalty : (float)
			A penalty applied to tokens IF they have previously occured in the sequence. Along with
			frequency_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.
		'''
		if top_k is not None and top_p is not None:
			raise ValueError("Only one of top_k or top_p may be specified.")

		if top_k is not None:
			if not isinstance(top_k, int):
				raise TypeError("top_k must be an int when specified.")
			if top_k <= 0:
				raise ValueError("top_k must be positive.")

		if top_p is not None:
			if not isinstance(top_p, (float, int)):
				raise TypeError("top_p must be a float when specified.")
			top_p = float(top_p)
			if not (0.0 < top_p <= 1.0):
				raise ValueError("top_p must be in the interval (0, 1].")

		if frequency_penalty <= 0.0:
			raise ValueError("frequency_penalty must be positive.")
		if presence_penalty <= 0.0:
			raise ValueError("presence_penalty must be positive.")

		self.top_k = top_k
		self.top_p = top_p
		self.frequency_penalty = float(frequency_penalty)
		self.presence_penalty = float(presence_penalty)


	def make_token_distribution(self, raw_unsorted_logits, previous_token_ids):
		'''
		param: raw_unsorted_logits (float numpy array)
			A one dimensional list of logits representing an unnormalized distribution over next tokens
			These are "unsorted" in the sense that their order aligns with vocabulary order, not with probability.

		param: previous_token_ids (int numpy array)
			A one dimensional list of ids representing the previous tokens, for calculating repetition penalties.

		returns:
			- the final probability distribution that this token is sampled from
			It should be returned back to token-id order (unsorted order) before returning.
		'''

		logits = np.asarray(raw_unsorted_logits, dtype=np.float64)
		if logits.ndim != 1:
			raise ValueError("raw_unsorted_logits must be a 1D array.")

		temps = np.ones_like(logits)

		if previous_token_ids is not None:
			prev_tokens = np.asarray(previous_token_ids, dtype=np.int64).reshape(-1)
			if prev_tokens.size:
				if np.any(prev_tokens < 0) or np.any(prev_tokens >= logits.shape[0]):
					raise ValueError("Token ids in previous_token_ids are out of range.")
				unique_tokens, counts = np.unique(prev_tokens, return_counts=True)
				for tok, count in zip(unique_tokens, counts):
					temps[tok] *= self.presence_penalty
					temps[tok] *= self.frequency_penalty ** count

		logits = logits - np.min(logits)
		scaled_logits = logits / temps

		# numerical stability for softmax
		stable_logits = scaled_logits - np.max(scaled_logits)
		exp_logits = np.exp(stable_logits)
		probs = exp_logits / np.sum(exp_logits)

		sort_indices_desc = np.argsort(probs)[::-1]
		sorted_probs = probs[sort_indices_desc]

		if self.top_k is not None:
			keep_count = min(self.top_k, sorted_probs.size)
			mask = np.zeros_like(sorted_probs, dtype=bool)
			mask[:keep_count] = True
		elif self.top_p is not None:
			cumulative = np.cumsum(sorted_probs)
			keep_count = np.searchsorted(cumulative, self.top_p, side='left') + 1
			keep_count = min(keep_count, sorted_probs.size)
			mask = np.zeros_like(sorted_probs, dtype=bool)
			mask[:keep_count] = True
		else:
			mask = np.ones_like(sorted_probs, dtype=bool)

		filtered_probs = np.where(mask, sorted_probs, 0.0)
		total = filtered_probs.sum()
		if total == 0.0:
			kept = np.nonzero(mask)[0]
			if kept.size == 0:
				raise RuntimeError("No tokens retained after filtering.")
			filtered_probs[kept] = 1.0 / kept.size
		else:
			filtered_probs /= total

		undo_sort_indices = np.argsort(sort_indices_desc)
		final_distribution = filtered_probs[undo_sort_indices]

		return final_distribution

		# very rough outline:
		# make temperature=1.0 for each vocabulary option
		# adjust temps as needed with penalties
		# logits = logits - np.min(logits) to make sure all are positive
		# apply temps & softmax
		# sort the distribution (and track the sort order so you can undo it later)
		# find either the top-p or top-k cutoff
		# renormalize this portion by simply dividing by the sum
		# revert back to original ordering of the distribution
			# helpful tip for this: 
			# indices = np.argsort(arr)
			# undo_indices = np.argsort(indices) # take argsort of the argsort
			# sorted_array = arr[indices]
			# put_back = sorted_array[undo_indices]
		# return distribution



	#==========================
	# for actually sampling the distribution
	def sample_one_token(self, raw_unsorted_logits, previous_token_ids):
		probs = self.make_token_distribution(raw_unsorted_logits, previous_token_ids)
		return np.random.choice(np.arange(len(raw_unsorted_logits)), p=probs)

	# for convenience, this is also callable
	def __call__(self, raw_unsorted_logits, previous_token_ids):
		return self.sample_one_token(raw_unsorted_logits, previous_token_ids)




if __name__ == "__main__":
    
    # example of using this with dummy data, keeping everything in token ids

    sampler = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

    sequence = [1,2,3,4,5]

    for i in range(10):
    	# fake logits for a vocab of size 500
    	logits = np.random.randn(500)

    	# get next token in sequence
    	next_token = sampler(logits, sequence)
    	sequence.append(next_token)

    print(sequence)
