import torch
from tqdm import tqdm
import numpy as np
from hftokenizer import HFTokenizer


def construct_dataset(data_txt_file, sequence_length=256):
    '''
    data_txt_file : a string path to a text file containing training data, one sample per line
    sequence_length : int, the desired length of each training sequence

    This method should use the trained tokenizer to convert samples to token_ids, and
    then pack them into a training set represented as a 2D array of size (sequences, sequence_length+1).
    The +1 is very important! It lets us compare our model outputs to the sequence shifted by one.

    You can save this training set in whatever format you wish for loading into the training script.
    I recommend using numpy's np.save() method or the pickle module.

    The saved data should be shuffled so we can directly load it and train on it in the training script.
    '''

    # construct tokenizer
    tokenizer = HFTokenizer()
    tokenizer.load()

    # get all samples
    f = open(data_txt_file, "r")
    samples = f.readlines()
    samples = [x.replace("\n", "") for x in samples]

    # ----------------------------------------
    # TODO
    # - add '<|endoftext|>' to each sample or add tokenizer.eos_token_id after tokenizing.
    # - use tokenizer.encode() to tokenize each sample
    # - pack into sequences of length sequence_length
    # - shuffle
    # - save out data
    eos_id = tokenizer.tokenizer.eos_token_id
    token_stream = []
    for sample in samples:
        token_ids = tokenizer.encode(sample)
        token_ids.append(eos_id)
        token_stream.extend(token_ids)

    token_stream = np.array(token_stream, dtype=np.int64)

    seq_len_with_target = sequence_length + 1
    usable_tokens = (token_stream.shape[0] // seq_len_with_target) * seq_len_with_target
    token_stream = token_stream[:usable_tokens]
    sequences = token_stream.reshape(-1, seq_len_with_target)

    rng = np.random.default_rng()
    rng.shuffle(sequences)

    np.save("dataset.npy", sequences)


if __name__ == "__main__":
    construct_dataset("./data.txt", 256)
