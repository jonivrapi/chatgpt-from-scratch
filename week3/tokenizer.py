import json

'''
This class should be constructed with trained tokenizer data:
vocab_file : a string path to a vocab.txt file
merges_file : a string path to a merges.json file

The class should implement two methods:
encode(string): returns a list of integer ids (tokenized text)
decode(list_of_ids): returns a string re-assembled from token ids

You may assume that only a single sample is passed in at a time (no batching).
You can add additional methods, classes, etc as you find helpful.

Important: Our vocabulary and merges may include 
punctuation. Just treat all non-space characters equally.

---

Notes on validating your solution:

A good sanity check is that decode(encode(x)) should return x.

Additionally, make sure that the tokenizer is using the merges in order.
For example, if your merges contain: ("m","o"), ("s","e"), ("u","s"), then
"mouse" should be represented as mo|u|se.

'''

class Tokenizer:
    
    def __init__(self, vocab_file, merges_file):
        # TODO


    def encode(self, string):
        '''
        param string : a string to be encoded
        returns a list of integers (token ids)
        '''

        # TODO


    def decode(self, list_of_integers):
        '''
        param list_of_integers : a list of token ids
        returns a string formed by decoding these ids.
        '''

        # TODO



if __name__ == "__main__":

    # example of using this class

    tok = Tokenizer("./vocab.txt", "./merges.json")
    x = tok.encode("Peter piper picked a peck of pickled peppers.")
    print(x)
    x = tok.decode(x)
    print(x) # should be our original text.
