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

import json

class Tokenizer:
    def __init__(self, vocab_file: str, merges_file: str):
        # Load vocabulary
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.id_to_token = [line.rstrip("\n") for line in f]
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

        # Load merges
        with open(merges_file, "r", encoding="utf-8") as f:
            merges = json.load(f)
        # Give each pair a rank where earlier = higher priority
        self.rank = {(a, b): i for i, (a, b) in enumerate(merges)}

    def encode(self, text: str):
        tokens = list(text)

        # Repeatedly apply the best-ranked merge present
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

            best = None
            best_rank = None
            for p in pairs:
                rank = self.rank.get(p)
                rank_exists = rank is not None
                if rank_exists:
                    rank_is_better = (best_rank is None) or (rank < best_rank)
                    if rank_is_better:
                        best, best_rank = p, rank

            no_more_merges = best is None
            if no_more_merges:
                break

            # Merge all occurrences of the best pair, left-to-right
            merged = []
            i = 0
            n = len(tokens)
            while i < n:
                has_next = i < n - 1
                next_pair = (tokens[i], tokens[i + 1]) if has_next else None
                next_pair_is_best = has_next and (next_pair == best)

                if next_pair_is_best:
                    merged.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        # Map final tokens to ids
        ids = []
        for t in tokens:
            ids.append(self.token_to_id[t])
        return ids

    def decode(self, ids):
        return "".join(self.id_to_token[i] for i in ids)



if __name__ == "__main__":
    tok = Tokenizer("./vocab.txt", "./merges.json")
    s = "Peter piper picked a peck of pickled peppers."
    ids = tok.encode(s)
    print(ids)
    print(tok.decode(ids))  # should equal s
    print(s == tok.decode(ids))

