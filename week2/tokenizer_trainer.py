'''
Your assignment is to implement BPE in the following method. You can add
classes or other routines if you find them helpful. 

This method should save two output files:
./vocab.txt : a list of the final vocabulary in order, one entry per line
./merges.json : a list of tuples of merges, in order

NOTE: Typically these file extensions are reversed (the vocabulary is a
json file and the merge list is a txt file), but for our purposes this way seems
simplier.

Does not need to return anything.

-------

This should implement a GPT-style tokenizer which prefixes words with a space.
You can assume that the base vocabulary contains all single characters that will occur.
Treat punctuation (besides spaces) just like the other characters.

You do NOT need to worry about using a placeholder token in place of a space. 
You do NOT need to worry about special tokens (pad, bos, eos, unk, etc.). We have not covered these yet.

IMPORTANT: If there are ties while computing the merges, you should use lexigraphic order to resolve.
Points will be taken off if a different tie-break is used as it will not match the homework solution.

For example, if the pairs ('ab','out') and ('sp','ite') are tied for most occuring,
then "about" should be recorded before "spite".

'''

from collections import Counter
from saving import save_vocab, save_merges

def _words_from_text(txt_file):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    return [tuple([" "] + list(w)) for w in words]

def _dedupe_preserve_order(seq):
    # dict.fromkeys keeps the first occurence of each item
    # python dicts preserve insertion order
    return list(dict.fromkeys(seq))

def _count_pairs(word_counts):
    # Count frequencies of adjacent pairs across the corpus
    pair_counts = Counter()
    for symbols, frequency in word_counts.items():
        has_pairs = len(symbols) >= 2
        if not has_pairs:
            continue
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i+1])] += frequency
    return pair_counts

def _apply_merge_to_word(symbols, left, right, merged_token):
    # Single pass, left-to-right merge of (left,right) -> merged_token
    out = []
    i = 0
    L = len(symbols)
    while i < L:
        is_merge = i < L - 1 and symbols[i] == left and symbols[i+1] == right
        if is_merge:
            out.append(merged_token)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)

def _merge(word_counts, left, right):
    # Apply merge to every distinct word sequence; re-aggregate counts
    merged_token = left + right
    new_word_counts = Counter()
    for symbols, frequency in word_counts.items():
        new_syms = _apply_merge_to_word(symbols, left, right, merged_token)
        new_word_counts[new_syms] += frequency
    return new_word_counts, merged_token

def train_tokenizer(txt_file, vocab_size, base_vocabulary):
    '''
    param : txt_file - a string path to a text file of data, i.e. "./data.txt"
    param : vocab_size - integer specifying the final vocab size
    param : base_vocabulary - list of strings to add to the vocabulary by default

    saves:
    ./vocab.txt : a list of the final vocabulary in order, one entry per line, ties broken alphabetically
    ./merges.json : a list of tuples of merges, in order
    '''
    
    # 1) Prepare corpus: words with leading " " symbol, and their counts
    word_syms = _words_from_text(txt_file)
    word_counts = Counter(word_syms)

    # 2) Initialize vocab (dedup, keep order) and merges list
    vocab = _dedupe_preserve_order(base_vocabulary)
    merges = []

    # 3) BPE loop
    while len(vocab) < vocab_size:
        pair_counts = _count_pairs(word_counts)

        nothing_left_to_merge = len(pair_counts) == 0
        if nothing_left_to_merge:
            break

        # Find max count; break ties lexicographically on the (left,right) pair
        max_count = max(pair_counts.values())
        candidates = [pair for pair, c in pair_counts.items() if c == max_count]
        best_left, best_right = min(candidates)  # Python tuple compare is lexicographic

        # Apply merge
        word_counts, new_token = _merge(word_counts, best_left, best_right)

        # Update vocab and history
        is_new_token = new_token not in vocab
        if is_new_token:
            vocab.append(new_token)

        merges.append((best_left, best_right))

        should_stop = len(vocab) >= vocab_size
        if should_stop:
            break

    save_vocab(vocab, "./vocab.txt")
    save_merges(merges, "./merges.json")



if __name__ == "__main__":

    # example of using this method.

    base = "abcdefghijklmnopqrstuvwxyz"
    base += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base += "0123456789"
    base += "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ "
    base += "\\"
    base += '"'

    train_tokenizer("./data.txt", len(base)+1000, [c for c in base])
