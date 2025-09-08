# test_tokenizer.py
import os
from collections import Counter
from saving import save_vocab, save_merges

# import your functions (adjust if your file/module name differs)
from tokenizer_trainer import train_tokenizer

BASE = "abcdefghijklmnopqrstuvwxyz" \
       "ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
       "0123456789" \
       "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ " + "\\" + '"'
BASE_LIST = [c for c in BASE]

def read_vocab(path="./vocab.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def read_merges(path="./merges.json"):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_text(fname, text):
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

def test_tiny_he_tiebreak():
    fname = "toy_he.txt"
    write_text(fname, "he he he\n")
    # ask for only a few extra merges
    vocab_size = len(BASE_LIST) + 2
    train_tokenizer(fname, vocab_size, BASE_LIST)

    merges = read_merges()
    assert len(merges) >= 1, "Should perform at least one merge."
    first = tuple(merges[0])
    # Tie between (' ', 'h') and ('h','e') with count 3; lexicographically choose (' ', 'h')
    assert first == (" ", "h"), f"Expected first merge (' ', 'h'), got {first}"

    vocab = read_vocab()
    assert vocab[:len(BASE_LIST)] == BASE_LIST, "Base vocab must be first and in order."
    assert " h" in vocab, "New merged token ' h' should be added to vocab."

def test_repeated_char():
    fname = "toy_aaa.txt"
    write_text(fname, "aaa aa a")
    vocab_size = len(BASE_LIST) + 3
    train_tokenizer(fname, vocab_size, BASE_LIST)

    merges = read_merges()
    assert len(merges) >= 1
    # Tie between (' ', 'a') and ('a','a') → lexicographic picks (' ', 'a')
    assert tuple(merges[0]) == (" ", "a"), f"Expected (' ', 'a'), got {merges[0]}"

def test_mixed_words_first_merge_space_h():
    fname = "toy_hello_world.txt"
    write_text(fname, "hello hello world")
    vocab_size = len(BASE_LIST) + 3
    train_tokenizer(fname, vocab_size, BASE_LIST)

    merges = read_merges()
    assert len(merges) >= 1
    assert tuple(merges[0]) == (" ", "h"), f"Expected (' ', 'h'), got {merges[0]}"

def test_no_pairs_no_crash():
    # Single-character words only → no pairs inside words (besides leading spaces)
    fname = "toy_single_chars.txt"
    write_text(fname, "a b c d")
    vocab_size = len(BASE_LIST) + 5
    train_tokenizer(fname, vocab_size, BASE_LIST)

    merges = read_merges()
    # The only possible pairs are (' ', 'a'), (' ', 'b'), ... which exist; so at least one merge should happen.
    # To test true "no pairs", try an empty file:
    fname2 = "toy_empty.txt"
    write_text(fname2, "")
    vocab_size2 = len(BASE_LIST) + 5
    train_tokenizer(fname2, vocab_size2, BASE_LIST)
    merges2 = read_merges()
    assert len(merges2) == 0, "Empty corpus should produce zero merges."

def test_vocab_growth_and_order():
    fname = "toy_order.txt"
    write_text(fname, "hello hello")
    extra = 5
    vocab_size = len(BASE_LIST) + extra
    train_tokenizer(fname, vocab_size, BASE_LIST)

    vocab = read_vocab()
    merges = read_merges()
    # new tokens should be appended in the same order as merges
    new_tokens = vocab[len(BASE_LIST):]
    # Each new token should be a concatenation of the merge pair strings at its step or derived from later merges.
    # At minimum, counts match: number of new vocab entries equals number of merges or the requested extra.
    assert 0 < len(new_tokens) <= extra, "New vocab should be >0 and <= requested extra merges."
    assert len(merges) == len(new_tokens), "One new token per merge step."

if __name__ == "__main__":
    # run tests manually without pytest
    test_tiny_he_tiebreak()
    test_repeated_char()
    test_mixed_words_first_merge_space_h()
    test_no_pairs_no_crash()
    test_vocab_growth_and_order()
    print("All tests passed ✅")
