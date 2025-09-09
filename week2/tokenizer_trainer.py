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

def _words_from_text_file(text_file_path):
    with open(text_file_path, "r", encoding="utf-8") as file_handle:
        full_text = file_handle.read()
    words_list = full_text.split()
    return [tuple([" "] + list(word)) for word in words_list]

def _deduplicate_preserving_order(sequence):
    return list(dict.fromkeys(sequence))

def _count_adjacent_pairs(word_count_map):
    # Count frequencies of adjacent pairs across the corpus
    adjacent_pair_counts = Counter()
    for symbol_sequence, count in word_count_map.items():
        has_adjacent_pairs = len(symbol_sequence) >= 2
        if not has_adjacent_pairs:
            continue
        for index in range(len(symbol_sequence) - 1):
            pair = (symbol_sequence[index], symbol_sequence[index + 1])
            adjacent_pair_counts[pair] += count
    return adjacent_pair_counts

def _apply_merge_to_symbol_sequence(symbol_sequence, left_symbol, right_symbol, merged_token):
    # Single pass, left-to-right merge of (left_symbol, right_symbol) to merged_token
    output_sequence = []
    index = 0
    length_of_sequence = len(symbol_sequence)
    while index < length_of_sequence:
        is_merge_site = (
            index < length_of_sequence - 1
            and symbol_sequence[index] == left_symbol
            and symbol_sequence[index + 1] == right_symbol
        )
        if is_merge_site:
            output_sequence.append(merged_token)
            index += 2
        else:
            output_sequence.append(symbol_sequence[index])
            index += 1
    return tuple(output_sequence)

def _merge_across_corpus(word_count_map, left_symbol, right_symbol):
    # Apply merge to every distinct word sequence then re-aggregate counts
    merged_token = left_symbol + right_symbol
    new_word_count_map = Counter()
    for symbol_sequence, count in word_count_map.items():
        new_symbol_sequence = _apply_merge_to_symbol_sequence(
            symbol_sequence, left_symbol, right_symbol, merged_token
        )
        new_word_count_map[new_symbol_sequence] += count
    return new_word_count_map, merged_token

def train_tokenizer(text_file_path, target_vocabulary_size, base_vocabulary):
    '''
    param : text_file_path - a string path to a text file of data, i.e. "./data.txt"
    param : target_vocabulary_size - integer specifying the final vocab size
    param : base_vocabulary - list of strings to add to the vocabulary by default

    saves:
    ./vocab.txt : a list of the final vocabulary in order, one entry per line, ties broken alphabetically
    ./merges.json : a list of tuples of merges, in order
    '''
    # Prepare corpus words with leading " " symbol and their counts
    word_symbol_sequences = _words_from_text_file(text_file_path)
    word_count_map = Counter(word_symbol_sequences)

    # Initialize vocabulary (deduplicate while keeping order)
    vocabulary_list = _deduplicate_preserving_order(base_vocabulary)
    merge_operations = []

    # The BPE loop
    while len(vocabulary_list) < target_vocabulary_size:
        adjacent_pair_counts = _count_adjacent_pairs(word_count_map)

        nothing_left_to_merge = len(adjacent_pair_counts) == 0
        if nothing_left_to_merge:
            break

        # Find max count and break ties lexicographically on the (left_symbol, right_symbol) pair
        maximum_count = max(adjacent_pair_counts.values())
        tied_candidates = [
            pair for pair, count in adjacent_pair_counts.items() if count == maximum_count
        ]
        best_left_symbol, best_right_symbol = min(tied_candidates)

        # Apply the merge
        word_count_map, new_token = _merge_across_corpus(
            word_count_map, best_left_symbol, best_right_symbol
        )

        # Update vocabulary and history
        is_new_token = new_token not in vocabulary_list
        if is_new_token:
            vocabulary_list.append(new_token)

        merge_operations.append((best_left_symbol, best_right_symbol))

        should_stop = len(vocabulary_list) >= target_vocabulary_size
        if should_stop:
            break

    save_vocab(vocabulary_list, "./vocab.txt")
    save_merges(merge_operations, "./merges.json")


if __name__ == "__main__":

    # example of using this method.

    base_characters = "abcdefghijklmnopqrstuvwxyz"
    base_characters += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base_characters += "0123456789"
    base_characters += "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ "
    base_characters += "\\"
    base_characters += '"'

    train_tokenizer("./data.txt", len(base_characters) + 1000, [character for character in base_characters])
