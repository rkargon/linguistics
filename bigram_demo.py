#!/usr/bin/python
import random
import re
import sys

from linguistics.bigram import BigramModel
from linguistics.token import Token, STOP
from linguistics.unigram import UnigramModel

def tokenize_author_corpus(training_lines):
    word_split_regex = r"([a-z0-9]+|[\(\).?!'\"]+)"
    tokenized_lines = []
    for l in training_lines:
        words = re.findall(word_split_regex, l, re.I)
        tokens = [Token(w) for w in words]

        # add stop tokens where necessary
        tokens_processed = [STOP]
        for i in range(len(tokens)):
            t = tokens[i]
            if t.value in "\"(":
                if i == 0 or tokens[i-1] != STOP:
                    tokens_processed.append(STOP)
            tokens_processed.append(t)
            if t.value in ".?!)":
                tokens_processed.append(STOP)

        tokenized_lines.extend(tokens_processed)
    return tokenized_lines


def main():
    training_filename = sys.argv[1]
    with open(training_filename, "r") as training_file:
        corpus = tokenize_author_corpus(training_file.readlines())
    held_out_idx = int(len(corpus) * 0.9)
    training_corpus = corpus[:held_out_idx]
    held_out_corpus = corpus[held_out_idx:]
    # train unigram model to get optimized alpha value
    ugram_model = UnigramModel.create_optimized_unigram_model(training_corpus, held_out_corpus)
    # create bigram model using optimized unigram model for smoothing
    bigram_model = BigramModel.create_optimized_bigram_model(training_corpus, held_out_corpus, ugram_model=ugram_model)

    # generate text based on bigram model
    prev_token = STOP
    next_token = None
    while next_token != STOP:
        next_token = random.choice
    # TODO
if __name__ == '__main__':
    main()
