#!/usr/bin/python
import sys

from .hidden_markov_model import HMM
from .token import tokenize_file_lines_pos, STOP


def tag(train, test, tag_type):
    """
    Trains a hidden markov model using the given training data set and then assigns tags to the testing set.
    Test sentences with tags are output to the standard out.
    :param train: Filename of a training corpus consisting of one sentence per line. Each sentence is a series of
    words each followed by their tag, separated by whitespace.
    :param test: A testing corpus structured similarly (with tags as well).
    :param tag_type: Which smoothing method to use for training the model. Either "tag" or "better_tag"
    """
    rare_word_smoothing = (tag_type == "better_tag")

    with open(train, "r") as ftrain:
        training_corpus = tokenize_file_lines_pos(ftrain)
    with open(test, "r") as ftest:
        test_corpus = tokenize_file_lines_pos(ftest)

    hmm = HMM(training_corpus, rare_word_smoothing=rare_word_smoothing)

    # tag each sentence and print output to standard out.
    for sentence in test_corpus:
        tokens = [t[0] for t in sentence]
        new_tags = hmm.tag_sentence(tokens)
        # print out words and their new tags to stdout (excluding STOP symbols)
        print(" ".join("%s %s" % (w, t) for w, t in zip(tokens, new_tags) if w != STOP))


def score(reference, test):
    """
    Checks the tagging of a corpus relative to a reference. Outputs the accuracy.
    :param reference: The tagged reference corpus
    :param test: The same corpus, but with tags produced by the markov model
    """
    with open(reference, "r") as freference:
        reference_corpus = tokenize_file_lines_pos(freference)
    with open(test, "r") as ftest:
        test_corpus = tokenize_file_lines_pos(ftest)

    n_correct = 0
    n_total = 0
    for ref_sentence, tst_sentence in zip(reference_corpus, test_corpus):
        n_total += len(ref_sentence)
        for i in range(len(ref_sentence)):
            ref_word, ref_tag = ref_sentence[i]
            tst_word, tst_tag = tst_sentence[i]
            n_correct += (ref_tag == tst_tag)

    print("%d out of %d correct, %f accuracy." % (n_correct, n_total, n_correct/float(n_total)))


def usage():
    """
    Prints out a usage string and exits the program.
    """
    usage_str = "Usage:\n" \
                "python pos.py [tag|better_tag] <train> <test>\n" \
                "python pos.py score <reference> <test>\n"
    print(usage_str)
    exit(0)


def main():
    command = sys.argv[1]
    args = sys.argv[2:]
    if len(args) != 2:
        usage()

    if command == "tag" or command == "better_tag":
        train, test = args
        tag(train, test, tag_type=command)
    elif command == "score":
        reference, test = args
        score(reference, test)
    else:
        usage()


if __name__ == '__main__':
    main()
