#!/usr/bin/python

"""
Functions for tokenizing string data.
"""

import re


class Token:
    def __init__(self, value, is_stop=False, is_unknown=False, parse_unk=False):
        if parse_unk and value.lower() == "*unk*":
            self.is_unknown = True
            self.value = None
            self.is_stop = False
            return

        self.value = value
        self.is_stop = is_stop
        self.is_unknown = is_unknown

    def __eq__(self, other):
        # check if both are stop (or only one)
        if self.is_stop and other.is_stop:
            return True
        elif self.is_stop or other.is_stop:
            return False

        # check if both are unknown (or only one)
        if self.is_unknown and other.is_unknown:
            return True
        elif self.is_unknown or other.is_unknown:
            return False

        # finally compare token values
        return self.value == other.value

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.value, self.is_stop, self.is_unknown))

    def __repr__(self):
        if self.is_stop:
            return "STOP"
        if self.is_unknown:
            return "*U*"
        return "%s" % self.value


STOP = Token(None, is_stop=True)
STOP_TAG = "STOP"
UNK = Token(None, is_unknown=True)


def tokenize_line(line, pad_start=False, pad_end=False, parse_unk=False):
    """
    Tokenizes a line of text by separating words using whitespace,
     and converting them to lower case.
     If line is empty, returns []
    :param line: The line to be tokenized
    :param pad_start: Whether to add a padding symbol at the start
    :param pad_end: Whether to add a padding symbol at the end
    :param parse_unk: Whether to convert the token "UNK" into an Uknown token, instead of the word "UNK".
    :return: The line as a list of tokens
    """
    tokens = [STOP] if pad_start else []
    new_tokens = [Token(w.lower(), parse_unk=parse_unk) for w in re.findall("[^\s]+", line)]
    if len(new_tokens) == 0:
        return []
    tokens += new_tokens
    if pad_end and len(new_tokens) > 0:
        tokens.append(STOP)
    return tokens


def tokenize_line_pos(line, pad_start=False, pad_end=False):
    """
    Tokenizes a line in which each word is followed by its part of speech.
     If line is empty, returns []
    :param line: The line to be tokenized
    :param pad_start: Whether to add a padding symbol at the start
    :param pad_end: Whether to add a padding symbol at the end
    :return: The line as a list of tuples of (token, tag) where tag is a string.
    """
    stop_pair = (STOP, "STOP")
    new_pairs = [stop_pair] if pad_start else []
    split_text = re.findall("[^\s]+", line)
    if len(split_text) < 2:
        return []

    for i in range(0, len(split_text), 2):
        word = split_text[i]
        tag = split_text[i+1]
        token = Token(word.lower())
        new_pairs.append((token, tag))
    if pad_end and len(split_text) > 0:
        new_pairs.append(stop_pair)
    return new_pairs


def tokenize_file_lines_pos(f):
    """
    Gievn a text file with words and parts of speech, tokenize each line separately.
    :param f: text file with each word followed by its part-of-speech tag, with words and tags
    separated by whitespace
    :return: A list of sentences, each sentence a padded list of tuples (token, tag)
    """
    tokenized_lines = []
    for l in f.readlines():
        sentence = tokenize_line_pos(l, pad_start=True, pad_end=True)
        if len(sentence) > 0:
            tokenized_lines.append(sentence)
    return tokenized_lines


def tokenize_file_lines(f, pad_start=False, pad_end=False, parse_unk=False):
    """
    Given a file, open the file and tokenize each line separately
    :param f: The file to be read
    :return: A list of sentences, each sentence a list of tokens. No stop symbols.
    """
    tokenized_lines = []
    for l in f.readlines():
        sentence = tokenize_line(l, pad_start=pad_start, pad_end=pad_end, parse_unk=parse_unk)
        if len(sentence) > 0:
            tokenized_lines.append(sentence)
    return tokenized_lines


def tokenize_file(f):
    """
    Given a filename, open the file and reads tokens (separated by whitespaces)
    into a list. Lines are considered to be sentences, and STOP tokens are
    added between sentences (and at the start and end of the corpus)
    :param f: The name of a text file
    :return: A list of tokens, with sentences separated by stop tokens
    """
    tokens = [STOP]
    for l in f.readlines():
        tokens += tokenize_line(l, pad_start=False, pad_end=True)
    return tokens
