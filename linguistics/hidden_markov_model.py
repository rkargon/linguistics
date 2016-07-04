#!/usr/bin/python
from .token import UNK, STOP_TAG


class HMM:
    """
    Represents a hidden markov model used for part-of-speech (or other) tagging. Given a list of tagged sentences,
    computes hidden markov parameters which can be used to assign most likely tags to other strings using
    Viturbi's algorithm.
    """
    def __init__(self, tagged_sentences, rare_word_smoothing=False):
        """
        Initializes a Hidden Markov model using training data of tagged sentences.
        This estimates hmm parameters using the training data so that the model can be used to tag other text.
        :param tagged_sentences:  A list of sentences, each sentence a list of (token, tag) pairs
        :param rare_word_smoothing: Whether or not to smooth by considering words that only occur once as "unknown"
        words. If this is False, then each part of speech has a count of 1 for "UNKNOWN" added to its transition (tau)
        values.
        """
        self.vocabulary = {}
        self.tags = set()
        self.sigma = {}
        self.tau = {}

        self.build_vocabulary(tagged_sentences, rare_word_smoothing)
        self.estimate_hmm_params(tagged_sentences, rare_word_smoothing)

    def build_vocabulary(self, tagged_sentences, rare_word_smoothing=False):
        """
        Given a list of tagged sentences, built a set of unique words, as well as a set of unique tags in the corpus
        :param tagged_sentences: A list of sentences, each sentence a list of (token, tag) pairs
        :param rare_word_smoothing: Whether or not to smooth by considering words that only occur once as "unknown"
        words. If this is False, then each part of speech has a count of 1 for "UNKNOWN" added to its transition (tau)
        values.
        """
        for sentence in tagged_sentences:
            for w, t in sentence:
                self.vocabulary[w] = self.vocabulary.get(w, 0) + 1
                self.tags.add(t)

        # when using rare word smoothing, remove words that have a count of 1 (they are considered "unknown")
        if rare_word_smoothing:
            self.vocabulary = {w: c for w, c in self.vocabulary.items() if c != 1}

    def estimate_hmm_params(self, tagged_sentences, rare_word_smoothing):
        """
        Estimates parameters for this hidden markov model using a set of tagged sentences
        :param tagged_sentences: A list of sentences, each sentence a list of (token, tag) pairs
        :param rare_word_smoothing: Whether or not to smooth by considering words that only occur once as "unknown"
        words. If this is False, then each part of speech has a count of 1 for "UNKNOWN" added to its transition (tau)
        values.
        """
        # counts for transitions between states. Used to estimate sigma values.
        n_yy = {tag: {tag: 0 for tag in self.tags} for tag in self.tags}
        # counts for which words correspond to which states. Used to estimate tau values.
        # note: - we add a pseudo-count of 1 for the unknown token for smoothing
        #       - when using rare word smoothing, the default count is 0
        n_yx = {tag: {UNK: 0 if rare_word_smoothing else 1} for tag in self.tags}

        # get parameter counts
        for sentence in tagged_sentences:
            for i in range(0, len(sentence)-1):
                (word, tag), (word_next, tag_next) = sentence[i:i+2]
                # increment tag and word transition values
                n_yy[tag][tag_next] += 1
                if word not in self.vocabulary:
                    n_yx[tag][UNK] += 1
                else:
                    n_yx[tag][word] = n_yx[tag].get(word, 0) + 1

        # estimate parameters from counts
        for y in self.tags:
            sigma_counts_total = float(sum(n_yy[y].values()))
            tau_counts_total = float(sum(n_yx[y].values()))

            self.sigma[y] = {ynext: n_yy[y][ynext] / sigma_counts_total for ynext in n_yy[y]}
            self.tau[y] = {x: n_yx[y][x] / tau_counts_total for x in n_yx[y]}

    def get_tau_value(self, tag, word):
        """
        Returns the probability that a certain tag produces a given word.
        :param tag: The tag (hidden variable) in question
        :param word: The given word
        :return: A probability that `word` will be produced given `tag`
        This is 0 if the word exists in the model's vocabulary and simply not seen with the given tag,
        or it's P(tag, UNK) if the word is unknown.
        """
        if word not in self.vocabulary:
            return self.tau[tag][UNK]
        return self.tau[tag].get(word, 0)

    def tag_sentence(self, sentence):
        """
        Returns a list of the most likely tags for a sentence using the Viterbi algorithm
        :param sentence: A list of tokens (with padding on both sides)
        :return: A list of tags as strings
        """

        # initialize probability values with initial stop state
        mu_i_y = [{STOP_TAG: (None, 1)}]

        # for each word in sentence
        for i in range(1, len(sentence)):
            # maps possible tags to tuple (prev_tag, probability)
            tag_probs = {}
            # for each possible tag
            for y in self.tags:
                best_prob = (None, 0)
                tau = self.get_tau_value(y, sentence[i])
                # find previous tag with best probability
                for y1 in mu_i_y[i-1]:
                    sigma = self.sigma[y1][y]
                    _, prev_prob = mu_i_y[i-1][y1]
                    prob = prev_prob * sigma * tau

                    if prob > best_prob[1] or best_prob[0] is None:
                        best_prob = (y1, prob)
                tag_probs[y] = best_prob
            mu_i_y.append(tag_probs)

        # iterate backwards through sentence, follow back pointers of best tags
        tags = [""] * len(sentence)
        best_tag = STOP_TAG
        for i in range(len(tags) - 1, -1, -1):
            tags[i] = best_tag
            prev_tag, _ = mu_i_y[i][best_tag]
            best_tag = prev_tag

        return tags
