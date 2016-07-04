#!/usr/bin/python

import math

from linguistics.token import UNK
from linguistics.util import golden_section_search


class UnigramModel:
    """
    Represents a unigram model for language modeling.
    The model is trained, with additive smoothing, on the given training when initialized.
     The model can also
    """
    def __init__(self, training_data, alpha=1):
        # get unigram counts
        self.unigram_counts = {}
        self.token_count = 0
        self.word_type_count = 0
        self.unigram_log_params = {}
        self.alpha = alpha

        self.token_count = len(training_data)
        for t in training_data:
            self.unigram_counts[t] = self.unigram_counts.get(t, 0) + 1
        self.unigram_counts[UNK] = 0
        self.word_type_count = len(list(self.unigram_counts.keys()))

        # train unigram frequencies with a certain alpha
        self.train_with_alpha(alpha)

    @staticmethod
    def create_optimized_unigram_model(training_corpus, held_out_corpus):
        """
        Create a unigram model trained on the given data, with an optimized smoothing parameter
        alpha found using golden section search.
        :param training_corpus: The training corpus
        :param held_out_corpus: The corpus of held out data on which to train alpha
        :return: A trained, optimized unigram model
        """
        alpha_min = 0
        alpha_max = 2
        unigram_model = UnigramModel(training_corpus)

        def update_model(alpha):
            unigram_model.train_with_alpha(alpha)
            return unigram_model.get_document_log_probability(held_out_corpus)

        # minimize negative log probability of held out data for parameter alpha
        #  if negative log probability is low, then the actualy probability is high,
        #  which is what we want.
        optimal_alpha = golden_section_search(update_model, alpha_min, alpha_max, maximize=False)
        unigram_model.train_with_alpha(optimal_alpha)
        return unigram_model

    @staticmethod
    def additive_smoothing(n_w, n_o, alpha, total_w):
        """
        Calculates a unigram frequency parameter for a word using additive smoothing
        :param n_w: The counts of the word in question
        :param n_o: The total word count
        :param alpha: The alpha parameter
        :param total_w: The number of total word types
        :return: theta_w, an estimation of the probability of a unigram occuring in a document
        """
        return (n_w + alpha) / float(n_o + alpha * total_w)

    def get_token_probability(self, token):
        neg_log_prob = self.get_token_log_probability(token)
        return math.exp(-neg_log_prob)

    def get_token_log_probability(self, token):
        """
        Returns the negative log probability of a ungiram occurring in a document.
        :param token: The given token to be checked
        :return: The negative log probability of the token ocurring
        """
        return self.unigram_log_params.get(token, self.unigram_log_params[UNK])

    def get_document_log_probability(self, document):
        """
        Returns the negative log likelihood of a given document according to this unigram model.
        :param document: The document in question, as a list of tokens
        :return: THe sum of negative log probabilities of each token
        """
        return sum(map(self.get_token_log_probability, document))

    def train_with_alpha(self, alpha):
        """
        Trains parameters for the unigram model using a new alpha,
         but without changing the training data and word counts.
        :param alpha: The new alpha value to use
        """
        for w, n in list(self.unigram_counts.items()):
            freq = self.additive_smoothing(n_w=n, n_o=self.token_count, alpha=alpha, total_w=self.word_type_count)
            self.unigram_log_params[w] = -math.log(freq)
        self.alpha = alpha
