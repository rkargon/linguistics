#!/usr/bin/python
import math

from linguistics.unigram import UnigramModel
from linguistics.util import golden_section_search


class BigramModel:
    def __init__(self, training_data, beta=1, alpha=1, ugram_model=None):
        self.beta = beta
        if ugram_model is not None:
            self.ugram_model = ugram_model
        else:
            self.ugram_model = UnigramModel(training_data, alpha=alpha)
        self.alpha = self.ugram_model.alpha
        self.bigram_counts = {}
        self.bigram_log_params = {}

        # get counts of each (w1, w2) in corpus
        for i in range(1, len(training_data)):
            w1 = training_data[i-1]
            w2 = training_data[i]
            if w1 not in self.bigram_counts:
                self.bigram_counts[w1] = {}
            self.bigram_counts[w1][w2] = self.bigram_counts[w1].get(w2, 0) + 1

        # get counts of all bigrams (w1, *) for each w1
        self.bigram_totals = {w1: sum(self.bigram_counts[w1].values()) for w1 in list(self.bigram_counts.keys())}

    @staticmethod
    def create_optimized_bigram_model(training_corpus, held_out_corpus, alpha=1, ugram_model=None):
        """
        Creates a bigram model with beta optimized on the given
        held out corpus.
        :param training_corpus: The corpus to train bigram counts
        :param held_out_corpus: The held out corpus on which to optimize beta
        :param alpha: The alpha to use for the unigram model
        :return: The optimized bigram model
        """
        beta_min = 0
        beta_max = 200
        bigram_model = BigramModel(training_corpus, alpha=alpha, ugram_model=ugram_model)

        def update_model(beta):
            bigram_model.set_beta(beta)
            return bigram_model.get_document_log_probability(held_out_corpus)

        optimal_beta = golden_section_search(update_model, beta_min, beta_max, maximize=False)
        bigram_model.set_beta(optimal_beta)
        return bigram_model

    def set_beta(self, beta):
        """
        Change the beta for this model.
        This clears out all the cached  bigram parameters, which will be re-calculated using
        the new beta next time the model is used.
        :param beta: The new value of beta
        """
        self.beta = beta
        self.bigram_log_params.clear()

    def get_bigram_log_probability(self, bigram):
        """
        Returns negative log probability of a bigram under the given bigram model,
        using additive smoothing
        :param bigram: The bigram in question
        :return: the negative log probability of the given bigram
        """
        w1, w2 = bigram
        if bigram not in self.bigram_log_params:
            if w1 not in self.bigram_counts:
                n_w1_w2 = 0
                n_w1_o = 0
            elif w2 not in self.bigram_counts[w1]:
                n_w1_w2 = 0
                n_w1_o = self.bigram_totals[w1]
            else:
                n_w1_w2 = self.bigram_counts[w1][w2]
                n_w1_o = self.bigram_totals[w1]

            freq = (n_w1_w2 + self.beta * self.ugram_model.get_token_probability(w2)) / float(n_w1_o + self.beta)
            self.bigram_log_params[(w1, w2)] = -math.log(freq)

        return self.bigram_log_params[bigram]

    def get_document_log_probability(self, document):
        """
        Returns the negative log probability of a document.
        This is equal to the negative log of the product of all
        bigram probabilities in the document
        :param document: A given document as a list of tokens
        :return: Negative log probability of the document.
        """
        neg_log_prob = 0
        for i in range(1, len(document)):
            w1, w2 = document[i-1], document[i]
            neg_log_prob += self.get_bigram_log_probability((w1, w2))
        return neg_log_prob
