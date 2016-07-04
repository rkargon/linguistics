#!/usr/bin/python

from math import log

from .token import STOP


class IBMModel1:
    def __init__(self, f_corpus, e_corpus):

        # map hash of every word to word itself, then use hashes themselves for internal calculations
        self.hash_to_token_map = {hash(t): t for sentence in e_corpus + f_corpus for t in sentence}

        e_corpus_hashed = [list(map(hash, sentence)) for sentence in e_corpus]
        f_corpus_hashed = [list(map(hash, sentence)) for sentence in f_corpus]

        # generate vocabularies, by adding every word in corpora to sets
        self.e_vocab = set().union(*e_corpus_hashed)
        self.f_vocab = set().union(*f_corpus_hashed)

        # The set of parameters that specify the likelihood of a foreign word translating to an english word
        self.tau_fe = {}

        # The set of parameters that specify the likelihood of,
        # in an English sentence of length L and a French sentence of length M,
        # that a French word aligns to English word k given that the previous French word aligned with English word j
        self.delta_lmjk = {}

        self.most_likely_translations = {}

        self.estimate_parameters(e_corpus=e_corpus_hashed, f_corpus=f_corpus_hashed)

    def estimate_parameters(self, e_corpus, f_corpus):
        """
        Use Expectation-Maximization to estimate the translation parameters given a foreign and local corpus
        """

        # initialize tau parameters
        initial_tau_value = 1.0
        n_ef = {}

        for i in range(1, 11):
            # E step:

            # initialize n_ef
            for f_dict in list(n_ef.values()):
                for f in list(f_dict.keys()):
                    f_dict[f] = 0

            # for each pair of sentences in corpora
            for E, F in zip(e_corpus, f_corpus):
                # for each French word in sentence
                for fk in F:
                    if fk not in self.tau_fe:
                        self.tau_fe[fk] = {e: initial_tau_value for e in E}
                    tau_fe_fk = self.tau_fe[fk]
                    pk = 0
                    for ej in E:
                        if ej not in self.tau_fe[fk]:
                            tau_fe_fk[ej] = initial_tau_value
                        pk += tau_fe_fk[ej]

                    for ej in E:
                        if ej not in n_ef:
                            n_ef[ej] = {f: 0 for f in F}
                        n_ef[ej][fk] = n_ef[ej].get(fk, 0) + self.tau_fe[fk][ej] / float(pk)

            # M step:
            for e in self.e_vocab:
                n_ef_e = n_ef[e]
                n_eo = sum(n_ef_e.values())
                for f in list(n_ef_e.keys()):
                    self.tau_fe[f][e] = n_ef_e[f] / float(n_eo)

        # map each foreign word to the 'english' word with the highest translation probability
        self.most_likely_translations = {f: max(self.tau_fe[f], key=self.tau_fe[f].get) for f in self.tau_fe}

    def estimate_alignment_params(self, e_corpus, f_corpus):
        """
        Uses EM to train a hidden markov model that estimates alignment parameters for Foreign to English words.
        It assumes that the alignment of each word is a hidden markov variable that depends on the alignment of the
        previous word.

        In this case, tau values from the previous estimate step are used, so only alignment probabilities are
        calculated.

        :param e_corpus: The English corpus, as a list of sentence, each sentence a list of hashes of tokens
        :param f_corpus: The Foreign corpus, as a list of sentence, each sentence a list of hashes of tokens
        """

        initial_sigma_estimate = 1.0
        n_lmyy = {}
        # n_lmyx = {}

        for i in range(1, 11):
            # set all expected counts to 0 (n_yy and n_yx)
            for ldict in list(n_lmyy.values()):
                for mdict in list(ldict.values()):
                    for ydict in list(mdict.values()):
                        for k in list(ydict.keys()):
                            ydict[k] = 0

            # E step
            for E, F in zip(e_corpus, f_corpus):
                len_e = len(E)
                len_f = len(F)

                if len_e not in n_lmyy:
                    n_lmyy[len_e] = {}
                if len_f not in n_lmyy[len_e]:
                    n_lmyy[len_e][len_f] = {}
                n_ef_yy = n_lmyy[len_e][len_f]

                # use sigam, tau estimates to calculate alpha, beta probabilities
                alpha = [{}] * len(E)
                beta = [{}] * len(E)

                

                # calculate n_i,y,y and n_y,y,x using alpha, beta
                    # - accumulate into n_yy, n_yx
                pass
            # M step
                # re-compute sigma estimates using expected count
            pass

        pass

    def get_translations(self, token):
        """
        Returns translations of a token as a set of tuples (token, probability)
        :param token: the foreign token in question
        :returns: a list [(token, probability), ...]
        """

        h = hash(token)
        if h in self.tau_fe:
            e_dict = self.tau_fe[h]
            return [(self.hash_to_token_map[eh], e_dict[eh]) for eh in list(e_dict.keys())]
        else:
            return []

    def get_most_likely_translation(self, token):
        """
        Returns the highest-probability translation for the given token
        """
        h = hash(token)
        if h in self.most_likely_translations:
            best_hash = self.most_likely_translations[h]
            return self.hash_to_token_map[best_hash]
        else:
            return token


def dumb_decoder(model, sentence):
    """
    A simple decoder that maps each foreign word to its most likely translation.
    :param model: An IBM model that provides translation probabilities for foreign to english words
    :param sentence: A list of tokens in the foreign language
    :return: A list of tokens in the local (english) language
    """
    return list(map(model.get_most_likely_translation, sentence))


def noise_channel_decoder(translation_model, bigram_model, sentence):
    """
    A decoder using a noise channel model, that takes into account translation probabilities as well as
    a bigram model to ensure high-probability english sentences are favored.
    :param translation_model: An IBM model that provides translation probabilities for foreign to english words
    :param bigram_model: A bigram model that provides probabilities for pairs of English words
    :param sentence: A list of foreign-language tokens
    :return: A list of tokens in the local (english) language
    """

    def translation_log_prob(translation, prev_e_token):
        """
        Find the log probability of a single translation
        :param translation: A tuple (english word, translation probability) for a certain foreign word
        :param prev_e_token: The previous English token
        :return: The log probability that a foreign word translates to an english one
        """
        token, trans_prob = translation
        return log(trans_prob) - bigram_model.get_bigram_log_probability((prev_e_token, token))

    english_sentence = []
    previous_e_token = STOP
    for i in range(len(sentence)):
        curr_f_token = sentence[i]
        # a list of (token, translation_prob)
        translations = translation_model.get_translations(curr_f_token)
        # return French if no translations exist
        if len(translations) == 0:
            english_translation = curr_f_token
        else:
            english_translation = max(translations, key=lambda t: translation_log_prob(t, previous_e_token))[0]
        english_sentence.append(english_translation)
        previous_e_token = english_translation

    return english_sentence
