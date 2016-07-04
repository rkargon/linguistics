#!/usr/bin/python

from math import log

from numpy.random.mtrand import seed, choice, randint


class TopicModel:
    def __init__(self, corpus, n_topics, alpha=0.5, n_iterations=10):
        self.n_topics = n_topics

        # convert tokens to hashes, and store them in the model
        self.corpus = []
        self.vocab = set()
        self.hashed_tokens = dict()
        for d in corpus:
            d_hashed = []
            for t in d:
                h = hash(t)
                self.hashed_tokens[h] = t
                self.vocab.add(h)
                d_hashed.append(h)
            self.corpus.append(d_hashed)

        # the parameters of the model
        self.p_t_d = []
        self.p_w_t = {}
        self.n_d_t = None
        self.n_t_w = None
        self.gibbs_sampling(alpha, n_iterations)

    def gibbs_sampling(self, alpha, n_iterations):
        n_topics = self.n_topics

        # initialize all counts to 0
        n_d_t = [[0 for _ in range(n_topics)] for _ in range(len(self.corpus))]
        n_t_w = {w: [0 for _ in range(n_topics)] for w in self.vocab}

        # store intermediate values for estimating parameters
        doc_sums = [float(len(d) + n_topics * alpha) for d in self.corpus]
        topic_sums = [len(self.vocab) * alpha for _ in range(n_topics)]

        word_topic_map = [[None for _ in d] for d in self.corpus]

        # randomly assign topics to words, and set up initial counts
        for di in range(len(self.corpus)):
            d = self.corpus[di]
            for wi in range(len(d)):
                w = d[wi]
                t = randint(0, n_topics)
                word_topic_map[di][wi] = t
                n_t_w[w][t] += 1
                n_d_t[di][t] += 1
                topic_sums[t] += 1

        for i in range(n_iterations):
            for di in range(len(self.corpus)):
                pct_done = 100 * (i * len(self.corpus) + di) / float(len(self.corpus) * n_iterations)
                print("Iteration %d/%d, Document %d/%d.... (%2.2f%%)" % \
                      (i, n_iterations, di, len(self.corpus), pct_done))

                d = self.corpus[di]

                for wi in range(len(d)):
                    w = d[wi]
                    # i. remove current word from counts
                    old_topic = word_topic_map[di][wi]
                    n_d_t[di][old_topic] -= 1
                    n_t_w[w][old_topic] -= 1
                    topic_sums[old_topic] -= 1

                    # ii. estimate probabilities using 5.6, 5.7
                    word_topic_probs = []
                    for t1 in range(n_topics):
                        p_t_d = (n_d_t[di][t1] + alpha) / doc_sums[di]
                        p_w_t = (n_t_w[w][t1] + alpha) / topic_sums[t1]
                        word_topic_probs.append(p_t_d * p_w_t)
                    s = float(sum(word_topic_probs))
                    word_topic_probs = [p/s for p in word_topic_probs]
                    # iii. assign w to a topic randomly
                    new_topic = choice(list(range(n_topics)), p=word_topic_probs)
                    word_topic_map[di][wi] = new_topic

                    # iv. increment counts accordingly
                    n_d_t[di][new_topic] += 1
                    n_t_w[w][new_topic] += 1
                    topic_sums[new_topic] += 1

        # finalize parameters
        for di in range(len(self.corpus)):
            self.p_t_d.append([(n_d_t[di][t1] + alpha) / doc_sums[di] for t1 in range(n_topics)])
        for w in self.vocab:
            self.p_w_t[w] = [(n_t_w[w][t1] + alpha) / topic_sums[t1] for t1 in range(n_topics)]

        # also store counts
        self.n_t_w = n_t_w

    def calc_log_probability(self):
        log_prob = 0
        for di in range(len(self.corpus)):
            d = self.corpus[di]
            for w in d:
                lp = log(sum([self.p_t_d[di][t] * self.p_w_t[w][t] for t in range(self.n_topics)]))
                log_prob += lp
        return log_prob

    def best_words_for_topics(self, n, theta):
        def rank_func(w, t):
            return (self.n_t_w[w][t] + theta) / float(sum(self.n_t_w[w]) + self.n_topics * theta)

        ranked_words = []
        for topic in range(self.n_topics):
            ranked_tokens = sorted(self.vocab, key=lambda w: rank_func(w=w, t=topic), reverse=True)[:n]

            ranked_words.append([self.hashed_tokens[tk] for tk in ranked_tokens])

        return ranked_words
