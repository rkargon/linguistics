#!/usr/bin/python
from numpy.random.mtrand import choice, randint, seed


class TopicModel:
    def __init__(self, corpus, n_topics, alpha=0.5, n_iterations=10):
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
        # self.delta_dt = {}
        # self.tau_tw = {}
        self.word_topic_map = {}
        self.gibbs_sampling(n_topics, alpha, n_iterations)

    def gibbs_sampling(self, n_topics, alpha, n_iterations):
        seed(0)

        # randomly assign topics to words
        self.word_topic_map = {w: randint(0, n_topics-1) for w in self.vocab}
        n_dt = [{t: 0 for t in range(n_topics)} for _ in range(len(self.corpus))]
        n_tw = [{w: 0 for w in self.vocab} for _ in range(n_topics)]

        for d_index in range(len(self.corpus)):
            d = self.corpus[d_index]
            for w in d:
                t = self.word_topic_map[w]
                n_dt[d_index][t] += 1
                n_tw[t][w] += 1

        for i in range(n_iterations):
            print("Iteration %d/%d (%f%%)..." % (i, n_iterations, 100 * i / float(n_iterations)))
            for d_index in range(len(self.corpus)):
                print("Document %d/%d (%f%%)..." % (d_index, len(self.corpus), 100 * d_index / float(len(self.corpus))))
                d = self.corpus[d_index]
                for w in d:
                    # i. remove current word from counts
                    old_topic = self.word_topic_map[w]
                    # TODO
                    if n_dt[d_index][old_topic] == 0:
                        print("oops dt", d_index, old_topic)
                    else:
                        n_dt[d_index][old_topic] -= 1
                    n_tw[old_topic][w] -= 1

                    # ii. estimate probabilities using 5.6, 5.7
                    word_topic_probs = []
                    for t in range(n_topics):
                        p_t_d = float(n_dt[d_index][t] + alpha) / (sum(n_dt[d_index].values()) + n_topics * alpha)
                        p_w_t = float(n_tw[t][w] + alpha) / (sum(n_tw[t].values()) + len(self.vocab) * alpha)
                        word_topic_probs.append(p_w_t * p_t_d)

                    # iii. assign w to a topic randomly
                    word_topic_probs = [float(p) / sum(word_topic_probs) for p in word_topic_probs]
                    self.word_topic_map[w] = choice(range(n_topics), p=word_topic_probs)

                    # iv. increment counts accordingly
                    topic = self.word_topic_map[w]
                    n_tw[topic][w] += 1
                    n_dt[d_index][topic] += 1
