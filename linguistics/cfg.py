#!/usr/bin/python
import re
from .token import Token, UNK


class RewriteRule:
    def __init__(self, head, children, probability):
        self.head = head
        self.children = children
        self.probability = probability

    def __repr__(self):
        return "<RewriteRule: %s --> %s (%f)>" % (self.head, self.children, self.probability)

    def __str__(self):
        return "%s --> %s (%f)" % (self.head, self.children, self.probability)


class Constituent:
    def __init__(self, label, left_child, right_child, probability):
        self.label = label
        self.left_child = left_child
        self.right_child = right_child
        self.probability = probability

    def __repr__(self):
        l = self.left_child.label if self.left_child is not None else "NULL"
        r = self.right_child.label if self.right_child is not None else "NULL"

        return "<Constituent: %s --> %s %s (%f)>" % (self.label, l, r, self.probability)


class SyntaxTree:
    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        if self.left is None and self.right is None:
            return str(self.value)
        return "(%s %s %s)" % (self.value, self.left or "", self.right or "")


class ProbabilisticContextFreeGrammar:
    def __init__(self, rules, root_label=Token("ROOT")):
        self.token_hashes = {}
        self.rules_hash = {}

        # convert tokens to hashes
        for r in rules:
            # add rule's constituents to hash, create new rule using hashed values
            head_h = hash(r.head)
            self.token_hashes[head_h] = r.head
            children_h = []
            for c in r.children:
                ch = hash(c)
                children_h.append(ch)
                self.token_hashes[ch] = c

            # add rule to rules_hash
            rh = RewriteRule(head_h, children_h, r.probability)
            if rh.children[0] not in self.rules_hash:
                self.rules_hash[rh.children[0]] = [rh]
            else:
                self.rules_hash[rh.children[0]].append(rh)

        self.root_label = hash(root_label)
        self.token_hashes[self.root_label] = root_label

    def parse_sentence(self, sentence):
        sentence_hashed = []
        for t in sentence:
            if hash(t) not in self.token_hashes:
                sentence_hashed.append(hash(UNK))
            else:
                sentence_hashed.append(hash(t))

        L = len(sentence_hashed)
        # allocate L x L chart to store constituents
        chart = [[{} for _ in range(L+1)] for _ in range(L+1)]

        # fill chart cells
        for l in range(1, L+1):
            for s in range(L - l + 1):
                self.fill_cell(chart, sentence_hashed, s, s+l)

        return self.get_syntax_tree(chart[0][L].get(self.root_label, None))

    def fill_cell(self, chart, sentence, i, k):
        if k == i+1:
            c = Constituent(sentence[i], None, None, 1)
            self.add_constituent(chart[i][k], c)
            return

        cell = chart[i][k]

        for j in range(i+1, k):
            for c_left in list(chart[i][j].values()):
                for r in self.rules_hash.get(c_left.label, []):
                    # skip unary rules for this step
                    if len(r.children) != 2:
                        continue

                    c_right = chart[j][k].get(r.children[1], None)
                    if c_right is None:
                        continue

                    prob = r.probability * c_left.probability * c_right.probability
                    old_constituent = cell.get(r.head, None)
                    if old_constituent is None or old_constituent.probability < prob:
                            c = Constituent(r.head, c_left, c_right, prob)
                            self.add_constituent(cell=cell, const=c)

    def add_constituent(self, cell, const):
        cell[const.label] = const

        # if there are no rules that produce the given constituent's label
        if const.label not in self.rules_hash:
            return

        # for each unary rule that produces `const`
        for r in self.rules_hash[const.label]:
            if len(r.children) > 1:
                continue
            new_prob = r.probability * const.probability
            c1 = cell.get(r.head, None)
            if c1 is None or c1.probability < new_prob:
                new_constituent = Constituent(r.head, const, None, new_prob)
                # print "adding constituent", new_constituent
                self.add_constituent(cell, new_constituent)

    def get_syntax_tree(self, c):
        if c is None:
            return None

        value = self.token_hashes[c.label]
        left = self.get_syntax_tree(c.left_child)
        right = self.get_syntax_tree(c.right_child)
        return SyntaxTree(value, left=left, right=right)

    @staticmethod
    def print_chart(chart):
        for r in chart:
            print(r)


def load_rules(f):
    """
    Loads a set of binarized rules for a PCFG from a text file, where each line is a rule
     of the form: "<count> HEAD --> CHILD1 [CHILD2]"
    :param f: The file from which to read
    :return: A list of RewriteRules
    """
    rule_regex = "^(\d+) (\S+) --> (\S+)(?: (\S+))?$"
    # dictionary that stores rules, by mapping each label to a map (CHILD1, CHILD2) ->
    rules_dict = {}

    for l in f:
        m = re.match(rule_regex, l)
        if m is not None:
            count_str, label_str, C1_str, C2_str = m.groups()

        else:
            continue

        count = int(count_str)
        label = UNK if label_str == "*UNK*" else Token(label_str)
        C1 = UNK if C1_str == "*UNK*" else Token(C1_str)
        if C2_str is None:
            C2 = None
        else:
            C2 = UNK if C2_str == "*UNK" else Token(C2_str)

        if label not in rules_dict:
            rules_dict[label] = set()

        # skip rare rules
        if count == 1:
            continue

        rule_tuple = (count, C1, C2)
        rules_dict[label].add(rule_tuple)

    # final list of rules
    rules = []

    # convert counts for each rule into probabilities
    for label in rules_dict:
        total_count = sum([r[0] for r in rules_dict[label]])
        for rt in rules_dict[label]:
            count, C1, C2 = rt
            if C2 is None:
                children = [C1]
            else:
                children = [C1, C2]
            rule = RewriteRule(label, children, count / float(total_count))
            rules.append(rule)

    return rules
