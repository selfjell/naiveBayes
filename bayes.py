import numpy as np
import csv

class Bayes():
    def __init__(self, pos_counts, neg_counts, vocab_counts, n_doc=25000, n_pos_doc=12500, n_neg_doc=12500):
        self.positive_freq = pos_counts
        self.negative_freq = neg_counts
        self.vocabulary_freq = vocab_counts

        self.pos_prior = np.log(n_pos_doc/float(n_doc))
        self.neg_prior = np.log(n_neg_doc/float(n_doc))
        #self.pos_prior, self.neg_prior = 0.5, 0.5

        self.pos_likelihood = {}
        self.neg_likelihood = {}

    def train(self):
        pos_vocab_wcount = sum([self.positive_freq[word]+1 for word, count in list(self.vocabulary_freq.most_common()) if self.positive_freq[word] != None])
        neg_vocab_wcount = sum([self.negative_freq[word]+1 for word, count in list(self.vocabulary_freq.most_common()) if self.negative_freq[word] != None])

        for word in list(self.vocabulary_freq):
            #Class: Positive
            wcount = self.positive_freq[word]
            self.pos_likelihood[word] = np.log((wcount+1) / float(pos_vocab_wcount))

            #Class: Negative
            wcount = self.negative_freq[word]
            self.neg_likelihood[word] = np.log((wcount+1) / float(neg_vocab_wcount))

    def test(self, doc):
        #Class: Positive
        sum_positive = self.pos_prior

        for word in doc.split():
            if(self.vocabulary_freq[word] >= 1):
                sum_positive += self.pos_likelihood[word]

        #Class: Negative
        sum_negative = self.neg_prior

        for word in doc.split():
            if(self.vocabulary_freq[word] >= 1):
                sum_negative += self.neg_likelihood[word]

        return sum_positive, sum_negative

    def save(self):
        with open('pos_training.csv', 'w', newline='') as f:
            w = csv.writer(f, delimiter=':')
            w.writerows(pos_likelihood.items())
        with open('neg_training.csv', 'w', newline = '') as f:
            w = csv.writer(f, delimiter = ':')
            w.writerows(neg_likelihood.items())

    def load(self):
        self.pos_likelihood = load_dict('./pos_training.csv')
        self.neg_likelihood = load_dict('./neg_training.csv')


    # Loads the processed dictionary from a csv-file
    def loadDict(self, filePath):
        _input = Counter()
        with open (filePath, 'r', errors="ignore", encoding='utf-8') as f:
            r = csv.reader(f, delimiter=':')
            for row in r:
                _input[row[0]] = float(row[1])
        return _input
