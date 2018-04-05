from collections import Counter
import numpy as np
from pathlib import PurePath
import csv

class Bayes():
    def __init__(self, vocab_counts = 0, trained = False, n_doc=25000, n_pos_doc=12500, n_neg_doc=12500):
        self.pos_likelihood = {}
        self.neg_likelihood = {}
        if not trained:
            self.vocabulary_freq = vocab_counts

            self.pos_prior = np.log(n_pos_doc/float(n_doc))
            self.neg_prior = np.log(n_neg_doc/float(n_doc))
        else:
            self.load()


    def train(self, positive_freq, negative_freq):
        pos_vocab_wcount = sum([positive_freq[word]+1 for word, count in list(self.vocabulary_freq.most_common()) if positive_freq[word] != None])
        neg_vocab_wcount = sum([negative_freq[word]+1 for word, count in list(self.vocabulary_freq.most_common()) if negative_freq[word] != None])

        for word in list(self.vocabulary_freq):
            #Class: Positive
            wcount = positive_freq[word]
            self.pos_likelihood[word] = np.log((wcount+1) / float(pos_vocab_wcount))

            #Class: Negative
            wcount = negative_freq[word]
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
        with open(PurePath('pos_training.csv'), 'w', newline='', encoding = 'utf-8') as f:
            w = csv.writer(f, delimiter=':')
            w.writerows(self.pos_likelihood.items())
        with open(PurePath('neg_training.csv'), 'w', newline = '', encoding = 'utf-8') as f:
            w = csv.writer(f, delimiter = ':')
            w.writerows(self.neg_likelihood.items())
        with open(PurePath('vocab.csv'), 'w', newline = '', encoding = 'utf-8') as f:
            w = csv.writer(f, delimiter = ':')
            w.writerows(self.vocabulary_freq.items())
        with open(PurePath('priors.txt'), 'w', newline = '', encoding = 'utf-8') as f:
            f.write(str(self.pos_prior) + "\n")
            f.write(str(self.neg_prior) + "\n")
        print("DONE")

    def load(self):
        self.pos_likelihood = self.load_dict(PurePath('.', 'pos_training.csv'))
        self.neg_likelihood = self.load_dict(PurePath('.', 'neg_training.csv'))
        self.vocabulary_freq = self.load_dict(PurePath('.', 'vocab.csv'))
        with open(PurePath('.', 'priors.txt'), 'r', encoding = 'utf-8') as f:
            self.pos_prior = float(f.readline())
            self.neg_prior = float(f.readline())
            print("DONE")

    # Loads a dictionary from a csv-file
    def load_dict(self, filePath):
        _input = Counter()
        with open (filePath, 'r', errors="ignore", encoding='utf-8') as f:
            r = csv.reader(f, delimiter=':')
            for row in r:
                _input[row[0]] = float(row[1])
        return _input
