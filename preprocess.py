from collections import Counter
import numpy as np
import os
from pathlib import Path
from bayes import Bayes

def isEndOfSentence(word):
    if word.endswith(".") or word.endswith("!") or word.endswith("?") or word.endswith(",") or word.endswith("\""):
        return True
    return False

def isNegationWord(word):
    if word.endswith("not") or word.endswith("n't") or word == "no" or word == "never":
        return True
    return False

def getList(path):
    _input = []
    with path.open('r', errors="ignore", encoding='utf-8') as f:
        _input = f.read().splitlines()
    #print(_input)
    return _input



# Cleans a list of String-reviews to lower(), no duplicate words, and negation fix for sentiment analysis
# Argument input_text = A list of String-reviews
# Return input_text = The cleaned up list for sentiment analysis
stop_list = getList(Path('.').joinpath('Etc').joinpath('stopwords.txt'))
neg_stop = getList(Path('.').joinpath('Etc').joinpath('neg_stopwords.txt'))
def clean_text(input_text, stop_words = stop_list, neg_stop = neg_stop):

    # String to lowercase letters and removes the <br /> thing
    input_text = input_text                         \
                    .lower().replace("\n", " ")     \
                    .replace("<br />", " ")         \
                    .replace(")", " ")              \
                    .replace("  ", " ")             \
                    .replace("(", "")               \
                    .replace("{", "")               \
                    .replace("}", "")

    # Convert the the review in the form of a String into a list of words
    words = []

    #Adding extra weight to words preceeded by "very"
    for i in range(len(words)):
        if words[i] == "very":
            if words[i+1] != "very":
                words.append(words[i+1])

    [words.append(x) for x in input_text.split(" ")]


    #Removing stopwords
    words = [word for word in words if not word in stop_words]


    # Ads NOT_ in front of the words following a negation operator: "not", "n't", "no" and "never"
    negation_word = ""
    for j in range(len(words)):
        words[j] = negation_word + words[j]
        if isNegationWord(words[j]):
            negation_word = "NOT_"
        if isEndOfSentence(words[j]):
            words[j] = words[j].replace(words[j][-1], "")
            negation_word = ""


    #Removing negated stopwords
    words = [word for word in words if not word in neg_stop]


    # Joins the list "words" back into String-format and puts it back into its place
    input_text = ' '.join(words)
    input_text = input_text \
        .replace(".", "")   \
        .replace(",", "")   \
        .replace("!","")    \
        .replace("?","")    \
        .replace("\"", "")  \
        .replace("\'", "")  \
        .replace(":","")

    # Returns a list of Strings with all the changes made
    return input_text


def txtToList(path):
    _list = []
    for file in path.glob('*.txt'):
        with file.open("r", errors = "ignore", encoding = "utf-8") as f:
            text = f.read()
            _list.append(text)
    return _list

def save_stats(scores):
    path = Path(".").joinpath("Saves").joinpath("stats.txt")
    path.touch(exist_ok = True)
    with path.open("w", newline = '\n') as f:
        for score in scores:
            f.write(str(score) + "\n")


def load_stats():
    _input = []
    try:
        path = Path(".").joinpath("Saves").joinpath("stats.txt")
        _input = path.open("r", encoding = "utf-8").readlines()
    except OSError as e:
        print("No stats saved, classifier must be trained first")
        print(e.message())
    return _input

def main():
    #Making list of .txt-files (per sentiment)
    print("\tLOADING FILES")

    path = Path('..').joinpath('Data')
    test_ = path.joinpath('test')
    train = path.joinpath('train')

    tp_reviews = txtToList(test_.joinpath('pos'))
    tn_reviews = txtToList(test_.joinpath("neg"))
    pos_reviews = txtToList(train.joinpath("pos"))
    neg_reviews = txtToList(train.joinpath("neg"))
    print("\tFILES LOADED")

    #Cleaning reviews
    reviews = [pos_reviews, neg_reviews, tp_reviews, tn_reviews]
    print("\tCLEANING REVIEWS")
    for list_ in reviews:
        for i, review in enumerate(list_):
            list_[i] = clean_text(review)

    #Joining the reviews into one string (per sentiment)
    pos_string= "".join([string for string in pos_reviews])
    neg_string = "".join([string for string in neg_reviews])

    #Counting the frequency of words (per sentiment and total)
    posCounter = Counter(pos_string.split())
    negCounter = Counter(neg_string.split())
    vocabCounter = Counter(pos_string.split() + neg_string.split())


    for term in list(posCounter):
        if(posCounter[term] == 1):
            del posCounter[term]

    for term in list(negCounter):
        if(negCounter[term] == 1):
            del negCounter[term]


    classifier = Bayes(vocab_counts = vocabCounter)
    classifier.train(posCounter, negCounter)

    testSets = [tp_reviews, tn_reviews]
    n_pos_tp, n_neg_tp = 0, 0
    n_pos_tn, n_neg_tn = 0, 0

    for i, testSet in enumerate(testSets):
        print("_"*15 + "RESULTS" + "_"*15)
        n_pos, n_neg = 0, 0

        for review in testSet:
            pos, neg = classifier.test(review)
            if(pos >= neg):
                n_pos+=1
            else:
                n_neg+=1

        if(i==0):
            print("Positive Testset: ")
            n_pos_tp, n_neg_tp = n_pos, n_neg
        else:
            print("Negative Testset: ")
            n_pos_tn, n_neg_tn = n_pos, n_neg

        print("Positive reviews: {}".format(n_pos))
        print("Negative reviews: {}".format(n_neg))

    pos_prec = n_pos_tp/(n_pos_tp + len(tn_reviews) - n_neg_tn)
    pos_rec = n_pos_tp/len(tp_reviews)
    pos_f1 = 2*((pos_prec*pos_rec)/(pos_prec+pos_rec))

    neg_prec = n_neg_tn/(n_neg_tn + len(tp_reviews) - n_pos_tp)
    neg_rec = n_neg_tn/len(tn_reviews)
    neg_f1 = 2*((neg_prec*neg_rec)/(neg_prec+neg_rec))

    scores = [pos_prec, pos_rec, pos_f1, neg_prec, neg_rec, neg_f1]

    save_stats(scores)
    print_stats(scores)


    return classifier

def print_stats(scores):
    print("_"*35)
    print(" "*10 + "POSITIVE")
    print("Precision:\t {}".format(scores[0]))
    print("Recall:\t\t {}".format(scores[1]))
    print("F1-score:\t {}".format(scores[2]))
    print("_"*35)
    print(" "*10 + "NEGATIVE")
    print("Precision:\t {}".format(scores[3]))
    print("Recall:\t\t {}".format(scores[4]))
    print("F1-score:\t {}".format(scores[5]))
