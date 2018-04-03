from collections import Counter
import numpy as np
import os
import csv
from bayes import Bayes

def isEndOfSentence(word):
    if word.endswith(".") or word.endswith("!") or word.endswith("?") or word.endswith(",") or word.endswith("\""):
        return True
    return False

def isNegationWord(word):
    if word == "not" or word.endswith("n't") or word == "no" or word == "never":
        return True
    return False

def isStopWord(word, list):
    if word in list:
        return True
    return False

def getStopWordList(path):
    _input = []
    with open (path, 'r', errors="ignore", encoding='utf-8') as f:
        _input = f.read().splitlines()
        print(_input)
    return _input

# Cleans a list of String-reviews to lower(), no duplicate words, and negation fix for sentiment analysis
# Argument input_text = A list of String-reviews
# Return input_text = The cleaned up list for sentiment analysis
def clean_text(input_text, stopList):
    for i in range(len(input_text)):
        if i == 1:
            print(input_text[i])

        # String to lowercase letters and removes the <br /> thing
        input_text[i] = input_text[i].lower().replace("\n", " ").replace("<br />", " ").replace(")", " ")
        input_text[i] = input_text[i].replace("  ", " ").replace("(", "").replace("{", "")
        input_text[i] = input_text[i].replace("}", "")

        # Convert the the review in the form of a String into a list of words and removes the duplicate words
        words = []
        [words.append(x) for x in input_text[i].split(" ")]

        # Ads NOT_ in front of the words following a negation operator: "not", "n't", "no" and "never"
        negation_word = ""
        for j in range(len(words)):
            words[j] = negation_word + words[j]
            if isNegationWord(words[j]):
                negation_word = "NOT_"
            if isEndOfSentence(words[j]):
                words[j] = words[j].replace(words[j][-1], "")
                negation_word = ""

        for word in words:
            if isStopWord(word,stopList):
                words.remove(word)

        # Joins the list "words" back into String-format and puts it back into its place
        input_text[i] = ' '.join(words)
        input_text[i].replace(".", "").replace(",", "").replace("!","").replace("?","").replace("\"", "").replace("\'", "")
        if i == 1:
            print("--------------------------")
            print(input_text[i])

    # Returns a list of Strings with all the changes made
    return input_text


def txtToList(path):
    _list = []
    for file in os.listdir(path):
            p = os.path.join(path, file)
            with open(p, "r", errors="ignore", encoding='utf-8') as f:
                text = f.read()
                _list.append(text)
    return _list

# Writes the processed words as a dictionary to file
def ratiosToFile(ratios):
    with open ('ratios.csv', 'w', newline='') as f:
        w = csv.writer(f, delimiter=':')
        w.writerows(ratios.items())

# Loads the processed dictionary from a csv-file
def loadTraining(filePath):
    _input = Counter()
    with open (filePath, 'r', errors="ignore", encoding='utf-8') as f:
        r = csv.reader(f, delimiter=':')
        for row in r:
            _input[row[0]] = float(row[1])
    return _input


#Making list of .txt-files (per sentiment)
tp_reviews = txtToList("./Data/test/pos")
tn_reviews = txtToList("./Data/test/neg")
pos_reviews = txtToList("./Data/train/pos")
neg_reviews = txtToList("./Data/train/neg")

#Cleaning reviews
stopList = getStopWordList('./stopwords.txt')
pos_reviews, neg_reviews = clean_text(pos_reviews,stopList), clean_text(neg_reviews,stopList)
tp_reviews, tn_reviews = clean_text(tp_reviews,stopList), clean_text(tn_reviews,stopList)

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
    #if(posCounter[term] >= 45000):
        #del posCounter[term]

for term in list(negCounter):
    if(negCounter[term] == 1):
        del negCounter[term]
    #if(negCounter[term] >= 45000):
        #del negCounter[term]


classifier = Bayes(posCounter, negCounter, vocabCounter)
classifier.train()

testSets = [tp_reviews, tn_reviews]

for i, testSet in enumerate(testSets):
    print()
    n_pos, n_neg = 0, 0
    if(i==0):
        print("Positive Testset: ")
    else:
        print("Negative Testset: ")

    for review in testSet:
        pos, neg = classifier.test(review)
        if(pos > neg):
            n_pos+=1
        else:
            n_neg+=1
    print("Positive reviews: {}".format(n_pos))
    print("Negative reviews: {}".format(n_neg))
    if(i==0):
        print("Correct: {}%".format(n_pos/len(testSet)*100))
    else:
        print("Correct: {}%".format(n_neg/len(testSet)*100))
print()
