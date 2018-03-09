from collections import Counter
import numpy as np
import os
import csv

# Cleans a list of String-reviews to lower(), no duplicate words, and negation fix for sentiment analysis
# Argument input_text = A list of String-reviews
# Return input_text = The cleaned up list for sentiment analysis
# Made by Jakob
def clean_text(input_text):
    for i in range(len(input_text)):

        # String to lowercase letters and removes the <br /> thing
        input_text[i] = input_text[i].lower().replace("\n", " ").replace("<br />", " ").replace("  ", " ")

        # Convert the the review in the form of a String into a list of words and removes the duplicate words
        words = []
        [words.append(x) for x in input_text[i].split(" ")] 

        # Ads NOT_ in front of the words following a negation operator: "not", "n't", "no" and "never"
        negation_word = ""
        for j in range(len(words)):
            words[j] = negation_word + words[j]
            if words[j] == "not" or words[j][-3:] == "n't" or words[j] == "no" or words[j] == "never":
                negation_word = "NOT_"
            if words[j][-1:] == "." or words[j][-1:] == "!" or words[j][-1:] == "?" or words[j][-1:] == "," or words[j][-1:] == "\"":
                words[j] = words[j].replace(words[j][-1], "")
                negation_word = ""

        # Joins the list "words" back into String-format and puts it back into its place
        input_text[i] = ' '.join(words)

    # Returns a list of Strings with all the changes made
    return input_text


def txtToList(path):
    _list = []
    for file in os.listdir(path):
            p = os.path.join(path, file)
            with open(p, "r",encoding="utf8") as f:
                text = f.read()
                _list.append(text)
    return _list

# Writes the processed words as a dictionary to file
def ratiosToFile(ratios):
    with open ('ratios.csv', 'w',encoding="utf8", newline='') as f:
        w = csv.writer(f, delimiter=':')
        w.writerows(ratios.items())

# Loads the processed dictionary from a csv-file
def loadTraining(filePath):
    _input = Counter()
    with open (filePath, 'r', encoding="utf8") as f:
        r = csv.reader(f, delimiter=':')
        for row in r:
            _input[row[0]] = float(row[1])
    return _input


#Making list of .txt-files (per sentiment)
pos_reviews, neg_reviews = txtToList("../DATA/aclImdb/train/pos"), txtToList("../DATA/aclImdb/train/neg")
#Cleaning reviews
pos_reviews, neg_reviews = clean_text(pos_reviews), clean_text(neg_reviews)

#Joining the reviews into one string (per sentiment)
pos_string= "".join([string for string in pos_reviews])
neg_string = "".join([string for string in neg_reviews])

#Counting the frequency of words (per sentiment and total)
posCounter, negCounter = Counter(pos_string.split()), Counter(neg_string.split())
totCounter = Counter(pos_string.split() + neg_string.split())

#Positive negative ratios
pn_ratios = Counter()
for term, count in list(totCounter.most_common()):
    if(count > 100):
        pn_ratio = posCounter[term] / float(negCounter[term]+1)
        if(pn_ratio > 1):
            pn_ratio = np.log(pn_ratio)
        else:
            pn_ratio = -np.log((1 / (pn_ratio + 0.01 )))
        pn_ratios[term] = pn_ratio

#print(pn_ratios.most_common(20))
#print(pn_ratios)
#print(pos_reviews[10])
ratiosToFile(pn_ratios)
newRatios = loadTraining('./ratios.csv')
print(newRatios.items())
