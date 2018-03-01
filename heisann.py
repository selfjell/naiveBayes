from collections import Counter
import numpy as np
import os

def txtToList(path):
    _list = []
    for file in os.listdir(path):
            p = os.path.join(path, file)
            with open(p, "r") as f:
                 text = f.read().replace("\n"," ").replace("<br />"," ")
                 _list.append(text)
    return _list

#Making list of .txt-files (per sentiment)
pos_reviews, neg_reviews = txtToList("./Data/train/pos"), txtToList("./Data/train/neg")

#Joining the reviews into one string (per sentiment)
pos_string= "".join([string for string in pos_reviews]).replace(".", "")
neg_string = "".join([string for string in neg_reviews]).replace(".", "")

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
print(pn_ratios)
