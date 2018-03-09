import pandas as pd
import numpy as np
import glob
import os
from collections import Counter

def filesToArray(label):
    st = "..\DATA\\aclImdb\\train\{}\*.txt".format(label)
    arr = glob.glob(st)
    text = []
    for f in arr:
        with open (f,"r",encoding="utf8") as myfile:
            text.append(myfile.read().replace("\n"," ").replace("<br />"," "))
    return text

postext = filesToArray("pos")
negtext = filesToArray("neg")

superstring = "".join([string for string in postext]).lower()
poscunt = Counter(superstring.split())
superstring = "".join([string for string in negtext]).lower()
negcunt = Counter(superstring.split())

alltext = negtext
for s in postext:
    alltext.append(s)
superstring = "".join([string for string in alltext]).lower()
temp = superstring.split()
allcunt = Counter(temp)

#print('POSITIVE: ')
#print(poscunt.most_common(20))
#print('NEGATIVE: ')
#print(negcunt.most_common(20))
#print('ALL')
#print(allcunt.most_common(20))

n = len(temp)
posprob = dict()
for s in postext:
    if(allcunt.has_key(s)):
        posprob[s] = poscunt[s] + 1 / allcunt[s] + n

negprob = dict()
for s in negtext:
    if(allcunt.has_key(s)):
        negprob[s] = negcunt[s] / allcunt[s]
