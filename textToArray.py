import pandas as pd
import numpy as np
import glob
from collections import Counter

def filesToArray(label):
    st = "..\DATA\\aclImdb\\train\{}\*.txt".format(label)
    arr = glob.glob(st)
    i = 0
    text = []
    for f in arr:
        with open (f,"r",encoding="utf8") as myfile:
            text.append(myfile.read().replace("\n"," ").replace("<br />"," "))
    return text
postext = filesToArray("pos")
negtext = filesToArray("neg")
#alltext = negtext.extend(postext)

superstring = "".join([string for string in postext]).lower()
poscunt = Counter(superstring.split())
superstring = "".join([string for string in negtext]).lower()
negcunt = Counter(superstring.split())
#superstring = "".join([string for string in alltext]).lower()
#allcunt = Counter(superstring.split())
print('POSITIVE: ')
print(poscunt.most_common(20))
print('NEGATIVE: ')
print(negcunt.most_common(20))
#print('ALL')
#print(allcunt.most_common(20))
