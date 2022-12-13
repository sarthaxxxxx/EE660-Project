""""
EE660 Project - Fall 2022
Title: Understanding linguistic patterns for text-based speaker classification
Authors: Sarthak Kumar Maharana, Gopi Tharun Maganti
Date: Dec 6, 2022

Note: Saving the embedding vectors for all combinations is taking a lot of time, so we have put all the results can be found in pdf version of code.
Caution: Running this code would take a considerable amount of time as it uses complex embedding techniques. Use of GPU is recommended.

Run the main.py for classifiers using trivial, supervised learning and transfer learning systems

"""


import sys
import subprocess

## installing sentene-transformer and xgboost models
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'sentence-transformers', 'xgboost'])


## importing all required libraries
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from sentence_transformers import SentenceTransformer, util
import os
import random
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
import gensim
from gensim.models import Word2Vec, KeyedVectors

## importing utility files
import binary_trivial as bt
import binary_sentence_transformer as bst
import multi_class_sentence_transformer as mst
import multi_word2vec as mw
import binary_word2vec as bw

## reading data file
data_sp = pd.read_csv('./data/All-seasons.csv')
data_sp.head()

print('====================================================')
print('Trival Classifiers')
bt.bianry_trivial(data_sp)

print('\n')
print('\n')

print('====================================================')
print('Supervised Learning')
bw.binary_word2vec(data_sp)
mw.multi_word2vec(data_sp)

print('\n')
print('\n')

print('====================================================')
print('Transfer Learning')
bst.binary_sentence_transfoemr(data_sp)
mst.mutli_class_sentence_transfoemr(data_sp)

print('====================================================')

