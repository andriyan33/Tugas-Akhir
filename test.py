import io
import requests
import re
import os
import numpy as np
import pandas as pd 
import nltk
# nltk.download('punkt')
import string
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer # to create Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer # tfid Vector 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # confussion matrix
from sklearn.preprocessing import LabelEncoder # to convert classes to number 
from sklearn.model_selection import train_test_split  # for splitting data 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # to calculate accuracy
import pickle

def tokenize(kalimat):
    tokens = nltk.tokenize.word_tokenize(kalimat)
    return tokens

def case_folding(kalimat):
    temp_kalimat = []
    for kl in kalimat:
        kl = kl.lower();
        temp_kalimat.append(kl)
    return temp_kalimat

def translatingText(casefolded):
    translatedText = casefolded.translate(
        str.maketrans('', '', string.punctuation))
    return translatedText

# Stop Word List dari Sastrawi
listStopword = set(stopwords.words('indonesian'))

# Penggunaan Stopword
listStopword =  set(stopwords.words('indonesian'))
def stopwords(kalimat):
 
    removed = []
    for t in kalimat:
        if t not in listStopword:
            removed.append(t)
    return removed

def stemming(translatedText):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    katadasar = stemmer.stem(translatedText)
    return katadasar

df_hs = pd.read_csv('hasilprepro.csv')
# print(df_hs['Kalimat'].head)
# token = tokenize(df_hs['Kalimat'])
# case = case_folding(df_hs['Kalimat'])
# stop = stopwords(case)
# stem = stemming(str(stop))
# df_hs['Clean_Kalimat'] = df_hs.apply(lambda row : stopwords(case), axis = 1)
print(df_hs.head())