from importlib import import_module

from sklearn import svm
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, flash
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
from PIL import Image
from flask.templating import render_template
from werkzeug.utils import secure_filename

import io
import requests
import re
import os
import numpy as np
import pandas as pd 
import nltk
# nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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

app = Flask(__name__)
UPLOAD_FOLDER = './static/assets/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','JPG'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ocr_text(filepath): 
    img = Image.open(filepath)
    text = tess.image_to_string(img)
    return text

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


def classification(kalimat):
    df_hs = pd.read_csv('hasilprepro.csv')    
    # Count Vectorizer \ mengurai kata 
    tfidf_vectorizer = TfidfVectorizer()
    encoder = LabelEncoder()
    X = df_hs['Clean_Kalimat']
    count_vectorizer = CountVectorizer()
    count_vector = count_vectorizer.fit_transform(X)
    tfid_vector = tfidf_vectorizer.fit_transform(X)
    bagofwordsTrain = pd.DataFrame(count_vector.toarray())
    bagofwordsTrain.columns = count_vectorizer.get_feature_names()
    tweet_label = encoder.fit_transform(df_hs['Label'])
    X_train, X_test, y_train, y_test = train_test_split(tfid_vector, tweet_label , shuffle = True, test_size=0.1, random_state=11)
    from sklearn.svm import SVC
    svmc = SVC(kernel='rbf',random_state=0, gamma=.03, C=10)

    # Training SVM
    svmc.fit(X_train, y_train)
    new_statement_features = tfidf_vectorizer.transform([kalimat]).toarray()

    # proses prediksi atau klasifikasi
    predict_sentiment = encoder.inverse_transform(svmc.predict(new_statement_features))
    res = ''
    if(predict_sentiment == ['Non_HS']):
        res =  'Hasil analisis Komentar Positif'
    elif(predict_sentiment == ['HS']):
        res = 'Hasil analisis Komentar Negatif'
    # print(res)
    return (res)

def svm_rbf():
    df_hs = pd.read_csv('hasilprepro.csv')    
    # Count Vectorizer \ mengurai kata 
    tfidf_vectorizer = TfidfVectorizer()
    encoder = LabelEncoder()
    X = df_hs['Clean_Kalimat']
    count_vectorizer = CountVectorizer()
    count_vector = count_vectorizer.fit_transform(X)
    tfid_vector = tfidf_vectorizer.fit_transform(X)
    bagofwordsTrain = pd.DataFrame(count_vector.toarray())
    bagofwordsTrain.columns = count_vectorizer.get_feature_names()
    tweet_label = encoder.fit_transform(df_hs['Label'])
    X_train, X_test, y_train, y_test = train_test_split(tfid_vector, tweet_label , shuffle = True, test_size=0.3, random_state=11)
    from sklearn.svm import SVC
    svmc = SVC(kernel='rbf',random_state=0, gamma=.03, C=10)
    svmc.fit(X_train, y_train)
    y_pred_svm = svmc.predict(X_test)
    print("Report : \n", classification_report(y_test, y_pred_svm), "\n Accuracy RBF : ",accuracy_score(y_test,y_pred_svm))
    # print("Accuracy RBF : ",accuracy_score(y_test,y_pred_svm))
    return ("Report : \n", classification_report(y_test, y_pred_svm), "\n Accuracy RBF : ",accuracy_score(y_test,y_pred_svm))
#ubah
def svm_poly():
    df_hs = pd.read_csv('hasilprepro.csv')    
    # Count Vectorizer \ mengurai kata 
    tfidf_vectorizer = TfidfVectorizer()
    encoder = LabelEncoder()
    X = df_hs['Clean_Kalimat']
    count_vectorizer = CountVectorizer()
    count_vector = count_vectorizer.fit_transform(X)
    tfid_vector = tfidf_vectorizer.fit_transform(X)
    bagofwordsTrain = pd.DataFrame(count_vector.toarray())
    bagofwordsTrain.columns = count_vectorizer.get_feature_names()
    tweet_label = encoder.fit_transform(df_hs['Label'])
    X_train, X_test, y_train, y_test = train_test_split(tfid_vector, tweet_label , shuffle = True, test_size=0.3, random_state=11)
    from sklearn import svm
    clf = svm.SVC(kernel='poly', random_state=0, degree=3, C=10)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    print("Report : \n", classification_report(y_test, predict), "\n Accuracy Polynomial : ",accuracy_score(y_test, predict))
    # print("Accuracy Polynomial : ",accuracy_score(y_test, predict))
    return("Report : \n", classification_report(y_test, predict), "\n Accuracy Polynomial : ",accuracy_score(y_test, predict))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')

@app.route('/ocr')
def ocr():
    return render_template('ocr.html')

@app.route('/preprocessing')
def preprocess():
    return render_template('preprocessing.html')

@app.route('/pengujian')
def pengujian():
    return render_template('pengujian.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            if f.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                basepath = os.path.dirname(__file__)
                file_path = os.path.join(
                    basepath, 'static/uploads', secure_filename(f.filename))
                f.save(file_path)
                ocr = ocr_text(file_path)
                return jsonify(res=ocr)
        else:
            flash('No selected file')
            return redirect(request.url)

@app.route('/komentar', methods=['GET', 'POST'])
def komentar():
    token = tokenize(request.form['komentar'])
    case = case_folding(token)
    stop = stopwords(case)
    stem = stemming(str(stop))
    return jsonify(token=token,case=case,stop=stop,stem=stem)

@app.route('/klasifikasi_svmc', methods=['GET', 'POST'])
def klasifikasi_svmc():
    klas = classification(request.form['klas'])
    return jsonify(klas=klas)

@app.route('/pengujian_svmc', methods=['GET', 'POST'])
def pengujian_svmc():
    rbf = svm_rbf()
    poly = svm_poly()
    return jsonify(rbf=rbf,poly=poly)

if __name__ == '__main__':
    app.run(threaded=True, debug=True)
