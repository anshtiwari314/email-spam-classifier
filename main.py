import numpy as np
import pandas as pd
import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


tfidfVec = pickle.load(open('./tfidf.pkl','rb'))
multinomial = pickle.load(open('./Multinomial.pkl','rb'))

def textPreprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        ps = PorterStemmer()
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Email/SMS Spam Classifier")
msg =None
msg = st.text_input("Enter the message")

tempVec = tfidfVec.transform( [textPreprocessing(msg)] ).toarray()

result =multinomial.predict(tempVec)

if(result==0 and msg !=''):
    st.text("msg is ham")
elif(result==1 and msg !=''):
    st.text("msg is spam")
