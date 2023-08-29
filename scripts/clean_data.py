import os
import pandas as pd
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_strip(text):
    text = text.strip() #strip
    return text

def clean_lowercase(text):
    text = text.lower()
    return text

def clean_digits(text):
    text = ''.join(char for char in text if not char.isdigit()) ## remove numbers
    return text

def clean_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '') ## remove punctuation
    return text

def clean_stopwords(text):
    tokenized_sentence = word_tokenize(text)
    stop_words = set(stopwords.words('english')) ## define stopwords
    text = [w for w in tokenized_sentence if not w in stop_words] ## remove stopwords
    text = ' '.join(word for word in text)
    return text

def clean_lemmatize(text):
    tokenized_sentence = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatized = [WordNetLemmatizer().lemmatize(word, pos = "v") for word in tokenized_sentence]
    text = ' '.join(word for word in lemmatized)
    return text

def clean_total(text):
    """Performs :
    -Strip
    -Lowercase
    -Digits
    -Punctuation
    -Stopwords
    -Lemmatization"""
    text = clean_strip(text)
    text = clean_lowercase(text)
    text = clean_digits(text)
    text = clean_punctuation(text)
    text = clean_stopwords(text)
    text = clean_lemmatize(text)
    return text

def clean_basic(text):
    """Performs:
    -Strip
    -Lowercase
    -Digits
    -Punctuation"""
    text = clean_strip(text)
    text = clean_lowercase(text)
    text = clean_digits(text)
    text = clean_punctuation(text)
    return text

def clean_nolemma(text):
    """Performs :
    -Strip
    -Lowercase
    -Digits
    -Punctuation
    -Stopwords
    -Lemmatization"""
    text = clean_strip(text)
    text = clean_lowercase(text)
    text = clean_digits(text)
    text = clean_punctuation(text)
    text = clean_stopwords(text)
    return text
