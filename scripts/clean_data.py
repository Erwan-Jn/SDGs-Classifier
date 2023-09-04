import os
import pandas as pd
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import numpy as np

def clean_strip(text):
    text = text.strip() #strip
    return text

def clean_lowercase(text):
    text = text.lower()
    return text

def clean_digits(text):
    text = ''.join(char for char in text if not char.isdigit()) ## remove numbers
    return text

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]**{4}**', '####', x)
        x = re.sub('[0-9]**{3}**', '###', x)
        x = re.sub('[0-9]**{2}**', '##', x)
    return x

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

def clean_stemming(text):
    tokenized_sentence = word_tokenize(text)
    porter = PorterStemmer()
    stem_sentence=[]
    for word in tokenized_sentence:
      stem_sentence.append(porter.stem(word))
    text = ' '.join(word for word in stem_sentence)
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

def clean_total_stem(text):
    """Performs :
    -Strip
    -Lowercase
    -Digits
    -Punctuation
    -Stopwords
    -Stemming"""
    text = clean_strip(text)
    text = clean_lowercase(text)
    text = clean_digits(text)
    text = clean_punctuation(text)
    text = clean_stopwords(text)
    text = clean_stemming(text)
    return text

translator_p = str.maketrans('', '', string.punctuation)
translator_d = str.maketrans('', '', string.digits)
stop_words = set(stopwords.words('english'))
word_lem = WordNetLemmatizer()
word_stem = PorterStemmer()

def clean(text):
    text = text.strip() #strip
    text = text.lower()
    text = text.translate(translator_p)
    text = text.translate(translator_d)
    return text
clean_vec = np.vectorize(clean)

def clean_lemma(text):
    return " ".join([word_lem.lemmatize(w) for w in iter(word_tokenize(text)) if w not in stop_words]) ## remove stopwords
clean_lemma_vec = np.vectorize(clean_lemma)

def clean_stem(text):
    return " ".join([word_stem.stem(w) for w in iter(word_tokenize(text)) if w not in stop_words]) ## remove stopwords
clean_stem_vec = np.vectorize(clean_stem)
