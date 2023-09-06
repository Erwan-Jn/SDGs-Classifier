import os
import pandas as pd
import string

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

import re
import numpy as np

import spacy
from tqdm import tqdm

translator_p = str.maketrans(string.punctuation, ' '*len(string.punctuation))
translator_d = str.maketrans('', '', string.digits)

nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words('english'))
word_lem = WordNetLemmatizer()

nlp = spacy.load('en_core_web_sm', disable = ['ner'])
print('Disabled spaCy components:', nlp.disabled)
print('SpaCy version:', spacy.__version__)


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

def clean(text):
    text = text.strip() #strip
    text = text.translate(translator_p)
    text = text.translate(translator_d)
    text = text.lower()
    return " ".join(text.split())
clean_vec = np.vectorize(clean)

def clean_lemma(text):
    return " ".join([word_lem.lemmatize(w) for w in iter(word_tokenize(text)) if w not in stop_words]) ## remove stopwords
clean_lemma_vec = np.vectorize(clean_lemma)

def preprocess_spacy(alpha: list[str]) -> list[str]:
    """
    Preprocess text input using spaCy.

    Parameters
    ----------
    alpha: List[str]
        a text corpus.

    Returns
    -------
    doc: List[str]
        a cleaned version of the original text corpus.
    """
    docs = list()

    for doc in tqdm(nlp.pipe(alpha, batch_size = 128)):
        tokens = list()
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                tokens.append(token.lemma_)
        docs.append(' '.join(tokens))

    return docs
