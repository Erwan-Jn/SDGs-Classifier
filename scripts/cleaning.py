import string

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import numpy as np

import spacy
import spacy_fastlang
from spacy.language import Language
from tqdm import tqdm

translator_p = str.maketrans(string.punctuation, ' '*len(string.punctuation))
translator_d = str.maketrans('', '', string.digits)

nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words('english'))
word_lem = WordNetLemmatizer()

nlp = spacy.load(('en_core_web_sm'))
nlp.add_pipe("language_detector") #fast_lang comes here

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
    text = text.strip()
    text = text.translate(translator_p)
    text = text.translate(translator_d)
    text = text.lower()
    return " ".join(text.split())
clean_vec = np.vectorize(clean)

#def lower_text(text:str) -> str:
#    return text.lower()
#lower_text_vec = np.vectorize(lower_text)

def preprocess_spacy(alpha: list[str]) -> list[str]:
    docs = list()
    alpha = [str(text) for text in alpha]
    for doc in tqdm(nlp.pipe(alpha, batch_size = 128)):
        if doc._.language == "en":
            tokens = list()
            for token in doc:
                if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    tokens.append(token.lemma_)
            docs.append(' '.join(tokens))

    return docs
