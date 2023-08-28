import os
import pandas as pd

class DataProcess():
    """Class to load and preprocess data. Requires a set up with
    parent directory:
            ---raw_data
            ---file being run"""

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.getcwd()), "raw_data", "data.csv")

    def load_data(self):
        df = pd.read_csv(self.path, sep="\t")
        return df

#---------------------------------------------------------------------

import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def cleaning(sentence):
    punctuation = string.punctuation
    wnl = WordNetLemmatizer()
    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation

    tokenized_sentence = word_tokenize(sentence) ## tokenize
    stop_words = set(stopwords.words('english')) ## define stopwords

    tokenized_sentence_cleaned = [ ## remove stopwords
        w for w in tokenized_sentence if not w in stop_words ]

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in tokenized_sentence_cleaned]

    cleaned_sentence = ' '.join(word for word in lemmatized)

    return cleaned_sentence
