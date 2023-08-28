import os
import pandas as pd
import string
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class DataProcess():
    """Class to load and preprocess data. Requires a set up with
    parent directory:
            ---raw_data
            ---file being run"""

    def __init__(self):
        '''Class carries 2 information:
        --The path
        --A sdg dict linking numbers to the SDG's themes'''
        self.path = os.path.join(os.path.dirname(os.getcwd()), "raw_data", "data.csv")
        self.sdg = dict(zip(
            [str(num) for num in np.arange(1, 17)],
            ["Poverty", "Hunger", "Health", "Education", "Gender",
             "Water", "Clean", "Work", "Inno", "Inequalities", "Cities",
             "Cons&Prod", "Climate", "OceanLife", "LandLife", "Peace"]
        )
                        )

    def load_data(self):
        df = pd.read_csv(self.path, sep="\t")
        df["sdg"] = df["sdg"].astype(str)
        df["lenght_text"] = df["text"].map(lambda row: len(row.split()))
        df["nb_reviewers"] = df["labels_negative"] + df["labels_positive"]
        df["sdg_txt"] = df["sdg"].map(self.sdg)
        return df

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
            w for w in tokenized_sentence if not w in stop_words]

        lemmatized = [
            WordNetLemmatizer().lemmatize(word, pos = "v")
            for word in tokenized_sentence_cleaned]

        cleaned_sentence = ' '.join(word for word in lemmatized)

        return cleaned_sentence

    def clean_data(self):
        df = self.load_data()
        df["cleaned_text"] = df["text"].map(cleaning)
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
