import os
import pandas as pd
import string
import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from scripts.clean_data import *

class DataProcess():
    """
    Define as dp
    Class to load and preprocess data. Requires a set up with
    parent directory:
    ---raw_data
    ---file being run
    """

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
        """
        Takes no argument (path given in __init__)
        Returns a pd.DataFrame
        Adds 4 columns to base data
        Converts 2 columns 2 different types
        """
        df = pd.read_csv(self.path, sep="\t")
        df["sdg"] = df["sdg"].astype(str)
        df["lenght_text"] = df["text"].map(lambda row: len(row.split()))
        df["nb_reviewers"] = df["labels_negative"] + df["labels_positive"]
        df["nb_reviewers"] = df["nb_reviewers"].astype(int)
        df["agreement_large"] = df["labels_positive"] / df["nb_reviewers"]
        df["sdg_txt"] = df["sdg"].map(self.sdg)
        return df

    def clean_data_short(self, agreement=0):
        """
        Takes no argument (path given in __init__)
        Returns a pd.DataFrame
        Adds 3 columns of cleaned text + 1 column with length of cleaned text
        """
        translator_p = str.maketrans('', '', string.punctuation)
        translator_d = str.maketrans('', '', string.digits)
        stop_words = set(stopwords.words('english'))

        df = self.load_data()
        df = df.loc[df["agreement"]>=agreement, : ]
        df["cleaned_text"] = clean_vec(df["text"])

        df["cleaned_text"] = df["cleaned_text"].map(lambda row: " ".join([w for w in iter(word_tokenize(row)) if w not in stop_words]))

        return df

    def clean_data(self, agreement=0):
        """
        Takes 1 parameter, agreement. Agreement will keep all values
        above the agreement threshold for the data. The path is given
        in __init__
        Returns a pd.DataFrame
        Adds 3 columns of cleaned text + 1 column with length of cleaned text
        """

        translator_p = str.maketrans('', '', string.punctuation)
        translator_d = str.maketrans('', '', string.digits)
        stop_words = set(stopwords.words('english'))
        word_lem = WordNetLemmatizer()
        word_stem = PorterStemmer()

        df = self.load_data()
        df = df.loc[df["agreement"]>=agreement, : ]

        cleaned_text = clean_vec(df["text"])
        df["lemma"] = clean_lemma_vec(cleaned_text)
        df["stem"] = clean_stem_vec(cleaned_text)
        df["lenght_text_cleaned"] = df["lemma"].map(lambda row: len(row.split()))

        return df
