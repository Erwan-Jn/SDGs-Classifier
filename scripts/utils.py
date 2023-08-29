import os
import pandas as pd
import string
import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from scripts.clean_data import *

class DataProcess():
    """Class to load and preprocess data. Requires a set up with
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

    def clean_data(self):
        df = self.load_data()
        df["cleaned_text"] = df["text"].map(clean_nolemma)
        df["lemma"] = df["cleaned_text"].map(clean_lemmatize)
        df["stem"] = df["cleaned_text"].map(clean_stemming)
        df["lenght_text_cleaned"] = df["cleaned_text"].map(lambda row: len(row.split()))
        return df
