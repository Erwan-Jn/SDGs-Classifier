from scripts.utils import DataProcess
from scripts.clean_data import clean_vec, clean_lemma_vec, clean_stem_vec
from scripts.preprocess_data import preprocess_bow, preprocess_tf_idf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import pandas as pd



def preprocess_features(X : str, root_words = 'lemma', vectorizer = 'tf-idf') -> pd.DataFrame:
    """Pipe that takes as input a dataframe with the texts
    Will output a dataframe witht text tf-idf vectorized
    Arguments :
        root_words = by default 'lemma' can be 'stem' as well
        vectorizer = by default 'tf-idf' can 'bow' as well
    Steps are:
    clean_vec:
        -Strip
        -Lowercase
        -Digits
        -Punctuation
    clean_lemma:
    -Stopwords
    -Lemmatization

    -tf-idf
    """
    cleaned_text = clean_vec(X)
    if root_words == 'lemma':
        X = clean_lemma_vec(cleaned_text)
        # X["lenght_text_cleaned"] = X["lemma"].map(lambda row: len(row.split()))
        X_prepro =  X

    elif root_words == 'stem':
        X = clean_stem_vec(cleaned_text)
        # X["lenght_text_cleaned"] = X["stem"].map(lambda row: len(row.split()))
        X_prepro

    return [str(X_prepro)]
