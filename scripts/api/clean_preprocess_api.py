from scripts.utils import DataProcess
from scripts.clean_data import clean_vec, clean_lemma_vec
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import pandas as pd



def preprocess_features(X : str) -> pd.DataFrame:
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
    X = clean_lemma_vec(cleaned_text)
    # X["lenght_text_cleaned"] = X["lemma"].map(lambda row: len(row.split()))
    X_prepro =  X

    return [str(X_prepro)]
