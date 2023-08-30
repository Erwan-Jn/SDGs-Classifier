from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scripts.utils import DataProcess
import pandas as pd

def preprocess_bow(df: pd.DataFrame, col="lemma", max_features = 1000, max_df = 0.7, ngram_range = (1,1)):
    """
    Return a bow in DataFrame format
    Parameters max_features, max_df and ngram_range are mandatory for the CountVectorizer()
    by default max_features = 1000, and max_df = 0.7, and ngram_range = (1,1)
    Returns a df
    """
    texts = df[col]
    count_vectorizer = CountVectorizer(max_features=max_features, max_df=max_df, ngram_range=ngram_range)
    X = count_vectorizer.fit_transform(texts)
    X.toarray()
    vectorized_texts = pd.DataFrame(X.toarray(), columns = count_vectorizer.get_feature_names_out(),
                                    index = texts)
    return vectorized_texts

def preprocess_tf_idf(df: pd.DataFrame, col="lemma", max_features = 1000, max_df = 0.7, ngram_range = (1,1)):
    """
    Return a tf_idf in DataFrame format
    Parameters max_features, max_df and ngram_range are mandatory for the CountVectorizer()
    by default max_features = 1000, and max_df = 0.7, and ngram_range = (1,1)
    Returns a df
    """
    texts = df[col]
    tf_idf_vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df, ngram_range=ngram_range)
    weighted_words = pd.DataFrame(tf_idf_vectorizer.fit_transform(texts).toarray(),
                 columns = tf_idf_vectorizer.get_feature_names_out())
    return weighted_words

def creating_csv(data = pd.DataFrame, vectorizer = 'tf_idf'):
    """Function to apply bow or tf idf and create csv
    data input must be dataframe
    and the parameter vectorizer is either 'tf-idf' or something else (=bow)"""
    target = data['sdg']
    if vectorizer == 'tf_idf':
        lemmatized = preprocess_tf_idf(data)
        target.reset_index(inplace=True, drop=True)
        lemmatized.reset_index(inplace=True, drop=True)
        result = pd.concat([target, lemmatized], ignore_index=False, axis=1)
        result.to_csv(f"../drafts/{'tf_idf'}.csv", sep=",")
    else :
        bow = preprocess_bow(data)
        target.reset_index(inplace=True, drop=True)
        bow.reset_index(inplace=True, drop=True)
        result = pd.concat([target, bow], ignore_index=False, axis=1)
        result.to_csv(f"../drafts/{'bow'}.csv", sep=",")

class PreprocData():
    def __init__(self):
        '''
        Define as PrepD
        Class inheriting the cleaning method from DataProcess
        Adding a BoW, TFIDF function on top of that
        '''
        dp = DataProcess()
        self.clean_data = dp.clean_data

    def preprocess_bow_full(self):
        df = self.clean_data()
        return preprocess_bow(df)

    def preprocess_tf_idf_full(self):
        df = self.clean_data()
        return preprocess_bow(df)
