from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

def preprocess_bow(df: pd.DataFrame, max_features = 1000, max_df = 0.7, ngram_range = (1,1)):
    """
    Return a bow in DataFrame format
    Parameters max_features, max_df and ngram_range are mandatory for the CountVectorizer()
    by default max_features = 1000, and max_df = 0.7, and ngram_range = (1,1)
    Returns a df
    """
    texts = df['text']
    count_vectorizer = CountVectorizer(max_features=max_features, max_df=max_df, ngram_range=ngram_range)
    X = count_vectorizer.fit_transform(texts)
    X.toarray()
    vectorized_texts = pd.DataFrame(X.toarray(), columns = count_vectorizer.get_feature_names_out(),
                                    index = texts)
    return vectorized_texts

def preprocess_tf_idf(df: pd.DataFrame, max_features = 1000, max_df = 0.7, ngram_range = (1,1)):
    """
    Return a tf_idf in DataFrame format
    Parameters max_features, max_df and ngram_range are mandatory for the CountVectorizer()
    by default max_features = 1000, and max_df = 0.7, and ngram_range = (1,1)
    Returns a df
    """
    texts = df['text']
    tf_idf_vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df, ngram_range=ngram_range)
    weighted_words = pd.DataFrame(tf_idf_vectorizer.fit_transform(texts).toarray(),
                 columns = tf_idf_vectorizer.get_feature_names_out())
    return weighted_words
