from scripts.utils import DataProcess

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, precision_score, f1_score, recall_score, accuracy_score



scoring = {'f1_score' : make_scorer(f1_score, average='macro'),
           'precision_score ': make_scorer(precision_score,average="macro"),
           'recall_score': make_scorer(recall_score,average="macro"),
           'accuracy':make_scorer(accuracy_score)}
X_train, y_train =  1, 1

def model_run(model, X=X_train, y=y_train):
    score=cross_validate(model,X,y,scoring = scoring)
    results_df=pd.DataFrame(score).mean()
    results_df["model"] = str(model)
    return results_df

class PreprocDl():
    def __init__(self):
        '''
        Define as pl
        Class inheriting the cleaning method from DataProcess.
        Adding a BoW, TFIDF function on top of that.
        Goal is to load, clean and preprocess data in one go.
        '''
        dp = DataProcess()
        self.clean_data_short = dp.clean_data_short

    def load_dl(self, agreement=0, max_features=100, max_len=100):
        df = self.clean_data_short(agreement)
        X= df["cleaned_text"]
        X = [text_to_word_sequence(text) for text in X]

        y= df["sdg"]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=42)

        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(X_train)+list(X_test))
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        X_train = pad_sequences(X_train, maxlen=max_len)
        X_test = pad_sequences(X_test, maxlen=max_len)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train.values)
        y_test = le.transform(y_test.values)

        return X_train, X_test, y_train, y_test
