from scripts.utils import DataProcess

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import layers

import tensorflow as tf

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

    def load_dl(self, agreement=0):
        '''
        Method to load the data
        Uses agreement to determine data loaded
        '''
        df = self.clean_data_short(agreement)

        X= df["cleaned_text"].values
        y = df["sdg"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            train_size=0.8,
                                            random_state=42)

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        cat_encode = layers.CategoryEncoding(num_tokens = len(np.unique(y_train)) + 1, output_mode="one_hot")
        y_train_cat = cat_encode(y_train)
        y_test_cat = cat_encode(y_test)

        return X_train, X_test, y_train_cat, y_test_cat

    def init_model(self, output, X_train, max_features=100, max_len=100,
                   embedding_dim=5, loss="categorical_crossentropy",
                   metrics=[tf.keras.metrics.CategoricalAccuracy()]):
        '''
        Method to prepare the model
        Numerous parameters to determine
        '''
        vectorize_layer = layers.TextVectorization(
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=max_len
            )

        vectorize_layer.adapt(X_train)

        text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
        x = vectorize_layer(text_input)

        x = layers.Embedding(max_features + 1, embedding_dim)(x)
        x = layers.Dropout(0.5)(x)

        # Conv1D + global max pooling
        x = layers.Conv1D(128, 7, padding='valid', activation='relu')(x)
        x = layers.GlobalMaxPooling1D()(x)

        # We add a vanilla hidden layer:
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        # We project onto a single unit output layer, and squash it with a sigmoid:
        predictions = layers.Dense(output, activation='softmax', name='pred')(x)
        model = tf.keras.Model(text_input, predictions)

        # Compile the model with binary crossentropy loss and an adam optimizer.
        model.compile(loss=loss,
                    optimizer='adam',
                    metrics=metrics)

        return model
