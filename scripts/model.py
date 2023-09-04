from scripts.utils import DataProcess

import numpy as np
import pandas as pd

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, precision_score, f1_score, recall_score, accuracy_score*

import gensim.downloader as api

def embed_sentence_with_TF(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return np.array(embedded_sentence)

def embedding(word2vec, sentences):
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence_with_TF(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed


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
        Goal is to load, clean and preprocess data in one go for DL.
        Loads the DL model as well
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
        y = df["sdg"].astype(int)
        y = y.values - 1

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            train_size=0.8,
                                            random_state=42)

        cat_encode = layers.CategoryEncoding(num_tokens = len(np.unique(y_train)), output_mode="one_hot")
        y_train_cat = cat_encode(y_train)
        y_test_cat = cat_encode(y_test)

        return X_train, X_test, y_train_cat, y_test_cat

    def word2vec_preproc(self, agreement=0, max_len=100):

        word2vec_transfer = api.load("glove-wiki-gigaword-50")

        X_train, X_test, y_train_cat, y_test_cat = self.load_dl(agreement)

        X_train = [tf.keras.preprocessing.text.text_to_word_sequence(text) for text in X_train]
        X_test = [tf.keras.preprocessing.text.text_to_word_sequence(text) for text in X_test]

        X_train_embed = embedding(word2vec_transfer, X_train)
        X_test_embed = embedding(word2vec_transfer, X_test)

        X_train_pad = pad_sequences(X_train_embed, dtype='float32', padding='post', maxlen=max_len)
        X_test_pad = pad_sequences(X_test_embed, dtype='float32', padding='post', maxlen=max_len)

        return X_train_pad, X_test_pad, y_train_cat, y_test_cat


    def init_model(self, input: 'np.array', output: 'int', vocab_size=1000, max_len=150,
                   embedding_dim=100, loss="categorical_crossentropy",
                   metrics=[tf.keras.metrics.CategoricalAccuracy()]):
        '''
        Method to prepare the model
        Numerous parameters to determine
        '''
        vectorize_layer = layers.TextVectorization(
            ngrams = 2,
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=max_len
            )

        vectorize_layer.adapt(input)

        text_input = tf.keras.Input(shape=(1, ), dtype=tf.string, name='text')

        x = vectorize_layer(text_input)
        x = layers.Embedding(vocab_size + 1, embedding_dim)(x)

        x = layers.GRU(50, activation="tanh", return_sequences=True)(x)
        x = layers.GRU(50, activation="tanh", return_sequences=True)(x)
        x = layers.GRU(50, activation="tanh", return_sequences=False)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(50, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        predictions = layers.Dense(output, activation='softmax', name='pred')(x)
        model = tf.keras.Model(text_input, predictions)

        # Compile the model with binary crossentropy loss and an adam optimizer.
        model.compile(loss=loss,
                    optimizer='adam',
                    metrics=metrics)

        return model
