########################### DL TEMPLATE ##############################
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers, models
from tensorflow.keras.models import Sequential

import os
import pandas as pd
import string
import numpy as np

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.model_selection import train_test_split

from scripts.utils import DataProcess

import gensim.downloader as api
word2vec_transfer = api.load("glove-wiki-gigaword-100")

path =os.path.join(os.getcwd(), "raw_data", "data.csv")

translator_p = str.maketrans('', '', string.punctuation)
translator_d = str.maketrans('', '', string.digits)
stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.strip() #strip
    text = text.lower()
    text = text.translate(translator_p)
    text = text.translate(translator_d)
    return text
clean_vec = np.vectorize(clean)

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

dp = DataProcess()
df = dp.clean_data_short()

agreement=0

df = df.loc[df["agreement"]>=agreement, : ]
df["cleaned_text"] = clean_vec(df["text"])
df["cleaned_text"] = df["cleaned_text"].map(lambda row: " ".join([w for w in iter(word_tokenize(row)) if w not in stop_words]))
X= df["cleaned_text"].values

y = df["sdg"].astype(int)
y = y.values - 1

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    train_size=0.8,
                                    random_state=42)

cat_encode = tf.keras.layers.CategoryEncoding(num_tokens = len(np.unique(y_train)), output_mode="one_hot")
y_train = cat_encode(y_train)
y_test = cat_encode(y_test)

max_len=200

X_train = [tf.keras.preprocessing.text.text_to_word_sequence(text) for text in X_train]
X_test = [tf.keras.preprocessing.text.text_to_word_sequence(text) for text in X_test]

X_train_embed = embedding(word2vec_transfer, X_train)
X_test_embed = embedding(word2vec_transfer, X_test)

X_train_pad = tf.keras.utils.pad_sequences(X_train_embed, dtype='float32', padding='post', maxlen=max_len)
X_test_pad = tf.keras.utils.pad_sequences(X_test_embed, dtype='float32', padding='post', maxlen=max_len)
X_train = tf.convert_to_tensor(X_train_pad)
X_test = tf.convert_to_tensor(X_test_pad)

embedding_dims = 200 #Length of the token vectors
filters = 10 #number of filters in your Convnet
kernel_size = 3 # a window size of 3 tokens

model = Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh", return_sequences=False)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh")))
model.add(layers.Dense(130, activation="relu"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(80, activation="relu"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(y_train.shape[1], activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam',
            metrics=[tf.keras.metrics.CategoricalAccuracy(),
                    tf.keras.metrics.Precision()]
            )

epochs = 500

es = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    batch_size = 16,
                    #verbose = 0,
                    epochs=epochs,
                    callbacks=[es]
                    )
