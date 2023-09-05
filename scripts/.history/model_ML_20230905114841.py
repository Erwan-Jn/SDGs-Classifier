########################### ML TEMPLATE ##############################
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from colorama import Fore, Style
from typing import Tuple

import numpy as np
import pandas as pd

import os
import pickle

from scripts.params import LOCAL_MODEL_PATH

from sklearn.metrics import make_scorer


def train_model(
        X: np.ndarray,
        y: np.ndarray,
        test_split=0.3,
        max_features=2000
    ):
    """
    Fit the model and return a tuple (fitted_model, history)
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    pipe_model = Pipeline([
        ('tf_idf', TfidfVectorizer(max_features=max_features, ngram_range = (1, 2))),
        ('model', LinearSVC(dual="auto", max_iter=10000))
        ])

    res = cross_validate(pipe_model, X_train, y_train)
    res = pd.DataFrame(res)

    print(f"✅ Model trained on {len(X_train)} rows with mean cross_validated accuracy: {round(np.mean(res.get('test_score')), 2)}")

    pipe_model.fit(X_train, y_train)

    return pipe_model, res

def evaluate_model(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        test_split=0.3
    ):
    """
    Evaluate trained model performance on the dataset
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(Fore.BLUE + f"\nEvaluating model on {len(X_test)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    y_pred = model.predict(X_test)

    precision = make_scorer(precision_score, average="macro")
    f1 = make_scorer(f1_score, average="macro")
    recall = make_scorer(recall_score, average="macro")

    metrics = [accuracy_score, precision, f1, recall]
    breakpoint()
    metrics = [elem(y_test, y_pred) for elem in metrics]
    metrics_name = ["accuracy", "precision", "f1", "recall"]

    print(f"✅ Model evaluated, accuracy: {metrics[0]}")

    return pd.DataFrame(dict(zip(metrics_name, metrics)))

def predict_model(
        model,
        X: str
        ):

    return model.predict(X), model.predict_proba(X)

def load_model(model_name:str = None):

    if model_name==None:
        full_file_path = os.path.join(LOCAL_MODEL_PATH, "None")
    else:
        full_file_path = os.path.join(LOCAL_MODEL_PATH, model_name)

    if not os.path.exists(full_file_path):
        files = [os.path.join(LOCAL_MODEL_PATH, file) for file in os.listdir(LOCAL_MODEL_PATH) if file.endswith(".pkl")]

        if len(files)==0:
            print("No model trained, please train a model")
            return None

        print("No specific model passed, returning latest saved model")
        full_file_path = max(files, key=os.path.getctime)

    model = pickle.load(open(full_file_path, 'rb'))
    return model
