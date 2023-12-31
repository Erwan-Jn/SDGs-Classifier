########################### ML TEMPLATE ##############################
#from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2

from imblearn.pipeline import Pipeline

from colorama import Fore, Style

from imblearn.over_sampling import SMOTE

import numpy as np
import pandas as pd

import os
import pickle

from scripts.params import LOCAL_MODEL_PATH

def to_arr(x):
    return x.toarray()

def train_model(
        X: np.ndarray,
        y: np.ndarray,
        test_split: float=0.3,
        max_features: int=100000
    ):
    """
    Fit the model and return a tuple (fitted_model, history)
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)

    pipe_model = Pipeline([
        ('tf_idf', TfidfVectorizer(max_features=max_features, ngram_range = (1, 3), max_df=0.8, norm="l2")),
        ('smote', SMOTE(random_state=42, k_neighbors=10)),
        ('selector', SelectKBest(chi2, k = 2000)),
        ('clf', LogisticRegression(penalty = 'l2', C = .9, multi_class = 'multinomial',
                                   class_weight = 'balanced', random_state = 42,
                                   solver = 'newton-cg', max_iter = 100))
        ])

    print(Fore.BLUE + "\nLaunching CV" + Style.RESET_ALL)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    res = cross_validate(pipe_model, X_train, y_train, verbose=2, cv=cv)
    res = pd.DataFrame(res)

    #breakpoint()
    size_data = y_train.value_counts().sort_values(ascending=False).iloc[0] * 16
    print(f"✅ Model trained on {size_data} total rows including \n {len(X_train)} original rows with mean cross_validated accuracy: {round(np.mean(res.get('test_score')), 2)}")

    pipe_model.fit(X_train, y_train)

    return pipe_model, res

def evaluate_model(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        test_split:float=0.3
    ):
    """
    Evaluate trained model performance on the dataset
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)

    print(Fore.BLUE + f"\nEvaluating model on {len(X_test)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    y_pred = model.predict(X_test)

    metrics = [accuracy_score(y_test, y_pred) , precision_score(y_test, y_pred, average="macro"),
               f1_score(y_test, y_pred, average="macro"), recall_score(y_test, y_pred, average="macro")]
    metrics_name = ["res_accuracy", "res_precision", "res_f1", "res_recall"]

    print(f"✅ Model evaluated, accuracy: {np.round(metrics[0], 2)}, precision: {np.round(metrics[1], 2)}")

    print(f"✅ Full Classification Report")
    print(classification_report(y_test, y_pred, zero_division = 0))

    results = dict(zip(metrics_name, metrics))
    results = {key: [value] for key, value in results.items()}
    return pd.DataFrame(results, index=[0]), y_pred, y_test

def predict_model(model, X:str) -> np.array:
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
