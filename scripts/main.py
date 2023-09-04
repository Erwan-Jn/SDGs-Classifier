import numpy as np
import pandas as pd
from datetime import datetime
import os
import glob
import pickle
import json

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import LinearSVC

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from scripts.utils import DataProcess
from scripts.model_ML import train_model, evaluate_model, predict_model

from taxifare.params import *
from taxifare.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from taxifare.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import load_model, save_model, save_results
from taxifare.ml_logic.registry import mlflow_run, mlflow_transition_model

def preprocess(exit_path=None: str) -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Process data
    dp = DataProcess()
    data_clean = dp.clean_data(grouped=True, agreement=0)

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()

    now = datetime.now()
    file_name = f"processed_data_{now.strftime("%d/%m/%Y-%H:%M")}.csv"

    if file_path is None:
        file_path = os.path.join(os.path.dirname(os.getcwd()), "data", "processed_data")
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    full_file_path = os.path.join(file_path, file_name)
    df.to_csv(full_file_path)



    print("✅ preprocess() done \n Saved localy")

def train(
    file_name = None: str,
    target = "sdg"
    test_split: float = 0.2
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    file_path = os.path.join(os.path.dirname(os.getcwd()), "data", "processed_data")
    full_file_path = os.path.join(file_path, file_name)

    if not os.path.exists(full_file_path):
        file_type = r'\*csv'
        files = glob.glob(folder_path + file_type)
        full_file_path = max(files, key=os.path.getctime)

    data_processed = pd.read_csv(full_file_path)
    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    y= data_processed[target]
    X = data_processed.drop(columns=["sdg", "esg"], axis=1)

    model, res = train_model(X, y)

    file_path = os.path.join(os.parentdir(os.getcwd()), "models", "saves")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    model_iteration = len(os.list_dir(file_path))
    file_name = f'model_V{model_iteration}'
    pickle.dump(model, open(os.path.join(file_path, file_name), 'wb'))

    file_path = os.path.join(os.parentdir(os.getcwd()), "models", "train", "results")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = f'model_V{model_iteration}'
    with open(file_name, "w") as outfile:
        json.dump(res, outfile)

def evaluate(file_name=None:str,
             target = "sdg"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

    file_path = os.path.join(os.path.dirname(os.getcwd()), "data", "processed_data")
    full_file_path = os.path.join(file_path, file_name)

    if not os.path.exists(full_file_path):
        file_type = r'\*csv'
        files = glob.glob(folder_path + file_type)
        full_file_path = max(files, key=os.path.getctime)

    if data_processed.shape[0] > 10:
        print("❌ No data to evaluate on")
        return None
    data_processed = pd.read_csv(full_file_path)

    y= data_processed[target]
    X = data_processed.drop(columns=["sdg", "esg"], axis=1)

    results = evaluate_model(X, y)
    file_path = os.path.join(os.parentdir(os.getcwd()), "models", "evaluate", "results")
    file_name = f'model_V{model_iteration}'
    with open(file_name, "w") as outfile:
        json.dump(res, outfile)





    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
        pickup_longitude=[-73.950655],
        pickup_latitude=[40.783282],
        dropoff_longitude=[-73.984365],
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    preprocess(min_date='2009-01-01', max_date='2015-01-01')
    train(min_date='2009-01-01', max_date='2015-01-01')
    evaluate(min_date='2009-01-01', max_date='2015-01-01')
    pred()
