import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

from colorama import Fore, Style

from scripts.utils import DataProcess, load_processed_data
from scripts.model_ML import train_model, evaluate_model, predict_model, load_model
from scripts.clean_data import clean_vec, clean_lemma_vec
from scripts.params import *


def local_setup()-> None:
    for file_path in LOCAL_PATHS:
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

def preprocess() -> None:
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
    data_clean = dp.clean_data(grouped=True, agreement=0, abs_path=True)

    now = datetime.now()

    exit_path = LOCAL_DATA_PATH
    file_name = f"processed_{now.strftime('%d-%m-%Y-%H-%M')}.csv"
    full_file_path = os.path.join(exit_path, file_name)

    print(full_file_path)
    data_clean.to_csv(full_file_path)

    print("✅ preprocess() done \n Saved localy")

def train(file_name: str = None,
          target = "sdg",
          test_split: float = 0.2) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    data_processed = load_processed_data(file_name=file_name)
    if data_processed is None:
        return None

    y= data_processed[target]
    X = data_processed["lemma"]

    model, res = train_model(X, y)

    model_iteration = len(os.listdir(LOCAL_MODEL_PATH)) + 1
    file_name = f'model_V{model_iteration}.pkl'
    full_file_path = os.path.join(LOCAL_MODEL_PATH, file_name)
    pickle.dump(model, open(full_file_path, 'wb'))

    file_name = f'model_train_V{model_iteration}'
    full_file_path = os.path.join(LOCAL_RESULT_PATH, file_name)
    res.to_csv(full_file_path)

def evaluate(file_name: str = None,
    target = "sdg"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    data_processed = load_processed_data(file_name=file_name)
    if data_processed is None:
        return None

    y= data_processed[target]
    X = data_processed["lemma"]

    model = load_model()
    results = evaluate_model(model, X, y)

    model_iteration = len(os.listdir(LOCAL_EVALUATE_PATH)) + 1
    file_name = f'model_evaluate_V{model_iteration}'

    full_file_path = os.path.join(LOCAL_EVALUATE_PATH, file_name)
    results.to_csv(full_file_path)

    print("✅ evaluate() done \n")

    return results

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model and provided data
    """
    if X_pred is None:
        X_pred = np.array(
            ["The UN debated a new plan to increase poverty-relief efforts in poor and emerging countries",
            "Results of the conference on the protection of biodiversity have stalled, with measures for large mammals especially problematic"
            ]
                )
    print("\n⭐️ Use case: predict")

    model = load_model()
    assert model is not None

    X_pred = clean_vec(X_pred)
    X_pred = clean_lemma_vec(X_pred)
    y_pred = model.predict(X_pred)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    #local_setup()
    #print("✅ Setup done")
    #preprocess()
    #print("✅ Process done")
    #train()
    #print("✅ Train done")
    evaluate()
    print("✅ Evaluate done")
    pred()
    print("✅ Pred done")
