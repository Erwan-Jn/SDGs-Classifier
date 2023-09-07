import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

from colorama import Fore, Style

from scripts.utils import DataProcess, load_processed_data, get_top_features, sdg_explainer
from scripts.model_ML import train_model, evaluate_model, predict_model, load_model
from scripts.clean_data import clean_vec, clean_lemma_vec
from scripts.params import *


def main(agreement=0.8, target="sdg"):

    print(Fore.MAGENTA + "\n ⭐️ Do you want to use specific parameter?" + Style.RESET_ALL)

    yes = bool(int(input("Enter 0 for no and 1 for yes: ")))

    if yes:
        agreement = float(input("Enter agreement (float between 0 and 1s): "))
        target = input("Enter target (sdg or esg): ")

    return agreement, target


def local_setup()-> None:
    for file_path in LOCAL_PATHS:
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

def preprocess(agreement:float = 0) -> None:
    """
    Load the raw data from the raw_data folder
    Save the data locally if not in the raw data foler
    Process query data
    Store processed data in the processed directory
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Process data
    dp = DataProcess()
    data_clean = dp.clean_data(grouped=True, agreement=agreement, abs_path=True)

    now = datetime.now()

    file_name = f"processed_{now.strftime('%d-%m-%Y-%H-%M')}.csv"
    full_file_path = os.path.join(LOCAL_DATA_PATH, file_name)

    data_clean.to_csv(full_file_path)

    print("✅ preprocess() done \n Saved localy")

def train(file_name:str = None,
          target:str = "sdg",
          test_split:float = 0.2) -> None:

    """
    Load data from the data folder
    Train the instantiated model on the train set
    Store training results and model weights
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    data_processed = load_processed_data(file_name=file_name)
    if data_processed is None:
        return None

    y= data_processed[target].astype(int)
    X = data_processed["lemma"]

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    model, res = train_model(X, y, test_split)

    model_iteration = len(os.listdir(LOCAL_MODEL_PATH)) + 1
    file_name = f'model_V{model_iteration}.pkl'
    full_file_path = os.path.join(LOCAL_MODEL_PATH, file_name)
    pickle.dump(model, open(full_file_path, 'wb'))

    file_name = f'model_train_V{model_iteration}'
    full_file_path = os.path.join(LOCAL_RESULT_PATH, file_name)
    res.to_csv(full_file_path)

def model_viz()-> None:
    model = load_model()

    df = get_top_features(model['tf_idf'], model['clf'], model['selector'], how = 'long')
    df = df.sort_values(['SDG', 'coef'], ignore_index = True)

    model_iteration = len(os.listdir(LOCAL_MODEL_PATH)) + 1
    file_name = f'coefs_model_V{model_iteration}.csv'

    full_file_path = os.path.join(LOCAL_COEFS_PATH, file_name)
    df.to_csv(full_file_path, index=False)

    fig = sdg_explainer(df=df)

    file_name = f'coefs_model_V{model_iteration}.jpeg'
    full_file_path = os.path.join(LOCAL_IMAGE_PATH, file_name)
    fig.write_image(full_file_path)

def evaluate(file_name:str = None,
    target:str = "sdg"
    ) -> pd.DataFrame:
    """
    Evaluate the performance of the latest production model on processed data
    Return accuracy, recall, precision and f1 as a pd.DataFrame
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

def pred(X_pred:pd.DataFrame = None) -> np.array:
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
    y_pred, y_pred_proba = predict_model(model, X_pred)

    sdg_dict = DataProcess().sdg
    sdg_dict = {int(key): value for key, value in sdg_dict.items()}

    print("\n✅ prediction done: ", y_pred, [sdg_dict[pred] for pred in y_pred], y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    agreement, target = main()
    local_setup()
    print("✅ Setup done")
    preprocess(agreement=agreement)
    print("✅ Process done")
    train(target=target)
    print("✅ Train done")
    model_viz()
    print("✅ Viz created")
    evaluate(target=target)
    print("✅ Evaluate done")
    pred()
    print("✅ Pred done")
