import os
import numpy as np

GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "Erwan-Jn", "10-Project", "SGDs-Classifier")


ROOT_PATH  = os.path.join(os.path.expanduser('~'), "sdg_predictor")
LOCAL_DATA_PATH = os.path.join(ROOT_PATH, "data", "processed_data")
LOCAL_MODEL_PATH = os.path.join(ROOT_PATH, "models", "saves")
LOCAL_RESULT_PATH = os.path.join(ROOT_PATH, "models", "results", "train")
LOCAL_EVALUATE_PATH = os.path.join(ROOT_PATH, "models", "results", "evaluate")
