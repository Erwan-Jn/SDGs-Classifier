import glob
import os
import time
import pickle
from colorama import Fore, Style
from scripts.api.gcs_models import BucketManager
from google.cloud import storage
from scripts.params import *
import pickle


def load_model(CONFIG):
    print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

    # try:
    manager = BucketManager(CONFIG['project_id'], CONFIG['bucket_name'])
    manager.download_file(CONFIG['remote_file_path'], CONFIG['local_file_path'])
    print("✅ Latest model downloaded from cloud storage")
    with open(CONFIG['local_file_path'], 'rb') as file:
        model = pickle.load(file)

    return model


    # except:
    #     print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

    #     return None
