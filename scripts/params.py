import os
import numpy as np

GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "Erwan-Jn", "10-Project", "SGDs-Classifier")
