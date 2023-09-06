import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from taxifare.interface.main import pred
from scripts.api.clean_preprocess_api import preprocess_features
from scripts.api.registery import load_model

CONFIG_16sdgs = {
    'project_id': "sdg-classifier-397610",
    'bucket_name': "sdg-classifier",
    'local_file_path': "model/lrm31_08/pipe_lrm.pkl",
    'remote_file_path': "pipe_lrm.pkl"
    }

CONFIG_3cats = {
    'project_id': "sdg-classifier-397610",
    'bucket_name': "sdg-classifier",
    'local_file_path': "model/lrm5_09/pipe_lrm_esg.pkl",
    'remote_file_path': "pipe_lrm_esg.pkl"
    }

app = FastAPI()
app.state.model_16sdgs = load_model(CONFIG=CONFIG_16sdgs)
app.state.model_cat = load_model(CONFIG=CONFIG_3cats)


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def root():
    return {
    'greeting': 'Hello'
    }

@app.get("/predict")
def predict(text):
    X_prepro = preprocess_features(text)
    return {'The text is talking about SDG:': float(app.state.model_16sdgs.predict(X_prepro))}

@app.get("/predict_proba")
def predict_proba(text):
    X_prepro = preprocess_features(text)
    result = app.state.model_16sdgs.predict_proba(X_prepro)
    dict_result = {}
    for sdg in np.arange(0,16,1):
        dict_result[f'Probability of the text talking about SDG {sdg}']= result[0][sdg]
    return dict_result

@app.get("/predict_category")
def predict_cats(text):
    X_prepro = preprocess_features(text)
    result = float(app.state.model_cat.predict(X_prepro))
    return {'This text most probably belongs to the following category:': result}
