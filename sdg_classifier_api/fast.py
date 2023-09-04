import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from taxifare.interface.main import pred
from scripts.api.clean_preprocess_api import preprocess_features
from scripts.api.registery import load_model


app = FastAPI()
app.state.model = load_model()


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
    return {'The text is talking about SDG:': float(app.state.model.predict(X_prepro))}
