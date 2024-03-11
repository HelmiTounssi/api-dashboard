from pyexpat import features
import shap
from pandas import DataFrame
import pandas as pd

# from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import pickle
import shap
import lime
import random
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from fastapi import Depends
from app.api.models import (
    ClientPredictResponse,
    ErrorResponse,
    ClientIDsResponse,
    PredictQueryParams,
    ClientPredictResponse2,
)

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request

# Your function definition

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    model_server_: str = "./saved_models"
    model_file: str = "lgbm_best_model.pickle"
    data_server: str = "./saved_models"
    data_file: str = "x_test.pickle"
    explainer_file: str = "lgbm_explainer.pickle"
    threshold: float = 0.54
    data_train: str = "../data/out/x_train.csv"
    data_test: str = "../data/out/y_train.csv"


app_config = AppConfig()

app = FastAPI()

model_path = f"{app_config.model_server_}/{app_config.model_file}"
explainer_path = f"{app_config.model_server_}/{app_config.explainer_file}"
data_path = f"{app_config.data_server}/{app_config.data_file}"
model_server_ = app_config.model_server_
model_file = app_config.model_file
data_server = app_config.data_server
data_file = app_config.data_file
explainer_file = app_config.explainer_file
default_threshold = app_config.threshold
data_x_train = app_config.data_train
data_y_train = app_config.data_test


def load_pickle(filename):
    with open(filename, "rb") as handle:
        return pickle.load(handle)


Model = imbpipeline or Pipeline or LGBMClassifier
Explainer = shap.TreeExplainer or shap.LinearExplainer

model: Model = load_pickle(model_path)
explainer: Explainer = load_pickle("./saved_models/lgbm_explainer.pickle")
data: pd.DataFrame = pd.DataFrame()
print(data_path)
# Load client data
if data_file.endswith("pickle"):
    # preprocessed data for Proof of Concept
    data: pd.DataFrame = load_pickle(data_path)
else:
    # Raw data files
    # In production, conduct an authenticated, authorised read CSV from AWS S3 bucket
    data: pd.DataFrame = pd.read_csv(data_path)
print(data.shape)
max_records = 1000  # Proof of concept (POC): limité pour accélerer le temps de réponse
if len(data) > max_records:
    data = data.head(max_records)

# data.index should already have been set to SK_ID_CURR
# this is so we do not have to drop the column before making predictions
if "SK_ID_CURR" in data.columns:
    data = data.set_index("SK_ID_CURR")

list_clients = list(data.index)

# Par défaut, le classifier est le model, et les données n'ont pas besoin de preprocess
data_prep = data
clf = model
type_model = type(model)

if isinstance(model, imbpipeline) or isinstance(model, Pipeline):
    # shap n'est pas capable de travailler sur les pipelines
    # il faut extraire le classificateur et preprocess les données (si besoin)
    clf = model.named_steps["clf"]
    type_model = type(clf)
    # on enleve le classificateur pour faire le preprocessing/feature_selection des données
    data_prep = pd.DataFrame(
        model[:-1].transform(data), index=data.index, columns=data.columns
    )


print(f"init_app, clf = {type_model}")
# if explainer is None:
#     if isinstance(clf,LGBMClassifier):
#         explainer=shap.TreeExplainer(clf,data_prep)
#     elif isinstance(clf,LogisticRegression):
#         explainer= shap.LinearExplainer(clf,data_prep)


# Endpoint to return list of client IDs


def clients_ids():
    """
    Return (test) list of clients
    """
    print(f"list_clients count: {len(list_clients)}")
    # Assuming list_clients is a list of client IDs
    return ClientIDsResponse(client_ids=list_clients)


def clients():
    """Return (test) list of clients"""
    # response_data = data.reset_index().to_dict(orient="records")
    json_data = data.reset_index().to_json(orient="records")
    return json_data


# Assuming `get_client_data()` returns a DataFrame record based on client ID
def get_client_data(data: DataFrame, id: int):
    client_data = get_client_data_dataframe(data, id)
    return client_data.to_dict(orient="records")[0]


def get_client_data_dataframe(data: DataFrame, id: int):
    client_data = data[data.index == int(id)]
    if client_data.empty:
        return None
    # Replace NaN and infinity values with None
    client_data.replace(
        {float("nan"): None, float("inf"): None, float("-inf"): None}, inplace=True
    )
    return client_data


# Endpoint to get client data


def get_client(id: int):
    """Renvoie les données d'un client"""
    client_data = get_client_data(data, id)
    if client_data is None:
        raise HTTPException(status_code=404, detail="Client inconnu")
    return client_data


def is_true(ch: str or bool) -> bool:
    if isinstance(ch, bool):
        return ch == True
    if isinstance(ch, str):
        return ch.lower() in ["true", "1", "t", "y"]
    return False


# Endpoint to predict client score

def predict(id: int, return_data: bool, threshold: float):
    """
    Renvoie le score d'un client en réalisant
    le predict à partir du modèle final sauvegardé
    """
    global type_model
    client_data = get_client_data_dataframe(data, id)
    if client_data is None:
        raise HTTPException(status_code=404, detail="Client inconnu")

    y_pred_proba = model.predict_proba(client_data)[:, 1]
    y_pred_proba = y_pred_proba[0]
    y_pred = int((y_pred_proba > threshold) * 1)

    client_data_response = {}
    if return_data:
        client_data_response = client_data.iloc[0].to_dict()
     # Convertir type_model en str
    type_model_str = str(type_model)
    response_body = {
       "id": id,
       "y_pred_proba": y_pred_proba,
       "y_pred": y_pred,
       "model_type": type_model_str,
       "client_data": client_data_response,
    }
    return JSONResponse(content=response_body)



def df_to_json(df):
    return df.to_dict(orient="records")


def explain_all(request: Request):
    """
    Renvoie les explications shap de jusqu'à 1000 clients à partir du modèle final sauvegardé
    Utilisé pour afficher les beeswarm et summary plots
    Example :
    - http://127.0.0.1:8000/explain/nb=100
    http://127.0.0.1:8000/explain?nb=100

    """
    global data, model, explainer
    sample_size: int = int(request.query_params.get("nb", 100))
    max_sample_size = 1000
    nb = min(max_sample_size, sample_size, len(data))
    data_sample: pd.DataFrame = data.sample(n=nb, random_state=42)
    # preprocess
    data_sample_prep = data_sample
    if isinstance(model, imbpipeline) or isinstance(model, Pipeline):
        data_sample_prep = pd.DataFrame(
            model[:-1].transform(data_sample),
            index=data_sample.index,
            columns=data_sample.columns,
        )
    client_data_json = df_to_json(data_sample_prep)
    shap_values = explainer.shap_values(
        data_sample_prep, check_additivity=False
    ).tolist()
    expected_value = explainer.expected_value  # only keep class 1)
    response_body = {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "client_data": client_data_json,
    }
    return JSONResponse(content=response_body)
    

def explain(id: int, return_data: bool, threshold: float):
    """
    Renvoie les explications shap d'un client à partir du modèle final sauvegardé
    Example :
    - http://127.0.0.1:8000/explain/395445?threshold=0.3&return_data=y
    """
    client_data = get_client_data_dataframe(data, id)
    if client_data is None:
        raise HTTPException(status_code=404, detail="Client inconnu")

    y_pred_proba = model.predict_proba(client_data)[:, 1][0]
    y_pred = int((y_pred_proba > threshold) * 1)

    # get first data row (as series)
    feature_names = list(client_data.columns)
    client_data = client_data.head(1)

    # preprocess
    client_data_prep = client_data
    if isinstance(model, imbpipeline) or isinstance(model, Pipeline):
        client_data_prep = DataFrame(
            model[:-1].transform(client_data),
            index=client_data.index,
            columns=client_data.columns,
        )

    explainer_shap_values = explainer.shap_values(
        client_data_prep, check_additivity=False
    )
    shap_values = dict(zip(feature_names, explainer_shap_values[0]))
    expected_value = explainer.expected_value
    client_data_dict = client_data.iloc[0].to_dict() if return_data else {}

    # Return the response as JSON
    response_data = {
        "id": id,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "shap_values": shap_values,
        "expected_value": expected_value,
        "client_data": client_data_dict,
    }
    return JSONResponse(content=response_data)
