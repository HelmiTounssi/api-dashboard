#! /usr/bin/env python
from app.api import app
from typing import Union
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request

# from app.api.app import  PredictQueryParams, ClientPredictResponse, ErrorResponse, ClientIDsResponse, ClientData
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import HTMLResponse
from starlette.responses import Response
from app.api.models import (
    ClientPredictResponse,
    ErrorResponse,
    ClientIDsResponse,
    PredictQueryParams,
)
from fastapi import Depends

appfast = FastAPI()


@appfast.get("/")
async def index():
    """List available API routes"""
    routes = ["/clients/", "/client/{id}", "/predict/{id}", "/explain/{id}"]

    html_content = "<html><body>"
    html_content += "<p>Valid routes are:</p><ul>"
    for route in routes:
        html_content += f"<li>{route}</li>"
    html_content += "</ul></body></html>"

    return HTMLResponse(content=html_content)


@appfast.get("/clients/ids/")
async def clients_ids():
    return app.clients_ids()


@appfast.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(content={"error": exc.detail}, status_code=exc.status_code)


@appfast.get("/clients/")
async def clients():
    return app.clients()


@appfast.get("/clients/{id}")
async def client(id: int):
    return app.get_client(id)


@appfast.get("/predict/{id}")
async def predict(
    id: int, return_data: bool = Query(False), threshold: float = Query(0.5)
):
    return app.predict(id, return_data, threshold)


@appfast.get("/explain/all")
async def explain_all(request: Request):
    return app.explain_all(request)

@appfast.get("/explain/{id}")
def explain(
      id: int, return_data: bool = Query(False), threshold: float = Query(0.5)
):
    """
    Renvoie les explications shap d'un client à partir du modèle final sauvegardé
    Example :
    - http://127.0.0.1:8000/explains/395445?threshold=0.3&return_data=y
    """
    return app.explain(id, return_data, threshold)
