import json
from fastapi import FastAPI
from summarizer.api.api import csv_router

app = FastAPI()

app.include_router(csv_router,)

@app.get("/")
async def home():
    return {'Просто домашнаяя страница': 'Привет, Мир!'}

@app.get("/info")
async def get_models():
    """
    This method returns model details to the front end. Based on the service argument
    * :param service: Service can from one of the services such as: sentiment, sentiment analysis etc.
    * :return:
    """
    with open("./application/config.json") as f:
        config = json.load(f)
    return config


