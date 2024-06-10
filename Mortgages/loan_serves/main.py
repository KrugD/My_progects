import json
import pandas as pd
import dill

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('model/price_cars.pkl', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str

class Prediction(BaseModel):
    id: str
    Result: float

@app.get('/status')
def status():
    return 'I am OK'

@app.get('/version')
def version():
    return model['metadata']

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'id': form.id,
        'Result': y[0]
    }
