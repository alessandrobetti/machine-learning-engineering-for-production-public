import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist

app = FastAPI(title="Predicting Wine Class with batching")

# Open classifier in global scope
# Remark: the classifier is loaded directly into global state instead of within a 
# function that runs when the server is started. This is done because you will be performing unit 
# tests on the classifier without starting the server
with open("models/wine.pkl", "rb") as file:
    clf = pickle.load(file)


class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]


@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}
