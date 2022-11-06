from typing import Union
import json
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.neural_network import MLPClassifier
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

clf = MLPClassifier(random_state=1, max_iter=300)

class Item(BaseModel):
    task : str
    data : int


app = FastAPI()

@app.post("/")
async def create_item(item: Item):
    df = pd.read_csv("diabetes.csv")
    train_X = df.values[:,:-1]
    train_y = df.values[:,-1]

    if item.task == "train":
        clf.fit(train_X, train_y)
        print("Container 3 MLPClassifier is trained")
        print()
        print()
        return "Container 3 MLPClassifier is trained"

    elif item.task == "predict":
        in_data = df.values[item.data:,:-1]
        res = clf.predict(in_data)
        print("Output of  MLPClassifier ===>  ")
        print(res)
        print()
        print()
        dic = dict()
        dic["values"] = res.tolist()
        encode = jsonable_encoder(dic)
        return JSONResponse(content=encode)