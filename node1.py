from typing import Union
import json
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.svm import SVC
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

clf = SVC()

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
        print("Container 1 SVC is trained")
        print()
        print()
        return "Container 1 SVC is trained"

    elif item.task == "predict":
        in_data = df.values[item.data:,:-1]
        res = clf.predict(in_data)
        print("Output of SVC classifier ===>  ")
        print(res)
        print()
        print()
        dic = dict()
        dic["values"] = res.tolist()
        encode = jsonable_encoder(dic)
        return JSONResponse(content=encode)