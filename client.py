import requests
import pandas as pd
import json
from sklearn.metrics import accuracy_score

urls = ["http://localhost:8001/", "http://localhost:8002/", "http://localhost:8003/"]

df = pd.read_csv("diabetes.csv")

while True:

    job = input("Enter the task you wanted to do: ")
    
    
    if job == "train":
        myobj = {'task':'train', 'data':0}
        for url in urls:
            res = requests.post(url,json=myobj)
            print(res.text)

    elif job == "predict":
        x = int(input("Enter the data range to predict: "))
        myobj = {'task':'predict','data': x}

        test = df.iloc[x:,:]
        test_X = test.values[:,:-1]
        test_y = test.values[:,-1]

        predictions = []
        for url in urls:
            res = requests.post(url,json=myobj)
            predictions.append(res.json()["values"])

        res1, res2, res3 = predictions

        print("SVM")
        print(accuracy_score(test_y, res1))
        print()

        print("GaussianNB")
        print(accuracy_score(test_y, res2))
        print()

        print("neural")
        print(accuracy_score(test_y, res3))
        print()

        final = []
        for nums in zip(res1,res2,res3):
            if nums.count(1) > nums.count(0):
                final.append(1)
            elif nums.count(1) < nums.count(0):
                final.append(0)

        print("ensemble")
        print(accuracy_score(test_y, final))
        print()
