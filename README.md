# Ensemble Learning With Docker Containers
***
_Ensemble learning is the process by which multiple models, such as classifiers or experts, are strategically generated and combined to solve a particular computational intelligence problem. Ensemble learning is primarily used to improve the (classification, prediction, function approximation, etc.)_

Each docker container contains single Machine Learning model.

There are 4 components:
|Sl. No | Component Name | Function |
|----|----|----|
|1. |Client | To send and receive data from all the docker containers and returns majority vote for prediction
|2. |Docker container 1| This container contains Support Vector classifier |
|3. |Docker container 2| This container contains Naive Bayes Classifier |
|4. |Docker container 3| This container contains MLP Neural Classifier |

To train the models **Pima Indian Diabetes** dataset was used. To download the dataset [click here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The client send and recieves http messages from each docker using **Fast API**

To run all the containers run the below command, make sure docker service is active.

```
docker compose up
```

To run the Client component run this below command, make sure docker and client component is running in different terminals

```
python client.py
```
Client sends 2 inputs:
1. Task : Train --> To tell the all the docker containers to train the model. 

2. Task : predict and data : 400 --> to tell the docker cntainers to predict the data from row 400 to last row of the dataset.

## Performance of Ensemble model.

| ML Classifier Name | Accuracy |
|----|----|
|SVC|0.7989130434782609|
|Naive Bayes|0.7880434782608695|
|MLP Neural |0.8043478260869565|
|**Ensemble**|**0.8260869565217391**|

As you can see Ensemble learning outperforms other machine learning models in terms of accuracy.









