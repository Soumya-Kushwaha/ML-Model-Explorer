import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        dataset = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        dataset = datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        dataset = datasets.load_wine()
    
    
    X = dataset.data
    y = dataset.target
    return X, y