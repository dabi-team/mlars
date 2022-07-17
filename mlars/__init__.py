from typing import List
import pandas as pd
import numpy as np
from colorama import Fore, Back, Style
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, neighbors,  linear_model, svm
from mlars.getData import clearData, getData
from mlars.models import new_func
import matplotlib.pyplot as plt

models = new_func()


def main(df, name: str, min: int = 103, top3: bool = False):
    df = clearData(df)
    unique = df[name].unique()
    data = getData(df, name, unique)
    dict = {
        "Name": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Average": [],
        "Hyperparameter": []
    }

    for item in models:
        highestModelName = None
        highestModelAccuracy = 0
        highestModelPrecision = 0
        highestModelRecall = 0
        highestModelF1 = 0
        highestModelAverage = 0
        highestModelHyperparameter = None
        for param in item[2][0]:
            model, accuracy, precision, recall, f1, average = try_model(
                item[0], data[0], data[2], item[1], data[1], data[3], param)
            if average > highestModelAverage:
                highestModelName = item[0]
                highestModelAccuracy = accuracy
                highestModelPrecision = precision
                highestModelRecall = recall
                highestModelF1 = f1
                highestModelAverage = average
                highestModelHyperparameter = param
        dict["Name"].append(highestModelName)
        dict["Accuracy"].append(highestModelAccuracy)
        dict["Precision"].append(highestModelPrecision)
        dict["Recall"].append(highestModelRecall)
        dict["F1"].append(highestModelF1)
        dict["Average"].append(highestModelAverage)
        # hyperparameter to string
        highestModelHyperparameter = str(highestModelHyperparameter)
        dict["Hyperparameter"].append(highestModelHyperparameter)
    df = pd.DataFrame(dict)
    return df


def try_model(name, X_train, y_train, Model, X_test, y_test, param):
    try:
        model = getModel(name, X_train, y_train, Model, param)
        y = getPredictions(model, X_test)
        accuracy = getAccuracy(y_test, y)
        precision = getPrecision(y_test, y)
        recall = recall_score(y_test, y, average='macro')
        f1 = f1_score(y_test, y, average="macro")
        average = getAverage(accuracy, precision, recall, f1)
        return model, accuracy, precision, recall, f1, average
    except:
        print(Fore.RED + "model not supported")
        return None, 0, 0, 0, 0, 0


def getModel(name, X_train, y_train, Model, param):
    model = Model(param)
    model.fit(X_train, y_train)
    # print(Fore.WHITE + name + "model fit done with params " + str(param), end=' ')
    return model


def getPredictions(model, X_test):
    predictions = model.predict(X_test)
    # print(Fore.GREEN + "predictions done")
    return predictions


def getAccuracy(y_test, predictions):
    try:
        accuracy = accuracy_score(y_test, predictions)
        # print(Fore.BLUE + str(accuracy))
        return accuracy
    except:
        return 0


def getPrecision(y_test, predictions):
    try:
        precision = precision_score(y_test, predictions, average='macro')
        # print(Fore.BLUE + str(precision))
        return precision
    except:
        print(Fore.RED + "precision not supported")
        return 0


def getRecall(y_test, predictions):
    try:
        recall = recall_score(y_test, predictions,  average='macro')
        # print(Fore.BLUE + str(recall))
        return recall
    except:
        print(Fore.RED + "recall not supported")
        return 0


def getF1(y_test, predictions):
    try:
        f1 = f1_score(y_test, predictions, average='macro')
        # print(Fore.BLUE + str(f1))
        return f1
    except:
        print(Fore.RED + "f1 not supported")
        return 0


def getAverage(accuracy, precision, recall, f1):
    try:
        if accuracy + precision + recall + f1 == 0:
            return 0.2
        average = (accuracy + precision + recall + f1) / 4
        # print(Fore.BLUE + str(average))
        return average
    except:
        print(Fore.RED + "average not supported")
        return 0.2
