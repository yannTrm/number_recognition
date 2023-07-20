# -*- coding: utf-8 -*-

# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import os
from tensorflow.keras.models import load_model
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# function
def load_adaboost(path_directory_adaboost):
    adaboost = []
    for network_file in os.listdir(path_directory_adaboost):
        adaboost.append(load_model(path_directory_adaboost + network_file))
    return adaboost

def load_single_model(path_model):
    model = load_model(path_model)
    return model

def predict_proba(model, data):
    return model.predict(data)

def predict(model, data):
    predictions = model.predict(data)
    return np.argmax(predictions, axis=1)

def predict_proba_boosted(adaboost, data):
    predictions = []
    for model in adaboost:
        prediction = model.predict(data, verbose=0)
        predictions.append(prediction)
    final_predictions = np.zeros_like(predictions[0])
    for prediction in predictions:
        final_predictions += prediction
    return final_predictions / len(adaboost)

def predict_boosted(adaboost, data):
    predictions = []
    for model in adaboost:
        prediction = model.predict(data, verbose = 0)
        predictions.append(prediction)
    final_predictions = np.zeros_like(predictions[0])
    for prediction in predictions :
        final_predictions += prediction
    predicted_labels = np.argmax(final_predictions, axis=1)
    return predicted_labels


def accuracy(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
 

def accuracy_adaboost(adaboost, x_test, y_test):
    predicted_labels = predict(adaboost, x_test)
    true_labels = np.argmax(y_test, axis=1) 
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy