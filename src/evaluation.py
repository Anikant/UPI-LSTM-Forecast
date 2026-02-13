import numpy as np

def mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    eps = 1e-8
    return np.mean(np.abs(actual - predicted) / (np.abs(actual) + eps)) * 100

def rmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))
