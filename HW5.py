from math import inf
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
import numpy as np
import itertools

def mse(y, h):
    sum = 0
    for i in range(len(y)):
        sum += (y[i] - h[i]) ** 2
    mse = sum / len(y)
    return mse

def lambdas2alpha(params):
    a_l1 = []
    for i in range(len(params)):
        if params[i][0] == 0 or params[i][0] > params[i][1]: a_l1.append(None)
        else: 
            alpha = params[i][0] + params[i][1]
            l1 = params[i][0]/alpha
            a_l1.append([alpha,l1])
    return a_l1

def bestMSE(X, y, models):
    MSE = []
    hypothesis = []
    smallestError = inf
    bestIndex = None
    for i in range(len(models)):
        model = models[i]
        if model is not None:
            h = model.predict(X)
            hypothesis.append(h)
            error = mse(y, h)
            MSE.append(error)
            if error < smallestError: 
                smallestError = error
                bestIndex = i
        else:
            MSE.append(None)
            hypothesis.append(None)
    return bestIndex, MSE, hypothesis  

def trainModels(X, y, a_l1):
    models = []
    for i in range(len(a_l1)):
        if a_l1[i] is not None:
            alpha = a_l1[i][0]
            l1_ratio = a_l1[i][1]
            model = ElasticNet(alpha = alpha, l1_ratio=l1_ratio)
            model.fit(X, y)
            models.append(model)
        else: models.append(None)
    return models

def printMSE(params, h, MSE):
    for i in range(len(params)):
        if h[i] is not None: print("Lambda1:",params[i][0],"Lambda2:",params[i][1],"Prediction:",h[i][i],"MSE",MSE[i])

def printBestMSE(params, h, MSE, bestIndex):
    print("Best - Lambda1:",params[bestIndex][0],"Lambda2:",params[bestIndex][1],"Prediction:",h[bestIndex][bestIndex],"MSE:",MSE[bestIndex])

#Get data from CSVs
training_data = np.loadtxt("train.csv",skiprows=1,delimiter=",")
validation_data = np.loadtxt("validate.csv",skiprows=1,delimiter=",")
testing_data = np.loadtxt("test.csv",skiprows=1,delimiter=",")

#alpha * l1_ratio
lambda1 = [0,.000001,.00001,.0001,.001,.01,.1,1]
#alpha * (1-l1_ratio)
lambda2 = [0,.000001,.00001,.0001,.001,.01,.1,1]

##Get valid a_l1 values
params = list(itertools.product(lambda1, lambda2))
a_l1 = lambdas2alpha(params)

#a) Fit an elastic net model to the training data with each possible combination of the following L1 and L2 regularization weights.
X = training_data[:,0:80]
y = training_data[:,81]
training_Models = trainModels(X, y, a_l1)

#b) For each model trained in step (a), make a prediction for each training example, using the
# predict method for sklearn.linear_model.ElasticNet and calculate the mean squared
# error (MSE) on the training examples.
print("---Training Data---")
training_bestIndex, training_MSE, training_h = bestMSE(X, y, training_Models)
printMSE(params, training_h, training_MSE)

#c) Make a prediction for each validation example
# and calculate the mean squared error on the validation examples
print("---Validation Data---")
X = validation_data[:,0:80]
y = validation_data[:,81]

validation_Models = trainModels(X, y, a_l1)
validation_bestIndex, validation_MSE, validation_h = bestMSE(X, y, validation_Models)
printMSE(params, validation_h, validation_MSE)  

#d) Which model (i.e., pair of λ1 and λ2) performed best on the training data?
printBestMSE(params, training_h, training_MSE, training_bestIndex)
#Which model performed best on the validation data?
printBestMSE(params, validation_h, validation_MSE, validation_bestIndex)

#e)Find the best hyperparameter set (pair of λ1 and λ2) on the validation data. Train a model with
#the same λ1 and λ2 on both the training and validation data. 
#Using this model, make predictions for each testing example and calculate the mean squared error on the test examples.

X = np.append(training_data[:,0:80], validation_data[:,0:80], axis=0)
y = np.append(training_data[:,81], validation_data[:,81], axis=0)

best_a_l1 = [a_l1[validation_bestIndex]]
combined_model = trainModels(X, y, best_a_l1)
combined_bestIndex, combined_MSE, combined_h = bestMSE(X, y, combined_model)
print("MSE:", combined_MSE[0])