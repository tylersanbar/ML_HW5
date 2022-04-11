from math import inf
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
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
            model = ElasticNet(alpha = alpha, l1_ratio=l1_ratio, tol = .1)
            model.fit(X, y)
            models.append(model)
        else: models.append(None)
    return models

def printMSE(params, MSE):
    for i in range(len(MSE)):
        if MSE[i] is not None: print("Lambda1:",params[i][0],"Lambda2:",params[i][1],"MSE",MSE[i])

def printBestMSE(params, MSE, bestIndex):
    print("Best - Lambda1:",params[bestIndex][0],"Lambda2:",params[bestIndex][1],"MSE:",MSE[bestIndex])

def getXY(data, num_features):
    X = data[:,0:num_features-1]
    y = data[:,num_features]
    return X, y

def exercise(training_data, validation_data, testing_data, num_features):
    #alpha * l1_ratio
    lambda1 = [0,.000001,.00001,.0001,.001,.01,.1,1]
    #alpha * (1-l1_ratio)
    lambda2 = [0,.000001,.00001,.0001,.001,.01,.1,1]

    ##Get valid a_l1 values
    params = list(itertools.product(lambda1, lambda2))
    a_l1 = lambdas2alpha(params)

    #a) Fit an elastic net model to the training data with each possible combination of the following L1 and L2 regularization weights.
    X, y = getXY(training_data, num_features)
    training_Models = trainModels(X, y, a_l1)

    #b) For each model trained in step (a), make a prediction for each training example, using the
    # predict method for sklearn.linear_model.ElasticNet and calculate the mean squared
    # error (MSE) on the training examples.
    print("---Training Data---")
    training_bestIndex, training_MSE, training_h = bestMSE(X, y, training_Models)
    printMSE(params, training_MSE)

    #c) Make a prediction for each validation example
    # and calculate the mean squared error on the validation examples
    print("---Validation Data---")
    X, y = getXY(validation_data, num_features)

    validation_Models = trainModels(X, y, a_l1)
    validation_bestIndex, validation_MSE, validation_h = bestMSE(X, y, validation_Models)
    printMSE(params, validation_MSE)  

    #d) Which model (i.e., pair of λ1 and λ2) performed best on the training data?
    print("---Best MSE---")
    print("Training Data:")
    printBestMSE(params, training_MSE, training_bestIndex)
    #Which model performed best on the validation data?
    print("Validation Data:")
    printBestMSE(params, validation_MSE, validation_bestIndex)

    #e)Find the best hyperparameter set (pair of λ1 and λ2) on the validation data. Train a model with
    #the same λ1 and λ2 on both the training and validation data. 
    #Using this model, make predictions for each testing example and calculate the mean squared error on the test examples.

    combined_data = np.append(training_data, validation_data, axis = 0)
    X, y = getXY(combined_data, num_features)

    best_a_l1 = [a_l1[validation_bestIndex]]
    combined_model = trainModels(X, y, best_a_l1)
    
    #Testing Data
    X, y = getXY(testing_data, num_features)
    testing_bestIndex, testing_MSE, testing_h = bestMSE(X, y, combined_model)
    print("---Testing Data---")
    print("MSE:", testing_MSE[0])

#Exercise 2

#Get data from CSVs
sc_training_data = np.loadtxt("train.csv",skiprows=1,delimiter=",")
sc_validation_data = np.loadtxt("validate.csv",skiprows=1,delimiter=",")
sc_testing_data = np.loadtxt("test.csv",skiprows=1,delimiter=",")

exercise(sc_training_data,sc_validation_data,sc_testing_data, 81)

#Exercise 3

#Get data from CSVs
stab_training_data = np.loadtxt("UCITraining.csv",skiprows=1,delimiter=",")
stab_validation_data = np.loadtxt("UCIValidation.csv",skiprows=1,delimiter=",")
stab_testing_data = np.loadtxt("UCITesting.csv",skiprows=1,delimiter=",")

exercise(stab_training_data,stab_validation_data,stab_testing_data, 11)

#Exercise 4

#a)
regularModel = LogisticRegression()
nonRegularModel = LogisticRegression(penalty='none')
X, y = getXY(stab_training_data, 11)
regularModel.fit(X, y)
nonRegularModel.fit(X, y)

#b)
X, y = getXY(stab_validation_data, 11)
h_reg = regularModel.predict(X)
h_non = nonRegularModel.predict(X)