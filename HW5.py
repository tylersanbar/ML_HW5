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
        if params[i][0] == 0 or params[i][0] < params[i][1]: a_l1.append(None)
        else: 
            alpha = params[i][0] + params[i][1]
            l1 = params[i][0]/alpha
            a_l1.append([alpha,l1])
    return a_l1

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

X = training_data[:,0:80]
y = training_data[:,81]

models = []
for i in range(len(a_l1)):
    if a_l1[i] is not None:
        alpha = a_l1[i][0]
        l1_ratio = a_l1[i][1]
        model = ElasticNet(alpha = alpha, l1_ratio=l1_ratio)
        model.fit(X, y)
        models.append(model)
    else: models.append(None)
#a) MSE for training examples
print("---Training---")
trainingMSE = []
hypothesis = []
smallestError = inf
bestIndex = None
for i in range(len(models)):
    model = models[i]
    if model is not None:
        h = model.predict(X)
        hypothesis.append(h)
        error = mse(y, h)
        trainingMSE.append(error)
        if error < smallestError: 
            smallestError = error
            bestIndex = i
        print("Lambda1:",params[i][0],"Lambda2:",params[i][1],"Prediction:",h[i],"MSE",trainingMSE[i])
    else:
        trainingMSE.append(None)
        hypothesis.append(None)
    
print("Best - Lambda1:",params[bestIndex][0],"Lambda2:",params[bestIndex][1],"Prediction:",h[bestIndex],"MSE:",trainingMSE[bestIndex])

#b) MSE for validation examples
print("---Validation---")
X = validation_data[:,0:80]
y = validation_data[:,81]
validationMSE = []
hypothesis = []
smallestError = inf
bestIndex = None
for i in range(len(models)):
    model = models[i]
    if model is not None:
        h = model.predict(X)
        hypothesis.append(h)
        error = mse(y, h)
        validationMSE.append(error)
        if error < smallestError: 
            smallestError = error
            bestIndex = i
        print("Lambda1:",params[i][0],"Lambda2:",params[i][1],"Prediction:",h[i],"MSE",validationMSE[i])
    else:
        validationMSE.append(None)
        hypothesis.append(None)
    
print("Best - Lambda1:",params[bestIndex][0],"Lambda2:",params[bestIndex][1],"Prediction:",h[bestIndex],"MSE:",validationMSE[bestIndex])

X = np.append(training_data[:,0:80], validation_data[:,0:80], axis=0)
y = np.append(training_data[:,81], validation_data[:,81], axis=0)
model = ElasticNet(alpha = a_l1[bestIndex][0], l1_ratio=a_l1[bestIndex][1])
model.fit(X,y)
h = model.predict(X)
error = mse(y, h)
print("MSE:", error)