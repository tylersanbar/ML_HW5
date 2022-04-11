from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
import numpy as np

training_data = np.loadtxt("train.csv",skiprows=1,delimiter=",")
print(training_data.shape)

delta1 = [0,.000001,.00001,.0001,.001,.01,.1,1]
delta2 = [0,.000001,.00001,.0001,.001,.01,.1,1]
X = training_data[:,0:80]
y = training_data[:,81]

model = ElasticNet
X, y = make_regression(n_features=81, random_state=0)
# regr = ElasticNet(random_state=0)
# regr.fit(X, y)
# print(regr.coef_)
