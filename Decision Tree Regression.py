import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset=pd.read_csv(r"D:\NIT\DECEMBER\18 DEC(POLY..)\18th\emp_sal.csv")

X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


##########################################
from sklearn.tree import DecisionTreeRegressor
re1=DecisionTreeRegressor()
re1.fit(X, y)

tree=re1.predict([[6.5]])
tree
######################################

from sklearn.tree import DecisionTreeRegressor
re2=DecisionTreeRegressor( criterion='friedman_mse')
re2.fit(X, y)

tree2=re2.predict([[6.5]])
tree2
######################################


from sklearn.tree import DecisionTreeRegressor
re3=DecisionTreeRegressor( criterion='absolute_error')
re3.fit(X, y)

tree3=re3.predict([[6.5]])
tree3
######################################

from sklearn.tree import DecisionTreeRegressor
re4=DecisionTreeRegressor( criterion='poisson')
re4.fit(X, y)

tree4=re4.predict([[6.5]])
tree4

######################################

from sklearn.tree import DecisionTreeRegressor
re5=DecisionTreeRegressor( criterion='friedman_mse',splitter="random")
re5.fit(X, y)

tree5=re5.predict([[6.5]])
tree5
######################################

from sklearn.tree import DecisionTreeRegressor
re5=DecisionTreeRegressor( criterion='poisson',splitter="random")
re5.fit(X, y)

tree5=re5.predict([[6.5]])
tree5