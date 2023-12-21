
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
re2=DecisionTreeRegressor( criterion='friedman_mse',random_state=0)
re2.fit(X, y)

tree2=re2.predict([[6.5]])
tree2
######################################


from sklearn.tree import DecisionTreeRegressor
re3=DecisionTreeRegressor( criterion='absolute_error',random_state=0)
re3.fit(X, y)

tree3=re3.predict([[6.5]])
tree3
######################################

from sklearn.tree import DecisionTreeRegressor
re4=DecisionTreeRegressor( criterion='poisson',random_state=0)
re4.fit(X, y)

tree4=re4.predict([[6.5]])
tree4

######################################

from sklearn.tree import DecisionTreeRegressor
re9=DecisionTreeRegressor( criterion='friedman_mse',splitter="random",random_state=0)
re9.fit(X, y)

tree9=re9.predict([[6.5]])
tree9
######################################

from sklearn.tree import DecisionTreeRegressor
re5=DecisionTreeRegressor( criterion='poisson',splitter="random",random_state=0)
re5.fit(X, y)

tree5=re5.predict([[6.5]])
tree5
######################################

from sklearn.tree import DecisionTreeRegressor
re6=DecisionTreeRegressor( criterion='poisson',splitter="random",min_samples_split=4,random_state=0)
re6.fit(X, y)

tree6=re6.predict([[6.5]])
tree6
######################################


from sklearn.tree import DecisionTreeRegressor
re7=DecisionTreeRegressor( criterion='friedman_mse',splitter="random",min_samples_split=7,random_state=0)
re7.fit(X, y)

tree7=re7.predict([[6.5]])
tree7
######################################
from sklearn.tree import DecisionTreeRegressor
re8=DecisionTreeRegressor( criterion='poisson',min_samples_split=4)
re8.fit(X, y)

tree8=re8.predict([[6.5]])
tree8