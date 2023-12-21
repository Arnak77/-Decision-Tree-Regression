import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset=pd.read_csv(r"D:\NIT\DECEMBER\18 DEC(POLY..)\18th\emp_sal.csv")

X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
########################################
from sklearn.neighbors import KNeighborsRegressor
re1=KNeighborsRegressor()
re1.fit(X, y)

y_pred=re1.predict(X)

# predicto 
knn_model_pred =re1.predict([[6.5]])
knn_model_pred

##############################

from sklearn.neighbors import KNeighborsRegressor
re2=KNeighborsRegressor(n_neighbors=3)
re2.fit(X, y)



# predicto 
knn_mode2_pred =re2.predict([[6.5]])
knn_mode2_pred

#####################################


from sklearn.neighbors import KNeighborsRegressor
re3=KNeighborsRegressor(n_neighbors=6)
re3.fit(X, y)


# predicto 
knn_mode3_pred=re3.predict([[6.5]])
knn_mode3_pred


####################################




from sklearn.neighbors import KNeighborsRegressor
re4=KNeighborsRegressor(n_neighbors=6, weights="distance")
re4.fit(X, y)


# predicto 
knn_mode4_pred=re4.predict([[6.5]])
knn_mode4_pred
############################################





from sklearn.neighbors import KNeighborsRegressor
re5=KNeighborsRegressor( weights="distance")
re5.fit(X, y)


# predicto 
knn_mode5_pred=re5.predict([[6.5]])
knn_mode5_pred

############################################



from sklearn.neighbors import KNeighborsRegressor
re6=KNeighborsRegressor(n_neighbors=6, weights="distance",algorithm="ball_tree")
re6.fit(X, y)


# predicto 
knn_mode6_pred=re6.predict([[6.5]])
knn_mode6_pred
############################################



from sklearn.neighbors import KNeighborsRegressor
re7=KNeighborsRegressor( weights="distance",algorithm="ball_tree")
re7.fit(X, y)


# predicto 
knn_mode7_pred=re7.predict([[6.5]])
knn_mode7_pred

############################################



from sklearn.neighbors import KNeighborsRegressor
re8=KNeighborsRegressor( algorithm="ball_tree")
re8.fit(X, y)


# predicto 
knn_mode8_pred=re8.predict([[6.5]])
knn_mode8_pred
############################################




from sklearn.neighbors import KNeighborsRegressor
re9=KNeighborsRegressor(n_neighbors=6, weights="distance",algorithm="kd_tree")
re9.fit(X, y)


# predicto 
knn_mode9_pred=re9.predict([[6.5]])
knn_mode9_pred
############################################


from sklearn.neighbors import KNeighborsRegressor
re10=KNeighborsRegressor(n_neighbors=6,algorithm="kd_tree")
re10.fit(X, y)


# predicto 
knn_mode10_pred=re10.predict([[6.5]])
knn_mode10_pred

############################################



from sklearn.neighbors import KNeighborsRegressor
re11=KNeighborsRegressor(n_neighbors=5,algorithm="kd_tree")
re11.fit(X, y)


# predicto 
knn_mode11_pred=re11.predict([[6.5]])
knn_mode11_pred
############################################

from sklearn.neighbors import KNeighborsRegressor
re12=KNeighborsRegressor(n_neighbors=6,algorithm="brute")
re12.fit(X, y)


# predicto 
knn_mode12_pred=re12.predict([[6.5]])
knn_mode12_pred

############################################

from sklearn.neighbors import KNeighborsRegressor
re13=KNeighborsRegressor(n_neighbors=6, weights="distance",algorithm="brute")
re13.fit(X, y)


# predicto 
knn_mode13_pred=re13.predict([[6.5]])
knn_mode13_pred
############################################



from sklearn.neighbors import KNeighborsRegressor
re14=KNeighborsRegressor(algorithm="brute")
re14.fit(X, y)


# predicto 
knn_mode14_pred=re14.predict([[6.5]])
knn_mode14_pred
############################################


