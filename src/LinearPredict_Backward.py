import pandas as pd 
import numpy as np




data = pd.read_csv("data/data.csv")

print(data)
outlook= data.iloc[:,:1].values

#categorical to numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
outlook[:,0] =le.fit_transform(data.iloc[:,0])

one = preprocessing.OneHotEncoder()
outlook = one.fit_transform(outlook).toarray()




windy = data.iloc[:,-2:-1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
windy[:,-1] = le.fit_transform(data.iloc[:,-2:-1])
print(windy)

one = preprocessing.OneHotEncoder()

windy = one.fit_transform(windy).toarray()
print(windy)


play = data.iloc[:,-1:].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
play[:,-1] = le.fit_transform(data.iloc[:,-1])
print(play)

one = preprocessing.OneHotEncoder()
play = one.fit_transform(play).toarray()
print(play)

s1 = pd.DataFrame(data = outlook, index = range(14), columns =['o','r','s'] )
s2= pd.DataFrame(data = windy[:,-1], index = range(14), columns = ["windy"])
s3 = pd.DataFrame(data = play[:,-1], index = range(14), columns =["play"])

x_variables = pd.concat([s1,data.iloc[:,1:2],s2], axis=1)
x_variables_last = pd.concat([x_variables, s3], axis = 1 )
humudity = data.iloc[:,2:3]
 


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(x_variables_last, humudity, test_size =0.33, random_state =  0)

from sklearn.linear_model import LinearRegression

lr= LinearRegression()

reg = lr.fit(x_train, y_train)
y_predict = reg.predict(x_test)

print(y_predict)
print(y_test)
print(x_variables_last)


#backward eliminiton algoritm

import statsmodels.api as sm 

X = np.append(np.ones((14,1)).astype(int), values = x_variables_last, axis = 1 )

X_l = x_variables_last.iloc[:,[0,1,2,3,5]].values

X_l = np.array(X_l, dtype=float)

model =sm.OLS(humudity, X_l).fit()

print(model.summary())


























