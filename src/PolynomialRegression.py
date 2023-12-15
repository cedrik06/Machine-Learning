""" Polynomial Regresyon yaparken önce , x degerlerini 
belirlediğimiz degree derecesine göre dönüşm yapıp daha sonra 
Linear Regresyona sokarız 
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("./data/maaslar.csv")



EDU = data.iloc[:,1:2].values 
PR =  data.iloc[:,-1:].values

#linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(EDU,PR)
PR_LİN_PRED = lr.predict(EDU)

#2 degree Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures


PF2 = PolynomialFeatures(degree=2)
EDU_POL2 = PF2.fit_transform(EDU)


lr_pol1 = LinearRegression()
lr_pol1.fit(EDU_POL2,PR)
PR_POLY2_PRED = lr_pol1.predict(EDU_POL2)

#4 degree Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

PF4 = PolynomialFeatures(degree=4)
EDU_POL4 = PF4.fit_transform(EDU)

lr_pol2 = LinearRegression()
lr_pol2.fit(EDU_POL4,PR)
PR_POLY4_PRED = lr_pol2.predict(EDU_POL4)
print(EDU_POL4)



plt.scatter(EDU, PR, color = "blue")
plt.plot(EDU, PR_LİN_PRED, color = "red")
plt.plot(EDU, PR_POLY2_PRED, color = "green")
plt.plot(EDU, PR_POLY4_PRED, color = "black")
plt.show()


#regressions

print(lr_pol2.predict(PF4.fit_transform([[6.6]])))



