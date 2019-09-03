#dataset at the location  https://github.com/munnupsk/machine-learning/blob/master/Salary_Data.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting linear regression to the dataset

from sklearn.linear_model import LinearRegression
l_reg1=LinearRegression()
l_reg1.fit(X,y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=5)
x_poly=poly_reg.fit_transform(X)
poly_reg.fit(x_poly,y)
l_reg2=LinearRegression()
l_reg2.fit(x_poly,y)

#visualizing the linear regression

plt.scatter(X,y,color="red")
plt.plot(X,l_reg1.predict(X),color="blue")
plt.xlabel("position level")
plt.ylabel("salary")
plt.title("truth or bluf of employee")
 

#visualizing the linear regression

plt.scatter(X,y,color="red")
plt.plot(X,l_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.xlabel("position level")
plt.ylabel("salary")
plt.title("truth or bluf of employee")
 
