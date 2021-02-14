import pandas as pd
import numpy as np
import pickle


df = pd.read_csv(r"C:\Users\Raheel\Desktop\HousePrices_HalfMil.csv")

x = df.iloc[:,:15]
y = df.iloc[:,15:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


from sklearn.tree import DecisionTreeRegressor

lr = DecisionTreeRegressor(random_state=0)
model = lr.fit(x_train, y_train)
# from sklearn.preprocessing import PolynomialFeatures
#
# poly = PolynomialFeatures(degree=2)
# X_Poly = poly.fit_transform(x)
#
# from sklearn.linear_model import LinearRegression
#
# Poly_Rg = LinearRegression()
# t = Poly_Rg.fit(x_train, y_train)

# from sklearn.linear_model import LinearRegression
#
# lr = LinearRegression()
# model = lr.fit(x_train, y_train)


pickle.dump(model,open('11.pkl','wb'))


