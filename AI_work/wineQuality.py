import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

df= pd.read_csv(r'dataset/winequalityN.csv')
#Drop Null values
df2=df.dropna()

X=df2.iloc[:,:12]

y=df2.iloc[:,12:]

# #Apply one HotEncoder
le=LabelEncoder()
X.type=le.fit_transform(X.type)
print(X)


x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.25)
print("=========== x train ==============")
print(x_train)
print("=========== x test ==============")
print(x_test)
print("=========== y train ==============")
print(y_train)
print("=========== y test ==============")
print(y_test)
from sklearn.preprocessing import StandardScaler
scc=StandardScaler()
x_train=scc.fit_transform(x_train)
x_test=scc.transform(x_test)

# #SVM
svmx = svm.SVC()
svmx.fit(x_train,y_train)
SVMprediction = svmx.predict(x_test)

trac=svmx.score(x_train,y_train)
trainingAcc=trac*100
SVMtrac='Training accuracy',trainingAcc,'%'
print("=========Training SVM=========")
print(SVMtrac)
teacLr=accuracy_score(y_test,SVMprediction)
testingAccLR=teacLr*100
SVMtesac='Testing accuracy',testingAccLR,'%'
print("=========Testing SVM=========")
print(SVMtesac)

