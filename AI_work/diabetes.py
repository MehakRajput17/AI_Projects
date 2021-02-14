import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#load data and extract independent and dependent variable
df = pd.read_csv(r'dataset\diabetes.csv')
print(df)
print("======================Null Values=====================")
print(df.isnull().sum())
# import seaborn as sns
# sns.pairplot(df,hue="Outcome")
# plt.show()

print("=======shape=======")
print(df.shape)
print("======info======")
print(df.info())

X = df.iloc[:,:-1]
y = df.iloc[:,-1:]
print("+========= y unique ==============")
print(pd.unique(df['Outcome']))
print("======================X Data==============================")
print(X)
print("==================================Y Data===============================")
print(y)

#Splitting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
model=rf.fit(X_train, y_train)
prediction = rf.predict(X_test)

# =====================ACCUARACY===========================
print("=====================Training Accuarcy RF=============")
trac=rf.score(X_train,y_train)
trainingAcc=trac*100
print(trainingAcc)
print("====================Testing Accuracy RF============")
teac=accuracy_score(y_test,prediction)
testingAcc=teac*100
print(testingAcc)

import pickle
pickle.dump(model, open('Flask/pickel files/diabetes.pkl', 'wb'))