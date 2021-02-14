from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'dataset/hepatitis.csv')
print(df)

# lb = LabelEncoder()
# df.Species = lb.fit_transform(df['Species'])

x = df.iloc[0:,0:19]
y = df.iloc[0:,19:20]
print("========X ==================")
print(x)
print("============Y=================")
print(y)

x_test,x_train,y_test,y_train = train_test_split(x,y,test_size=0.7,random_state=33)

DT = DecisionTreeClassifier(random_state=33)
ModelDT = DT.fit(x_train, y_train)
PredictionDT = DT.predict(x_test)

# =====================ACCUARACY===========================
print("=====================DT Training Accuarcy=============")
tracDT=DT.score(x_train,y_train)
trainingAccDT=tracDT*100
print(trainingAccDT)
print("====================DT Testing Accuracy============")
teacDT=accuracy_score(y_test,PredictionDT)
testingAccDT=teacDT*100
print(testingAccDT)


import pickle
pickle.dump(ModelDT, open('Flask/pickel files/hep.pkl', 'wb'))