import pandas as pd
df=pd.read_csv(r"C:\Users\mehak\Downloads\heart.csv")

print("=========== Encoding=============")

import pickle
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df.target=le.fit_transform(df.target)
print(df)

x=df.iloc[0:,0:13]
print("========= X Data =========")
y=df.iloc[0:,13:]
print(x)
print('========== Y Data ==========')
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print("======== X Train ========")
print(x_train)
print(x_train.shape)
print("======== X Test ========")
print(x_test)
print(x_test.shape)
print("======== Y Train ========")
print(y_train)
print(y_train.shape)
print("======== Y test ========")
print(y_test)
print(y_test.shape)

print("============ Model Trainig =============")

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(random_state=0)
modlr=lr.fit(x_train,y_train)
pickle.dump(modlr, open('hd.pkl', 'wb'))
prelr=modlr.predict(x_test)
print("===============Model Prediction=======================")
print(prelr)
print("===============ACtual ANswer=======================")
print(y_test)

from sklearn.metrics import accuracy_score
print("========= Logistic Regression TRAINING ACC ================")
lrtrac=modlr.score(x_train,y_train)
print(lrtrac)
print("========= Logistic Regression Testing ACC================")
lrtsac=accuracy_score(y_test,prelr)
print(lrtsac)





