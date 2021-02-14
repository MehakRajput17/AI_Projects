import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'dataset/oasis_longitudinal.csv')
print("=====================Data ===========================")
print(df)

#here we using drop code to drop columns that wont help in analyzing data
df = df.drop(columns = ['Subject ID','MRI ID','Hand'])
df = pd.DataFrame(df)
#checking Null values
print("======================NULL VALUES==========================")
print(df.isnull().sum()) #SES and MMSE has null values


#here we are removing the null values using mean
me_an = np.nanmean(df['SES'])
df['SES'].fillna(me_an,inplace= True )
me_an = np.nanmean(df['MMSE'])
df['MMSE'].fillna(me_an,inplace= True )

def checkingdata():
    print("============================shape====================================")
    print(df.shape)
    print("============================size====================================")
    print(df.size)
    print("============================info====================================")
    print(df.info())
    print("============================stats====================================")
    print(df.describe())
    print("====================Dimension=======================")
    ab1 = np.ndim(df)
    print("Dimension:",ab1)
    print("====================Minimum=======================")
    print("Minimum: ",df.min())
    print("=====================Maximum======================")
    print("Maximum: ",df.max())

    print("====================Shape=================")
    print("Shape: ",df.shape)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# here we are encoding the colums that contain string
df['Group'] = le.fit_transform(df['Group'])
d1 = {'M': 1, 'F': 2}
df['M/F'].replace(d1, inplace=True)

yy=pd.unique(df['Group'])
print("==== Y unique ================")
print(yy)

y = df.iloc[:,:1]
x = df.iloc[:,1:]
print("==================== X data=========")
print(x.columns)

#it is used to break the data into testing and Training Data
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,train_size=0.7,random_state=32)


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
modelRF=rf.fit(x_train,y_train)

prediction6 = modelRF.predict(x_test)

from sklearn.metrics import accuracy_score
# =====================ACCUARACY===========================
print("=====================Training Accuarcy RF=============")
trac=rf.score(x_train,y_train)
trainingAccLR=trac*100
print(trainingAccLR)
print("====================Testing Accuracy RF============")
teacLr6=accuracy_score(y_test,prediction6)
testingAccLR6=teacLr6*100
print(testingAccLR6)
import pickle
pickle.dump(modelRF, open('Flask/pickel files/AD.pkl', 'wb'))