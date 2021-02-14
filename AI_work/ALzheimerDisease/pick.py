import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('oasis_longitudinal.csv')


df = df.drop(columns = ['Subject ID','MRI ID','Hand'])
df = pd.DataFrame(df)

me_an = np.nanmean(df['SES'])
df['SES'].fillna(me_an,inplace= True )
me_an = np.nanmean(df['MMSE'])
df['MMSE'].fillna(me_an,inplace= True )


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Group'] = le.fit_transform(df['Group'])

d1 = {'M': 1, 'F': 2}
df['M/F'].replace(d1, inplace=True)


y = np.array(df.iloc[:,0:1])
x = np.array(df.iloc[:,1:12])

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,train_size=0.7)

# from sklearn.neighbors import KNeighborsClassifier
# # lr = LogisticRegression()
# knn = KNeighborsClassifier(n_neighbors=3)
# sv = knn.fit(x_train,y_train)

# from sklearn.svm import SVC
# sv = SVC(kernel='linear').fit(x_train,y_train)

from sklearn.ensemble import RandomForestClassifier
svr = RandomForestClassifier()
sv = svr.fit(x_train,y_train)
#
pickle.dump(sv, open('saif.pkl', 'wb'))

