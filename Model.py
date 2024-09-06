import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle

df=pd.read_csv('HousePricePrediction\ParisHousing.csv')
df.head()
d=df.drop(['hasYard','hasPool','isNewBuilt','hasStormProtector','hasStorageRoom','made'], axis=1)
X=d.drop(['price'],axis=1)
y=d.price

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25,random_state=55)
print(X_train.shape,Y_train.shape)
X_train.describe()

model=linear_model.LinearRegression()
model.fit(X_train,Y_train)

print(model.coef_)
print(model.intercept_)

y_pred_train=model.predict(X_train)
y_pred_test=model.predict(X_test)

Model_Test=pd.DataFrame(X_test)
Model_Test['Test Actual Outcome']=Y_test
Model_Test['Test Predicted Outcome']=y_pred_test
Model_Test['Test Predicted Outcome']=Model_Test['Test Predicted Outcome'].astype('int64')
Model_Test['Test Actual Outcome']=Model_Test['Test Actual Outcome'].astype('int64')

from sklearn.metrics import r2_score
print(r2_score(Y_test,y_pred_test))
print(r2_score(Y_train,y_pred_train))
pickle.dump(model,open ('HousePrdictionParis.pkl','wb'))