import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler


#visulizing different attribution distribtuon and count them，drop minor part
#convet binary type categorical to zero or one
#numerical data --> standard scaler
#big gap data --> minmax scaler
#keep outlier --> robust scaler


sns.set(style='whitegrid')
percent = 0.01
df = pd.read_csv('train.csv')
train = df.head(round(df.shape[0]*(1-percent)))
test = df.tail(round(df.shape[0]*percent))

num_feat = ['Age','Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes','Region_Code','Policy_Sales_Channel']
train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
train=pd.get_dummies(train,drop_first=True)
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')

train['Region_Code'] = train['Region_Code'].factorize()[0] 
train['Policy_Sales_Channel'] = train['Policy_Sales_Channel'].factorize()[0] 

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
train[num_feat] = ss.fit_transform(train[num_feat])


mm = MinMaxScaler()
train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])
train=train.drop('id',axis=1)
for column in cat_feat:
    train[column] = train[column].astype('str')

test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test=pd.get_dummies(test,drop_first=True)
test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')

test['Region_Code'] = test['Region_Code'].factorize()[0] 
test['Policy_Sales_Channel'] = test['Policy_Sales_Channel'].factorize()[0] 

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
test[num_feat] = ss.fit_transform(test[num_feat])


mm = MinMaxScaler()
test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])
test=test.drop('id',axis=1)
for column in cat_feat:
    test[column] = test[column].astype('str')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

x_train,x_test,y_train,y_test = train.drop(['Response'], axis = 1), test.drop(['Response'], axis = 1), train['Response'], test['Response']

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

clf = RandomForestClassifier(n_jobs=-1, random_state=0)
clf.fit(x_train, y_train)


y_test = y_test.to_numpy()

result = clf.predict(x_test).reshape((-1,1))

array = np.hstack((result, y_test.reshape((-1,1))))
print(array.shape)
dddd = pd.DataFrame(array,columns = ['result','target'])
print(dddd['result'].value_counts())
print(dddd['target'].value_counts())
print (classification_report(y_test, result))

