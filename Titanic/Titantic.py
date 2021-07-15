import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
pd.options.mode.chained_assignment = None  # default='warn'

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
def preprocess(train_data):
	if 'Survived' in train_data:
		train_data = train_data[['Pclass', 'Sex', 'Age', 'SibSp','Fare','Survived']]
	else:
		train_data = train_data[['Pclass', 'Sex', 'Age', 'SibSp','Fare']]
	age_mean = train_data.Age.mean()
	fare_mean = train_data.Fare.mean()
	# deal with NaN
	train_data['Age'] = train_data['Age'].fillna(age_mean)
	train_data['Fare'] = train_data['Fare'].fillna(fare_mean)
	# check NaN
	# print(train_data.isnull().values.sum())
	# preprocess
	if 'Survived' in train_data:
		label = train_data[['Survived']]
		label = label.to_numpy()
	else:
		label = np.array([])
	# convert sexual to 1/0
	train_data = train_data.replace(['male','female'],[1,0])
	
	if 'Survived' in train_data:
		train_data = train_data.drop(['Survived'], axis = 1)
	# data for SVM

	train_data = train_data.to_numpy()
	return train_data, label

train, label_train = preprocess(train_data)[0], preprocess(train_data)[1]

test = preprocess(test_data)[0]

XGB = XGBClassifier()
XGB.fit(train, label_train)
result_XGB = XGB.predict(test)

# clf = RandomForestClassifier(n_jobs=10, random_state=0)
# clf.fit(train, label_train.ravel())
# result = clf.predict(test)

df = pd.DataFrame(result_XGB)
temp = pd.read_csv('gender_submission.csv')
temp['Survived'] = df
temp.to_csv(r'gender_submission.csv', index=False)
