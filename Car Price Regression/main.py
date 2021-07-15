import os
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


file_name = os.listdir()
file_name.remove('main.py')
# print((file_name))
k = 0
for i in range(len(file_name)):
	name = (file_name[i].split('.')[0])
	tempdf = pd.read_csv(file_name[i])
	df = tempdf if k == 0 else pd.concat([df,tempdf], join="inner")
	k +=1
def detect_data_type(df):
	numerical =[]
	nominal = []
	for i in df.columns.to_list():
		if (str(df[str(i)].dtypes)) == 'object':
			nominal.append(i)
		else:
			numerical.append(i)

	return numerical, nominal
df = df.drop(['model'], axis=1)
# divide dataset to numerical columns and nominal columns
numerical_column, nominal_column = detect_data_type(df)
# detect null element
def detect_null(df):
	for i in df.columns:
		if df[i].isnull().values.any():
			print(i+' has null!')
		else:
			print(i+' has no null')
# show countplot of nominal attribute
def show_nominal(nominal_column):
	for i in range(len(nominal_column)):
		plt.figure(i,figsize=(10,5))
		sns.countplot(x = nominal_column[i], data = df)
	plt.show()
# show_nominal(nominal_column)
# show distribution of numerical attribute

# plt.figure(5,figsize=(8,4))
# sns.distplot(df[numerical_column[5]])
# plt.show()

# labelize by standard deviation and min,max
def one_hot(df):
	df = df.replace({'Other': None}).dropna()
	transimission = df[nominal_column[0]].str.get_dummies()
	fuel = df[nominal_column[1]].str.get_dummies()
	new_df = pd.concat([df, transimission, fuel], axis=1)
	new_df = new_df.drop(columns = [nominal_column[0], nominal_column[1]])
	return new_df
new_df = one_hot(df)

# remove abnormal value
new_df = new_df.drop(new_df[new_df.year>2021].index)
new_df['year'] = new_df['year'].sub(new_df['year'].max()).abs()

# scale data
normali = MinMaxScaler()
new_df[['mileage']] = normali.fit_transform(new_df[['mileage']])
new_df[['engineSize']] = normali.fit_transform(new_df[['engineSize']])
new_df[['year']] = normali.fit_transform(new_df[['year']])

# partition for label
def labelize(df):
	min_val = df['price'].min()
	max_val = df['price'].max()
	std = df['price'].std()/0.95
	for k in range(0,8):
		df['price'] = df['price'].mask((df['price']>= min_val+k*std*2) & (df['price'] <= min_val+(k+1)*std*2), k)

	return df['price']
label = labelize(new_df)
new_df = new_df.drop(['price'], axis=1)

# evaluate mutuial information
def mutual(new_df, label):
	mi_scores = mutual_info_classif(new_df,label)
	mi_scores = pd.Series(mi_scores, name="MI Scores", index=new_df.columns)
	mi_scores = mi_scores.sort_values(ascending=False)
	return mi_scores
# print(mutual(new_df,label))
X_train, X_test, y_train, y_test = train_test_split(new_df, label, test_size=0.9, random_state=42)

rdf = RandomForestClassifier(n_jobs=-1, random_state=10)
rdf.fit(X_train,y_train.ravel())
result_rdf = rdf.predict(X_test)

XGB = XGBClassifier(objective='multi:softprob')
XGB.fit(X_train, y_train)
result_XGB = XGB.predict(X_test)
print(classification_report(y_test, result_rdf,zero_division = 0))
print(classification_report(y_test, result_XGB,zero_division = 0))

# SVM for small data set and performe bad
# sc= SVC()
# sc.fit(X_train,y_train.ravel())
# sc = rdf.predict(X_test)
# print(classification_report(y_test, sc,zero_division = 0))
