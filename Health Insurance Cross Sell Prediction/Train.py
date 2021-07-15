import os
import csv
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


pd.options.mode.chained_assignment = None  # default='warn'

percent = 0.3
file = os.listdir()
df = pd.read_csv('train.csv')
train = df.head(round(df.shape[0]*(1-percent)))
test = df.tail(round(df.shape[0]*percent))

def preprocess(df):
	X = df.drop(columns = ['Response','id','Policy_Sales_Channel','Region_Code']).replace(['Male','Female'],[1,0])
	target = df['Response']

	X['Vehicle_Age'] = X['Vehicle_Age'].replace(['> 2 Years', '1-2 Year','< 1 Year'],[2,1,0])
	X['Vehicle_Damage'] = X['Vehicle_Damage'].replace(['Yes','No'],[1,0])
	for i in range(300//150):
		# &-and, |-or
		X['Vintage'] = X['Vintage'].mask((X['Vintage']>=i*150) & (X['Vintage']<=(i+1)*150),i)
	for i in range(85//5):
		X['Age'] = X['Age'].mask((X['Age']>=i*5) & (X['Vintage']<=(i+1)*5),i)
	for i in range(600000//300000):
		X['Annual_Premium'] = X['Annual_Premium'].mask((X['Annual_Premium']>=i*300000) & (X['Vintage']<=(i+1)*300000),i)
	
	return X, target

train_set = preprocess(train)
X,target= train_set[0], train_set[1]

mi_scores = mutual_info_classif(X, target,discrete_features=True)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_name = list(mi_scores.index)
print(mi_scores)
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

# plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_scores(mi_scores)
# plt.show()

# X = X[[mi_name[0],mi_name[1],mi_name[2]]]
# clf = RandomForestClassifier(n_jobs=-1, random_state=0)
# clf.fit(X, target.ravel())

# test_set = preprocess(test)
# X1, target1 = test_set[0], test_set[1]

# print(df['Region_Code'].value_counts())
# print(target1.value_counts())

# X1 = X1[[mi_name[0],mi_name[1],mi_name[2]]]
# result = clf.score(X1, target1.ravel())
# print('RF ='+str(result))

# XGboost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
# 	max_depth=1, random_state=0).fit(X, target.ravel())
# result_xgboost = XGboost.score(X1, target1.ravel())
# print('XGboost = '+str(result_xgboost))