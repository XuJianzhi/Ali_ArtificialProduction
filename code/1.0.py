
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
import math
#import time
#import gc
import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,GenericUnivariateSelect
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,PolynomialFeatures
 
#读取数据
way_data='/home/m/桌面/天池智能制造比赛/data/'
train0=pd.read_excel(way_data+'训练.xlsx').set_index('ID')
x_train0=train0.iloc[:,:-1]
y_train0=train0.Y
x_test0a=pd.read_excel(way_data+'测试A.xlsx').set_index('ID')
x_test0b=pd.read_excel(way_data+'测试B.xlsx').set_index('ID')

##################################

#合并train和test
all0=pd.concat([x_train0,x_test0a,x_test0b])

#由于数据中第302行和303行的ID都是“ID466”，故删去第303行
all0.iloc[303,:]=np.nan
all0.dropna(axis=0,how='all',inplace=True)

#找出“全空列”和“同值列”并删去
list_waste=[]
for i in all0.columns:
	data=all0[i][all0[i].notnull()]
	if data.shape[0]==0:
		list_waste.append(i)
		continue
	data=np.array(data)
	if data[data!=data[0]].shape[0]==0:
		list_waste.append(i)
all0.drop(list_waste,axis=1,inplace=True)

#scale（暂略）

#补缺
all0.fillna(-999,inplace=True)

#找出定性，编码、哑编码
list_qualitative=[]
for i in all0.columns:
	if all0[i].dtype not in [float,int]:
		list_qualitative.append(i)
		encoder=LabelEncoder()
		all0[i] = encoder.fit_transform(all0[i])
onehot=OneHotEncoder(sparse=False,dtype=np.int)
tempt_onehoted = pd.DataFrame(onehot.fit_transform(all0[list_qualitative]),index=all0.index)
all0.drop(list_qualitative,inplace=True,axis=1)
all0=pd.concat([all0,tempt_onehoted],axis=1)
		
#升维（暂略）

#分train和test
x_train1 = all0.loc[x_train0.index,:]
y_train1 = y_train0.copy()
x_test1a = all0.loc[x_test0a.index,:]
x_test1b = all0.loc[x_test0b.index,:]

#####################
xtrain, xtest, ytrain, ytest_real = train_test_split(x_train1, y_train1, test_size=0.5)
#####################

dtrain=xgb.DMatrix(xtrain,ytrain)
dtest=xgb.DMatrix(xtest,ytest_real)

watchlist=[(dtrain,'train'),(dtest,'test')]

params = {
			'objective': 'reg:gamma',
			'eval_metric':'rmse',
			'eta': 0.01,
			'seed': 0,
			'missing': -999,
			'silent' : 1,
			'gamma' : 0.05,
			'subsample' : 0.5,
			'alpha' : 0.05,
			'max_depth':10,
			'min_child_weight':1,

			#'colsample_bytree':0.6,	
			#'colsample_bylevel':1
			#'max_delta_step':0.8
            }
num_rounds=1000
#clf=xgb.train(params,dtrain,num_rounds,watchlist)


#result
dtrain=xgb.DMatrix(x_train1,y_train1)
dtesta=xgb.DMatrix(x_test1a)
dtestb=xgb.DMatrix(x_test1b)

watchlist=([dtrain,'train'],[dtrain,'test'])

clf_all=xgb.train(params,dtrain,num_rounds,watchlist)

y_a_pred = pd.Series(clf_all.predict(dtesta),index=x_test1a.index)
y_b_pred = pd.Series(clf_all.predict(dtestb),index=x_test1b.index)

result_a = pd.DataFrame(y_a_pred,columns=['Y'])
result_b = pd.DataFrame(y_b_pred,columns=['Y'])

way_result='/home/m/桌面/天池智能制造比赛/result/'
result_a.to_csv(way_result+'a.csv')
result_b.to_csv(way_result+'b.csv')


















