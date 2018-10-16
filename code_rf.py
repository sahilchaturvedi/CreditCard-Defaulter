import pandas as pd
import numpy as np
from sklearn import preprocessing,cross_validation,svm,neighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import csv
from fancyimpute import KNN , IterativeImputer 
from MICE import MiceImputer
# from fancyimpute import MICE as MICE
df_train = pd.read_csv("../Training_dataset_Original.csv")

print(df_train.shape)

# def remove_outlier(df_in, col_name):
#     q1 = df_in[col_name].quantile(0.001)
#     q3 = df_in[col_name].quantile(0.99)
#     iqr = q3-q1
#     fence_low  = q1-1.5*iqr
#     fence_high = q3+1.5*iqr
#     df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
#     return df_out

# keys = df_train.keys()
# for key in keys:
# 	if("mvar" in key and key not in ['mvar11','mvar41','mvar40','mvar31','mvar23','mvar30', 'mvar22', 'mvar21','mvar14','mvar4','mvar5','mvar18','mvar35','mvar39','mvar45','mvar46','mvar47']):
# 		df_train = remove_outlier(df_train, key)



df_test = pd.read_csv("../Leaderboard_dataset.csv")
print(df_test.shape)

rid = np.array(df_test['application_key'])
y = np.array(df_train['default_ind'])


df_train = df_train.drop(['default_ind', 'application_key'],1)
df_test = df_test.drop(['application_key'],1)

train_size = len(df_train)
df_train = df_train.append(df_test)



df_train = df_train.drop(['mvar11','mvar41','mvar40','mvar31','mvar23'], axis=1 )
df_train = df_train.drop(['mvar30', 'mvar22', 'mvar21'], axis=1)
# df_train = df_train.drop(['mvar6', 'mvar8', 'mvar12', 'mvar16', 'mvar24'], axis=1)
# df_train = df_train.drop(['mvar14','mvar4','mvar5','mvar18','mvar35','mvar39','mvar45','mvar46'], axis=1)

keys = df_train.keys()
mapping = {'C':0,'L':1}
df_train = df_train.replace({'mvar47':mapping})




df_train.fillna(df_train.mean(),inplace=True)

# for key in df_train.keys():
# 	if("mvar" in key and "mvar47" not in key):
# 		df_train[key] = df_train.groupby("mvar47").transform(lambda x: x.fillna(x.mean()))

scaler = preprocessing.StandardScaler()
df = df_train
print(df_train.describe())
df_train = pd.DataFrame(scaler.fit_transform(df_train))
df_train.columns = df.columns
df_train.index = df.index

pca = PCA(n_components=25)
principalComponents = pca.fit_transform(df_train)
principalDf = pd.DataFrame(data = principalComponents)
df_train = principalDf







df_test = df_train[train_size:]
df_train = df_train[:train_size]


df_train["default_ind"] = y
df_test["application_key"] = rid

df_test.to_csv("../test_final1.csv", index=False);
df_train.to_csv("../train_final1.csv", index=False);
