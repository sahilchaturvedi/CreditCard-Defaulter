import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn import preprocessing,cross_validation,svm,neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import sklearn.ensemble as ske
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
import matplotlib.pyplot as plt
import xgboost as xgb
import csv
from sklearn.model_selection import StratifiedKFold, cross_val_score  , KFold
import os
import numpy as np
import pandas as pd
import tensorflow
import keras
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew


X_train = pd.read_csv("../train_final.csv")
X_test = pd.read_csv("../test_final.csv")

y = X_train["default_ind"]
rid = X_test["application_key"]



X_train = X_train.drop(["default_ind"], axis=1)
X_test = X_test.drop(["application_key"], axis=1)

size = len(X_train)
X_train = X_train.append(X_test)


X_train = X_train.drop(['mvar11','mvar40','mvar31','mvar41','mvar45' ], axis=1)
keys = X_train.keys()

print(X_train.index)

scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))

X_test = X_train[size:]
X_train = X_train[:size]

X_test = np.array(X_test)
X_train = np.array(X_train)

x_train = X_train
y_train = y
x_test = X_test

model = Sequential()
model.add(Dense(350, input_dim=len(keys), kernel_initializer='normal', activation='relu'))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
# model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
modeldata = model.fit(x_train, y_train, epochs=50, batch_size=100)
model.save("model_4.h5")


probab_score =  model.predict(x_test)
score = [int(k > 0.5) for k in probab_score]

print([score[i] for i  in range(10)])
print([probab_score[i] for i  in range(10)])
t = time.time()
temp_l = [ x for x in zip(rid,score,probab_score)]
temp_l = sorted(temp_l, key=itemgetter(2), reverse=True)
temp_l_0 = []
temp_l_1 = []
temp_l_0_rest = []
temp_l_1_rest = []

temp_l_0_id = []
temp_l_1_id = []
temp_l_0_rest_id = []
temp_l_1_rest_id = []


for x in temp_l:
	if(x[1]==0 and x[2] >= 0.60):
		temp_l_0_id.append(x[0])
		temp_l_0.append(x[1])
	elif(x[1]==0 and x[2] < 0.60):
		temp_l_0_rest_id.append(x[0])
		temp_l_0_rest.append(x[1])
	
	elif(x[1]==1 and x[2] <= 0.1):
		temp_l_1_id.append(x[0])
		temp_l_1.append(x[1])
	else:
		temp_l_1_rest_id.append(x[0])
		temp_l_1_rest.append(x[1])

rid = temp_l_0_id + temp_l_1_id + temp_l_0_rest_id +  temp_l_1_rest_id
score = temp_l_0 + temp_l_1 + temp_l_0_rest +  temp_l_1_rest

final = np.column_stack((rid,score))
toWrite = []
fd = open("NN_"+str(t)+".csv" , "w+")
csvwriter = csv.writer(fd)

for t in final:
	toWrite.append([str(t[0]), str(t[1])])
csvwriter.writerows(toWrite)	
