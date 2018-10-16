import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.neural_network import MLPClassifier
import sklearn.ensemble as ske

from sklearn import preprocessing,cross_validation,svm,neighbors

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score  , KFold
import os
import time
import csv




X_train = pd.read_csv("../train_final1.csv")
X_test = pd.read_csv("../test_final1.csv")

y = X_train["default_ind"]
rid = X_test["application_key"]



# X_train = X_train.drop(["default_ind"], axis=1)
# X_test = X_test.drop(["application_key"], axis=1)

keys = X_test.keys()
# size = len(X_train)
# X_train = X_train.append(X_test)


# # X_train = X_train.drop(['mvar11','mvar40','mvar31','mvar41','mvar45' ], axis=1)

# print(X_train.index)

# scaler = preprocessing.StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train))

# X_test = X_train[size:]
# X_train = X_train[:size]

X_test = np.array(X_test)
X_train = np.array(X_train)



# clf1 = svm.SVC()
clf2 = ske.RandomForestClassifier(n_estimators=500,max_depth=35)
# clf3 = neighbors.KNeighborsClassifier(n_neighbors=5)
clf4 = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(600, 400, 50))
clf5 = DecisionTreeClassifier(criterion = "entropy", random_state = 10,max_depth=100, min_samples_leaf=5)
clf6 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100), n_estimators=600, learning_rate=1)
clf7 = xgb.XGBClassifier(n_estimators=400, max_depth=50, learning_rate=0.1, subsample=0.5)

# print(rf_random.best_params_)
# print(rf_random.best_score_)


clfs = [(clf7, "RandomForestClassifier")]
# train modl
preds = []
preds_proba = []
final_x_test = X_test
X = X_train
for clf, name in clfs:
	# print(name)
	i = 1
	j= 42
	skf = KFold(n_splits=5, random_state=42)
	for train_index, test_index in skf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = y[train_index], y[test_index]
		# clf = ske.RandomForestClassifier(n_estimators=500,max_depth=65,random_state=j)
		clf.fit(X_train, Y_train)
		print("Report for epoch : " + str(i))
		i = i+1
		j = j + 10
		print(classification_report(clf.predict(X_test), Y_test))
		preds.append(clf.predict(final_x_test))
		preds_proba.append(clf.predict_proba(final_x_test))



	print("\n")
	# print([preds[i] for i in range(5)])
	preds = (preds[0] + preds[1] + preds[2] + preds[3] + preds[4])
	preds_proba = (preds_proba[0] + preds_proba[1] + preds_proba[2] + preds_proba[3] + preds_proba[4])/5
	# print(rf_random.best_params_)
	# print(rf_random.best_score_)
	print([ x for x in zip(keys,clf.feature_importances_)])
	
	# clf = ske.RandomForestClassifier(n_estimators=450,max_depth=60,random_state=42)
	# clf.fit(X,y)
	# preds = clf.predict(final_x_test)
	# preds_proba = clf.predict_proba(final_x_test)
	score = [int(k > 2) for k in preds]
	probab_score = [k[0] for k in preds_proba]

	print([score[i] for i  in range(10)])
	print([probab_score[i] for i  in range(10)])
	# 	score = clf.predict(X_valid)
	# 	print("Report for epoch : " + str(epoch))
	# 	report = classification_report(y_valid, score)
	# 	print(report)

	t = time.time()
	temp_l = [ x for x in zip(rid,score,probab_score)]
	fp = open("sub.data", "w+")
	fp.write(str(temp_l))
	fp.close()

	temp_l = sorted(temp_l, key=itemgetter(2), reverse=True)
	
	# for x in temp_l:
	# 	print(x)

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
	i = i+1
	fd = open("Enigma_"+str(i)+"_"+str(t)+".csv" , "w+")
	csvwriter = csv.writer(fd)

	for t in final:
		toWrite.append([str(t[0]), str(t[1])])
	csvwriter.writerows(toWrite)	


# np.savetxt("sub.csv",final,delimiter=",")
