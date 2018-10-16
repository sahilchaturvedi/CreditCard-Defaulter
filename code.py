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
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import classification_report
import os

duration = 1  # second
freq = 440  # Hz

df = pd.read_csv("/home/sahil/Desktop/amex/train_preprocessed.csv")
print(df.describe())
print(df.head())

# treat categorical dataset 
#mapping = {'C':0,'L':1}
#df = df.replace({'mvar47':mapping})


# treat missing values
df.fillna(df.mean(),inplace=True)



# training data
X = np.array(df.drop(['default_ind', 'application_key'],1))
y = np.array(df['default_ind'])

# preprocessing train data

# scandard scale
X = preprocessing.scale(X)


# avoid overfitting

# tuning didnt worked
# 1. dimentionality reduction
pca = PCA(n_components=40)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# fig = plt.figure()
# plt.plot(var1)
# fig.savefig('temp.png', dpi=fig.dpi)

X = np.array(principalDf)
X = preprocessing.scale(X)

	
# 2. sgd/mini batch GD but this is already their in sklearn 



# test data
df = pd.read_csv("/home/sahil/Desktop/amex/test_preprocessed.csv")
rid = np.array(df['application_key'])
#mapping = {'C':0,'L':1}
#df = df.replace({'mvar47':mapping})
df.fillna(df.mean(),inplace=True)


X_test = np.array(df.drop(['application_key'],1))
X_test = preprocessing.scale(X_test)#

# if pca applied
X_test = np.array(pd.DataFrame(data = pca.transform(X_test)))
X_test = preprocessing.scale(X_test)#





# different classifiers tested for the price pred dataset 


clf1 = svm.SVC()
clf2 = ske.RandomForestClassifier(n_estimators=100)
clf3 = neighbors.KNeighborsClassifier(n_neighbors=5)
clf4 = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(150, 100, 50, 10, 3))
clf5 = DecisionTreeClassifier(criterion = "entropy", random_state = 10,max_depth=100, min_samples_leaf=5)
clf6 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100), n_estimators=600, learning_rate=1)



clfs = [clf2 ]
# train model

i = 1
for clf in clfs:
	epochs = 10
	for epoch in range(epochs):
		X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
		print("Training on clf"+ str(i))
		clf.fit(X_train,y_train)
		score = clf.predict(X_valid)
		print("Report for epoch : " + str(epoch))
		report = classification_report(y_valid, score)
		print(report)
		os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

	score = clf.predict(X_test)
	probab_score = clf.predict_proba(X_test)
	probab_score = [x[0] for x in probab_score]
	temp_l = [ x for x in zip(rid,score,probab_score)]
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

	fp = open("sub.data", "w+")

	for x in temp_l:
		fp.write(str(x) + "\n")
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
	fd = open("You-know-who_IITGuwahati_"+str(i)+".csv" , "w+")
	csvwriter = csv.writer(fd)

	for t in final:
		toWrite.append([str(t[0]), str(t[1])])
	csvwriter.writerows(toWrite)	
	os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (5, freq))


# np.savetxt("sub.csv",final,delimiter=",")
