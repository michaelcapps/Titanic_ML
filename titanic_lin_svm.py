#!/usr/bin/env python

from sklearn.svm import LinearSVC
import csv
import pandas as pd


# read in and set up training data
# with open('train.csv', 'r') as f:
#   reader = csv.reader(f)
#   train_list = list(reader)

# train_vec = [[0,0,0] for k in range(0,len(train_list)-1)]
# for k in range(1,len(train_list)):
# 	train_vec[k-1][0] = int(train_list[k][2]) #Pclass
# 	if train_list[k][4] == 'male': #sex
# 		train_vec[k-1][1] = 1
# 	if train_list[k][5]: #age if known
# 		train_vec[k-1][2] = float(train_list[k][5])

train = pd.read_csv("train.csv")

train["Sex"].replace('female',0,inplace=True)
train["Sex"].replace('male',1,inplace=True)
train.fillna(-1,inplace=True)
train_vec = train.loc[:,["Pclass","Sex","Age"]].values

labels = train["Survived"].values

print(train_vec)
#labels = [int(person[1]) for person in train_list[1:]]

# Perform SVM fit (on class, sex, and age)
clf = LinearSVC(random_state=0)
clf.fit(train_vec, labels)

# Read in and set up test data
test = pd.read_csv("test.csv")

test["Sex"].replace('female',0,inplace=True)
test["Sex"].replace('male',1,inplace=True)
test.fillna(-1,inplace=True)
test_vec = test.loc[:,["Pclass","Sex","Age"]].values

# Predict survival in test set
predict = [clf.predict(test_vec)]

# output to csv
test = pd.read_csv("test.csv")
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predict[0]})
submission.to_csv("submission.csv", index=False)

