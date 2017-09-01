#!/usr/bin/env python

from sklearn.svm import LinearSVC
import csv
import pandas as pd

# read in and set up training data
with open('train.csv', 'r') as f:
  reader = csv.reader(f)
  train_list = list(reader)

train_vec = [[0,0,0] for k in range(0,len(train_list)-1)]
for k in range(1,len(train_list)):
	train_vec[k-1][0] = int(train_list[k][2]) #Pclass
	if train_list[k][4] == 'male': #sex
		train_vec[k-1][1] = 1
	if train_list[k][5]: #age if known
		train_vec[k-1][2] = float(train_list[k][5])

labels = [int(person[1]) for person in train_list[1:]]

# Perform SVM fit (on class, sex, and age)
clf = LinearSVC(random_state=0)
clf.fit(train_vec, labels)

# Read in and set up test data
with open('test.csv', 'r') as f:
  reader = csv.reader(f)
  test_list = list(reader)

test_vec = [[0,0,0] for k in range(0,len(test_list)-1)]
predict = [0 for k in range(0,len(test_vec))]

for k in range(1,len(test_vec)):
	test_vec[k-1][0] = int(test_list[k][1]) #Pclass
	if test_list[k][3] == 'male':
		test_vec[k-1][1] = 1
	if test_list[k][4]:
		test_vec[k-1][2] = float(test_list[k][4])

# Predict survival in test set
predict = [clf.predict(test_vec)]

# output to csv
test = pd.read_csv("test.csv")
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predict[0]})
submission.to_csv("submission.csv", index=False)

