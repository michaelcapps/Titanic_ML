#!/usr/bin/env python

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# read in and set up training data
train = pd.read_csv("train.csv")

train["Sex"].replace('female',0,inplace=True)
train["Sex"].replace('male',1,inplace=True)
train.fillna(-1,inplace=True)
train_vec = train.loc[:,["Pclass","Sex","Age"]].values

labels = train["Survived"].values

# Perform SVM fit (on class, sex, and age)
clf = RandomForestClassifier(max_depth=2, random_state=0)
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

