import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Path to your data set
pth='/mnt/c/Users/markb/Documents/Kaggle/Titanic/train.csv'

#Load csv to Pandas Dataframe
df =pd.read_csv(pth)

#print the size of the data frame
print(df.shape)
