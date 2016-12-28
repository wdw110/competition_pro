#encoding=utf-8

import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import time


train_num = 20000
test_num = 30000
data = pd.read_csv('train.csv')
train_data = data.values[0:train_num,1:]
train_label = data.values[0:train_num,0]
test_data = data.values[train_num:test_num,1:]
test_label = data.values[train_num:test_num,0]

for i in range(len(train_data)):
	train_data[i][train_data[i]>0]=np.ones(len(train_data[i][train_data[i]>0]))

for i in range(len(test_data)):
	test_data[i][test_data[i]>0]=np.ones(len(test_data[i][test_data[i]>0]))
	
t = time.time()
pca = PCA(n_components=0.8)
train_x = pca.fit_transform(train_data)
test_x = pca.fit_transform(test_data)
neighbors = KNeighborsClassifier(n_neighbors=4)
neighbors.fit(train_data,train_label)
pre = neighbors.predict(test_data)

acc = float((pre==test_label).sum())/len(test_data)
print u'准确率: %f,花费时间: %.fs' %(acc,time.time()-t)

