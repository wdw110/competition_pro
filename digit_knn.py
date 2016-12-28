#encoding=utf-8

#读取表头信息
import csv
import sklearn.cross_validation as cross_validation
from sklearn import neighbors
from sklearn import neighbors
import sklearn.metrics as metrics

csvfile = open('train.csv', 'r')
reader=csv.reader(csvfile)
headers = reader.next()
print headers

#读取特征信息和结果信息
featureList=[]
labelList=[]
for row in reader:
    labelList.append(row[0])
    featureList.append(row[1:])

#将原始信息按9：1分割为训练集与测试集
train_data, test_data, train_target, test_target = cross_validation.train_test_split(featureList,labelList, test_size=0.1, random_state=0)

#输入默认模型
knn=neighbors.KNeighborsClassifier()
#训练模型
knn.fit(train_data,train_target)
#预测测试集
predict_test=knn.predict(test_data)
#现实预测结果
print metrics.classification_report(test_target, predict_test)
