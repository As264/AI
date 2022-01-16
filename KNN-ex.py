#匯入程式庫
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#取得數據
iris = datasets.load_iris()

#切割訓練集和測試集
iris_X_train, iris_X_test, iris_Y_train, iris_Y_test = train_test_split(
    iris.data, iris.target, test_size = 0.2)

#建立模型
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_Y_train)

#顯示結果
print("預測", knn.predict(iris_X_test))
print("實際", iris_Y_test)
print('準確率: %.2f' % knn.score(iris_X_test, iris_Y_test))