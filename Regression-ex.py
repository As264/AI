#匯入程式庫
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

#取得數據
data = datasets.load_diabetes()

#取得特徵值
data_X = data.data[:, np.newaxis, 2]

#切割特徵值做訓練
data_X_train = data_X[:-20]
data_X_test = data_X[-20:]

data_Y_train = data.target[:-20]
data_Y_test = data.target[-20:]

#建立模型
regr = linear_model.LinearRegression()
regr.fit(data_X_train, data_Y_train)
print('Coefficients: \n', regr.coef_)

#均方誤差
print("Mean squard error: %.2f"
      % np.mean((regr.predict(data_X_test) - data_Y_test) ** 2))
#顯示方差
print("Variance score: %.2f" %regr.score(data_X_test, data_Y_test))

#繪圖
plt.scatter(data_X_test, data_Y_test, color = 'black')
plt.plot(data_X_test, regr.predict(data_X_test), color = 'blue',
         linewidth = 3)
plt.xticks()
plt.yticks()
plt.show()