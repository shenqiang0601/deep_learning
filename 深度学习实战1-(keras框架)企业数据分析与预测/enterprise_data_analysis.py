import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import matplotlib.pylab as plt   # 绘制图像库

def GM11(x0): #自定义灰色预测函数
  import numpy as np
  x1 = x0.cumsum() #1-AGO序列
  z1 = (x1[:len(x1)-1] + x1[1:])/2.0
  z1 = z1.reshape((len(z1),1))
  B = np.append(-z1, np.ones_like(z1), axis = 1)
  Yn = x0[1:].reshape((len(x0)-1, 1))
  [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数
  f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #还原值
  delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))
  C = delta.std()/x0.std()
  P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)
  return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率

data = pd.read_csv('data.csv') #读取数据
data.index = range(2000,2020) # 标注索引信息年份

data.loc[2020] = None
data.loc[2021] = None
data.loc[2022] = None
l = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7']
l1 = ['x3', 'x5', 'x7']
for i in l1:
  f, _, _, _, C, _ = GM11(data[i].loc[range(2000, 2020)].values)
  print("%s后验差比值：%0.4f" % (i, C))  # 后验差比值c，即：真实误差的方差同原始数据方差的比值。
  data[i].loc[2020] = f(len(data) - 2)  # 2014年预测结果
  data[i].loc[2021] = f(len(data) - 1)  # 2015年预测结果
  data[i].loc[2022] = f(len(data))  # 2016年预测结果
  data[i] = data[i].round(2)  # 保留两位小数

data[l1 + ['y']].to_csv('GM11.csv')  # 结果输出

data = pd.read_csv('GM11.csv',index_col = 0) #读取数据
feature = ['x3','x5','x7']  # 提取特征

data_train = data.loc[range(2000, 2020)]  # 取2014年前的数据建模
print(data_train)
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean) / data_std  # 数据标准化 后进行训练

x_train = data_train[feature].values  # 特征数据
y_train = data_train['y'].values  # 标签数据

model = Sequential() #建立模型
model.add(Dense(12,activation='relu',input_dim=3))
model.add(Dense(24,activation='relu'))  # 隐藏层
model.add(Dense(1))  # 输出层
model.compile(loss='mean_squared_error', optimizer='adam') #编译模型
model.fit(x_train, y_train, epochs = 10000, batch_size = 16,verbose=2) #训练模型，训练1000次
model.save('net.h5') #保存模型参数

x = ((data[feature] - data_mean[feature])/data_std[feature]).values
data[u'y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
data.to_csv('result.csv')

p = pd.read_csv('result.csv')
p = p[['y','y_pred']].copy()
p.index=range(2000,2023)
p.plot(style=['b-o','r-*'],xticks=p.index,figsize=(15,5))
plt.xlabel("Year")
plt.show()