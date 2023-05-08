import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import matplotlib .pyplot as plt
import pandas as pd
import csv

data = pd.read_csv('train_new.csv',encoding = 'utf-8')

# 提取数据特性x1,x2,x3,x4,作为训练集
train = data[['x1', 'x2', 'x3', 'x4']]

# 设置标签值  one-hot编码
y_train = np.zeros((len(data), 5), dtype=np.int)
for i in range(len(data)):
    y_train[i][data['class'][i]] = 1
print(np.array(y_train))

model=Sequential()
model.add(Dense(input_dim=4,units=666,activation='relu'))
model.add(Dropout(0.5))  # Dropout(0.5) 表示随机丢弃50%的神经元，防止过拟合
model.add(Dense(units=666,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=666,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=5,activation='softmax')) #输出层 输出5个等级结果

model.compile(loss='mse',optimizer='adam',metrics=['acc'])
history = model.fit(train,y_train,batch_size=123,epochs=500,validation_split=0.2) #训练500次

weights = np.array(model.get_weights())
result2 = model.evaluate(train, y_train)

# 绘制图形函数
def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(history,'acc','val_acc')

show_train_history(history,'loss','val_loss')

