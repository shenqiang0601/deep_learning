import data_loader
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np

x_train,y_train,x_test,y_test =data_loader.load_data()

#创建评论数据的词库索引
vocalen,word_index = data_loader.createWordIndex(x_train,x_test)
#print(vocalen)

#获取训练数据每个词的索引
x_train_index =data_loader.word2Index(x_train,word_index)
x_test_index=data_loader.word2Index(x_test,word_index)

#最大长度的限制
maxlen =25
x_train_index =sequence.pad_sequences(x_train_index,maxlen=maxlen )
x_test_index =sequence.pad_sequences(x_test_index,maxlen=maxlen)
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)

model =Sequential()
model.add(Embedding(trainable=False, input_dim= vocalen+1, output_dim=300, input_length=maxlen))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation= 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) #二分类问题

print(x_train_index, y_train)
model.fit(x_train_index, y_train,batch_size=512, epochs=200)
score, acc = model.evaluate(x_test_index, y_test)
print('Test score:', score)
print('test accuracy:',acc)

test = np.array([x_test_index[1000]])
print(test)
print(test.shape)

predict = model.predict(test)
print(predict)
print(np.argmax(predict,axis=1))

