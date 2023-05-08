import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pymysql
import datetime
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
data =pd.read_csv("weather.csv",encoding='gb18030')
data = data.drop(columns=data.columns[-1])
print(data)

pca = PCA(n_components=2)
new_pca = pd.DataFrame(pca.fit_transform(data))
X = new_pca.values
print(new_pca)

kms = KMeans(n_clusters=6)  # 6表示聚类的个数
#获取类别标签
Y= kms.fit_predict(data)
data['class'] = Y
data.to_csv("weather_new.csv",index=False) #保存文件

#绘制聚类发布图
d = new_pca[Y == 0]
plt.plot(d[0], d[1], 'r.')
d = new_pca[Y == 1]
plt.plot(d[0], d[1], 'g.')
d = new_pca[Y == 2]
plt.plot(d[0], d[1], 'b.')
d = new_pca[Y == 3]
plt.plot(d[0], d[1], 'y.')
d = new_pca[Y == 4]
plt.plot(d[0], d[1],'c.')
d = new_pca[Y == 5]
plt.plot(d[0], d[1],'k.')
#plt.show()


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=1),
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            # 线性分类器
            nn.Linear(128 * 6 * 1, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            # nn.Softmax(dim=1),
        )
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(params=self.parameters(), lr=1e-3)
        self.start = datetime.datetime.now()

    def forward(self, inputs):
        out = self.con1(inputs)
        out = self.con2(out)
        out = out.view(out.size(0), -1)  # 展开成一维
        out = self.fc(out)
        return out

    def train(self, x, y):
        out = self.forward(x)
        loss = self.mls(out, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss

    def test(self, x):
        out = self.forward(x)
        return out

    def get_data(self):
        with open('weather_new.csv', 'r') as f:
            results = csv.reader(f)
            results = [row for row in results]
            results = results[1:1500]
        inputs = []
        labels = []
        for result in results:
            # one-hot独热编码
            one_hot = [0 for i in range(6)]
            index = int(result[6]) - 1
            one_hot[index] = 1
            labels.append(one_hot)
            input = result[:6]
            input = [float(x) for x in input]

            inputs.append(input)

        inputs = np.array(inputs)
        labels = np.array(labels)
        inputs = torch.from_numpy(inputs).float()
        inputs = torch.unsqueeze(inputs, 1)

        labels = torch.from_numpy(labels).float()
        return inputs, labels

    def get_test_data(self):
        with open('weather_new.csv', 'r') as f:
            results = csv.reader(f)
            results = [row for row in results]
            results = results[1500: 1817]
        inputs = []
        labels = []
        for result in results:
            label = [result[6]]
            input = result[:6]
            input = [float(x) for x in input]
            label = [float(y) for y in label]
            inputs.append(input)
            labels.append(label)
        inputs = np.array(inputs)

        inputs = torch.from_numpy(inputs).float()
        inputs = torch.unsqueeze(inputs, 1)
        labels = np.array(labels)
        labels = torch.from_numpy(labels).float()
        return inputs, labels

if __name__ == '__main__':
    EPOCH = 100
    BATCH_SIZE = 50

    net = MyNet()
    x_data, y_data = net.get_data()
    torch_dataset = Data.TensorDataset(x_data, y_data)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            # print(step)
            # print(step,'batch_x={};  batch_y={}'.format(batch_x, batch_y))
            a = net.train(batch_x, batch_y)
            print('step:',step,a)
    # 保存模型
    torch.save(net, 'net.pkl')

    # 加载模型
    net = torch.load('net.pkl')
    x_data, y_data = net.get_test_data()
    torch_dataset = Data.TensorDataset(x_data, y_data)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=1,
    )
    num_success = 0
    num_sum = 317
    for step, (batch_x, batch_y) in enumerate(loader):
        # print(step)
        output = net.test(batch_x)
        # output = output.detach().numpy()
        y = batch_y.detach().numpy()
        for index, i in enumerate(output):
            i = i.detach().numpy()
            i = i.tolist()
            j = i.index(max(i))
            print('输出为{}标签为{}'.format(j+1, y[index][0]))
            loss = j+1-y[index][0]
            if loss == 0.0:
                num_success += 1
    print('正确率为{}'.format(num_success/num_sum))
