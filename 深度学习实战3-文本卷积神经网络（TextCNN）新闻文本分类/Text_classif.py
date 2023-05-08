import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpu0], "GPU")

# 导入库包
import tensorflow.keras as keras
from config import Config
import os
from sklearn import metrics
import numpy as np
from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, Conv1D, ReLU, GlobalMaxPool1D, InputLayer

trainingSet_path = "cnews.train.txt"
valSet_path = "cnews.val.txt"
model_save_path = "CNN_model.h5"
testingSet_path = "cnews.test.txt"

# 创建 文本处理类：preprocesser
class preprocesser(object):
    def __init__(self):
        self.config = Config()

    # 读取文本txt 函数
    def read_txt(self, txt_path):
        with open(txt_path, "r", encoding='utf-8') as f:
            data = f.readlines()
        labels = []
        contents = []
        for line in data:
            label, content = line.strip().split('\t')
            labels.append(label)
            contents.append(content)
        return labels, contents

    # 读取分词文档
    def get_vocab_id(self):

        vocab_path = "vocab.txt"
        with open(vocab_path, "r", encoding="utf-8") as f:
            infile = f.readlines()
        vocabs = list([word.replace("\n", "") for word in infile])
        vocabs_dict = dict(zip(vocabs, range(len(vocabs))))
        return vocabs, vocabs_dict

    # 获取新闻属性id 函数
    def get_category_id(self):

        categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
        cates_dict = dict(zip(categories, range(len(categories))))
        return cates_dict

    # 将语料中各文本转换成固定max_length后返回各文本的标签与文本tokens
    def word2idx(self, txt_path, max_length):

        # vocabs:分词词汇表
        # vocabs_dict:各分词的索引
        vocabs, vocabs_dict = self.get_vocab_id()
        # cates_dict:各分类的索引
        cates_dict = self.get_category_id()

        # 读取语料
        labels, contents = self.read_txt(txt_path)
        # labels_idx：用来存放语料中的分类
        labels_idx = []
        # contents_idx:用来存放语料中各样本的索引
        contents_idx = []

        # 遍历语料
        for idx in range(len(contents)):
            # tmp:存放当前语句index
            tmp = []
            # 将该idx(样本)的标签加入至labels_idx中
            labels_idx.append(cates_dict[labels[idx]])
            # contents[idx]:为该语料中的样本遍历项
            # 遍历contents中各词并将其转换为索引后加入contents_idx中
            for word in contents[idx]:
                if word in vocabs:
                    tmp.append(vocabs_dict[word])
                else:
                    # 第5000位设置为未知字符
                    tmp.append(5000)
            # 将该样本index后结果存入contents_idx作为结果等待传回
            contents_idx.append(tmp)

        # 将各样本长度pad至max_length
        x_pad = keras.preprocessing.sequence.pad_sequences(contents_idx, max_length)
        y_pad = keras.utils.to_categorical(labels_idx, num_classes=len(cates_dict))

        return x_pad, y_pad

    def word2idx_for_sample(self, sentence, max_length):
        # vocabs:分词词汇表
        # vocabs_dict:各分词的索引
        vocabs, vocabs_dict = self.get_vocab_id()
        result = []
        # 遍历语料
        for word in sentence:
            # tmp:存放当前语句index
            if word in vocabs:
                result.append(vocabs_dict[word])
            else:
                # 第5000位设置为未知字符，实际中为vocabs_dict[5000]，使得vocabs_dict长度变成len(vocabs_dict+1)
                result.append(5000)

        x_pad = keras.preprocessing.sequence.pad_sequences([result], max_length)
        return x_pad

pre = preprocesser()  # 实例化preprocesser()类

num_classes = 10  # 类别数
vocab_size = 5000  # 语料词大小
seq_length = 600  # 词长度

conv1_num_filters = 128  # 第一层输入卷积维数
conv1_kernel_size = 1  # 卷积核数

conv2_num_filters = 64  # 第二层输入卷维数
conv2_kernel_size = 1  # 卷积核数

hidden_dim = 128  # 隐藏层维度
dropout_keep_prob = 0.5  # dropout层丢弃0.5

batch_size = 64  # 每次训练批次数


def TextCNN():
    # 创建模型序列
    model = Sequential()
    model.add(InputLayer((seq_length,)))
    model.add(Embedding(vocab_size + 1, 256, input_length=seq_length))
    model.add(Conv1D(conv1_num_filters, conv1_kernel_size, padding="SAME"))
    model.add(Conv1D(conv2_num_filters, conv2_kernel_size, padding="SAME"))
    model.add(GlobalMaxPool1D())
    model.add(Dense(hidden_dim))
    model.add(Dropout(dropout_keep_prob))
    model.add(ReLU())
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["acc"])
    print(model.summary())

    return model


def train(epochs):
    model = TextCNN()
    model.summary()

    x_train, y_train = pre.word2idx(trainingSet_path, max_length=seq_length)
    x_val, y_val = pre.word2idx(valSet_path, max_length=seq_length)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

    model.save(model_save_path, overwrite=True)


def test():
    if os.path.exists(model_save_path):
        model = keras.models.load_model(model_save_path)
        print("-----model loaded-----")
        model.summary()

    x_test, y_test = pre.word2idx(testingSet_path, max_length=seq_length)
    print(x_test.shape)
    print(type(x_test))
    print(y_test.shape)
    # print(type(y_test))
    pre_test = model.predict(x_test)
    # print(pre_test.shape)
    # metrics.classification_report(np.argmax(pre_test, axis=1), np.argmax(y_test, axis=1), digits=4, output_dict=True)
    print(metrics.classification_report(np.argmax(pre_test, axis=1), np.argmax(y_test, axis=1)))


if __name__ == '__main__':
    train(20)

    model = keras.models.load_model(model_save_path)
    print("-----model loaded-----")
    model.summary()
    test = preprocesser()

    # 测试文本
    x_test = '5月6日，上海莘庄基地田径特许赛在第二体育运动学校鸣枪开赛。男子110米栏决赛，19岁崇明小囡秦伟搏以13.35秒的成绩夺冠，创造本赛季亚洲最佳。谢文骏迎来赛季首秀，以13.38秒获得亚军'
    x_test = test.word2idx_for_sample(x_test, 600)

    categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]

    pre_test = model.predict(x_test)

    index = int(np.argmax(pre_test, axis=1)[0])

    print('该新闻为:', categories[index])
