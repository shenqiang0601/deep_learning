import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

#导入字体
DroidSansFallbackFull = ImageFont.truetype("DroidSansFallback.ttf", 36, 0)
fonts = [DroidSansFallbackFull,]


# 生成图片，48*48大小
def affineTrans(image, mode, size=(48, 48)):
    # print("AffineTrans ...")
    if mode == 0:  # padding移动
        which = np.array([0, 0, 0, 0])
        which[np.random.randint(0, 4)] = np.random.randint(0, 10)
        which[np.random.randint(0, 4)] = np.random.randint(0, 10)
        image = cv2.copyMakeBorder(image, which[0], which[0], which[0], which[0], cv2.BORDER_CONSTANT, value=0)
        image = cv2.resize(image, size)
    if mode == 1:
        scale = np.random.randint(48, int(48 * 1.4))
        center = [scale / 2, scale / 2]
        image = cv2.resize(image, (scale, scale))
        image = image[int(center[0] - 24):int(center[0] + 24), int(center[1] - 24):int(center[1] + 24)]

    return image


# 图片处理 除噪
def noise(image, mode=1):
    # print("Noise ...")
    noise_image = (image.astype(float) / 255) + (np.random.random((48, 48)) * (np.random.random() * 0.3))
    norm = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())
    if mode == 1:
        norm = (norm * 255).astype(np.uint8)
    return norm


# 绘制中文的图片
def DrawChinese(txt, font):
    # print("DrawChinese...")
    image = np.zeros(shape=(48, 48), dtype=np.uint8)
    x = Image.fromarray(image)
    draw = ImageDraw.Draw(x)
    draw.text((8, 2), txt, (255), font=font)
    result = np.array(x)
    return result


# 图片标准化
def norm(image):
    # print("norm...")
    return image.astype(np.float) / 255

char_set = open("chinese.txt",encoding = "utf-8").readlines()[0]
print(len(char_set[0]))  # 打印字的个数

# 生成数据：训练集和标签
def Genernate(batchSize, charset):
    # print("Genernate...")
    #    pass
    label = [];
    training_data = [];

    for x in range(batchSize):
        char_id = np.random.randint(0, len(charset))
        font_id = np.random.randint(0, len(fonts))
        y = np.zeros(dtype=np.float, shape=(len(charset)))
        image = DrawChinese(charset[char_id], fonts[font_id])
        image = affineTrans(image, np.random.randint(0, 2))
        # image = noise(image)
        # image = augmentation(image,np.random.randint(0,8))
        image = noise(image)
        image_norm = norm(image)
        image_norm = np.expand_dims(image_norm, 2)

        training_data.append(image_norm)
        y[char_id] = 1
        label.append(y)

    return np.array(training_data), np.array(label)

def Genernator(charset,batchSize):
    print("Generator.....")

    while(1):
        label = [];
        training_data = [];
        for i in range(batchSize):
            char_id = np.random.randint(0, len(charset))
            font_id = np.random.randint(0,len(fonts))
            y = np.zeros(dtype=np.float,shape=(len(charset)))
            image = DrawChinese(charset[char_id],fonts[font_id])
            image = affineTrans(image,np.random.randint(0,2))
            #image = noise(image)
            #image = augmentation(image,np.random.randint(0,8))
            image = noise(image)
            image_norm = norm(image)
            image_norm  = np.expand_dims(image_norm,2)
            y[char_id] = 1
            training_data.append(image_norm)
            label.append(y)

        y[char_id] = 1
        yield (np.array(training_data),np.array(label))

def Getmodel(nb_classes):

    img_rows, img_cols = 48, 48
    nb_filters = 32
    nb_pool = 2
    nb_conv = 4

    model = Sequential()
    print("sequential..")
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            padding='same',
                            input_shape=(img_rows, img_cols, 1)))
    print("add convolution2D...")
    model.add(Activation('relu'))
    print("activation ...")
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

#评估模型
def eval(model, X, Y):
    print("Eval ...")
    res = model.predict(X)

#训练函数
def Training(charset):
    model = Getmodel(len(charset))
    while (1):
        X, Y = Genernate(64, charset)
        model.train_on_batch(X, Y)
        print(model.loss)

#训练生成模型
def TrainingGenerator(charset, test=1):

    set = Genernate(64, char_set)
    model = Getmodel(len(charset))
    BatchSize = 64
    model.fit_generator(generator=Genernator(charset, BatchSize), steps_per_epoch=BatchSize * 10, epochs=15,
                        validation_data=set)

    model.save("ocr.h5")

    X = set[0]
    Y = set[1]
    if test == 1:
        print("============6 Test == 1 ")
        for i, one in enumerate(X):
            x = one
            res = model.predict(np.array([x]))
            classes_x = np.argmax(res, axis=1)[0]
            print(classes_x)

            print(u"Predict result：", char_set[classes_x], u"Real result：", char_set[Y[i].argmax()])
            image = (x.squeeze() * 255).astype(np.uint8)
            cv2.imwrite("{0:05d}.png".format(i), image)

TrainingGenerator(char_set)  #函数TrainingGenerator 开始训练


model = tf.keras.models.load_model("ocr.h5")
img1 = cv2.imread('00001.png',0)
img = cv2.resize(img1,(48,48))

print(img.shape)
img2 = tf.expand_dims(img, 0)
res = model.predict(img2)
classes_x = np.argmax(res, axis=1)[0]
print(classes_x)
print(u"Predict result：", char_set[classes_x])


