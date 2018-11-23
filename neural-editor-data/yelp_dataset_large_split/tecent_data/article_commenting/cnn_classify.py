# coding: utf8
import json
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import random
# import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def cnn(x, y):
    # 全局变量
    batch_size = 100
    nb_classes = 2
    epochs = 100
    # input image dimensions
    img_rows, img_cols = 2, 16
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (1, 1)
    # convolution kernel size
    kernel_size = (2, 4)

    # the data, shuffled and split between train and test sets
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = x[100:]
    y_train = y[100:]
    X_test = x[:100]
    y_test = y[:100]

    # 根据不同的backend定下不同的格式
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
#     X_train /= 255
#     X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # 转换为one_hot类型
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    #构建模型
    model = Sequential()
    """
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape))
    """
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding='same',
                            input_shape=input_shape)) # 卷积层1
    model.add(Activation('relu')) #激活层
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2
    model.add(Activation('relu')) #激活层
    model.add(MaxPooling2D(pool_size=pool_size)) #池化层
    model.add(Dropout(0.25)) #神经元随机失活
    model.add(Flatten()) #拉成一维数据
    model.add(Dense(128)) #全连接层1
    model.add(Activation('relu')) #激活层
    model.add(Dropout(0.5)) #随机失活
    model.add(Dense(nb_classes)) #全连接层2
    model.add(Activation('softmax')) #Softmax评分

    #编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    #训练模型
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_test, Y_test))
    #评估模型
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    #存储模型
    model.save('cnn_model1.h5')

def cnn_predict(x):
    # 全局变量
    batch_size = 100
    nb_classes = 2
    epochs = 100
    # input image dimensions
    img_rows, img_cols = 2, 16
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (1, 1)
    # convolution kernel size
    kernel_size = (2, 4)

    # the data, shuffled and split between train and test sets
    X_test = x

    # 根据不同的backend定下不同的格式
    if K.image_dim_ordering() == 'th':
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_test = X_test.astype('float32')
    print(X_test.shape[0], 'test samples')

    #加载模型
    model = load_model('cnn_model.h5')
    #评估模型
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    labels_pre = model.predict(X_test)

    labels_pre2 = [[np.argmax(t)] for t in labels_pre]
    labels_pre2 = []
    for i in range(len(labels_pre)):
        if labels_pre[i][0] > 0.55:
            labels_pre2.append([0])
        else:
            labels_pre2.append([1])

    return np.array(labels_pre)

def lgb_test(test_data):
    nst = lgb.Booster(model_file="model1.txt")
    preds = nst.predict(test_data, num_iteration=nst.best_iteration) # 输出的是概率结果
    real_result = []
    for pred in preds:
        result = prediction = int(np.argmax(pred))
        real_result.append(result)
    return real_result

if __name__ == '__main__':
    with open('year1516_arr.json') as f:
        data = json.load(f)

    df1_sub, df2_sub, labels = data
    print(np.array(df1_sub).shape, np.array(df2_sub).shape, np.array(labels).shape)

    data2 = []
    for i in range(np.array(df1_sub).shape[0]):
        data2.append([df1_sub[i], df2_sub[i], labels[i]])
    random.shuffle(data2)
    print(len(data2))

    df1_sub_sh, df2_sub_sh, labels_sh = [], [], []
    for i in range(len(data2)):
        df1_sub_sh.append(data2[i][0])
        df2_sub_sh.append(data2[i][1])
        labels_sh.append(data2[i][2])
    print(np.array(df1_sub_sh).shape, np.array(df2_sub_sh).shape, np.array(labels_sh).shape)

    x_train_ls = []
    for i in range(len(df1_sub_sh)):
        x_train_ls.append([df1_sub_sh[i], df2_sub_sh[i]])
    x_train = np.array(x_train_ls)
    labels = np.array(labels_sh)
    print(x_train.shape, labels.shape)
    cnn(x_train, labels)

    # labels1 = cnn_predict(x_train)
    # print(labels1)
    

    # df1_sub_ls1 = df1_sub_sh[:20000]
    # df2_sub_ls1 = df2_sub_sh[:20000]
    # labels1 = labels[:20000]
    # x_train_ls = []
    # for i in range(len(df1_sub_ls1)):
    #     x_train_ls.append([df1_sub_ls1[i], df2_sub_ls1[i]])
    # x_train = np.array(x_train_ls)
    # labels1_ = cnn_predict(x_train)
    # count_cor = 0
    # for i in range(len(labels1_)):
    #     if labels1_[i][0] == labels1[i][0]:
    #         count_cor += 1
    # print(labels1_)
    # print('acc: ', (count_cor/len(labels1)))
    # 