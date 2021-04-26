import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
sys.path.append("../models")
sys.path.append("../base")
filename = os.path.basename(__file__)
from dbn import DBN
from cnn import CNN
from sup_sae import supervised_sAE
from base_func import run_sess
# from tensorflow.examples.tutorials.mnist import input_data
train_path = './mnist_image_label/mnist_train_jpg_60000/'
train_txt = './mnist_image_label/mnist_train_jpg_60000.txt'
x_train_savepath = './mnist_image_label/mnist_x_train.npy'
y_train_savepath = './mnist_image_label/mnist_y_train.npy'

test_path = './mnist_image_label/mnist_test_jpg_10000/'
test_txt = './mnist_image_label/mnist_test_jpg_10000.txt'
x_test_savepath = './mnist_image_label/mnist_x_test.npy'
y_test_savepath = './mnist_image_label/mnist_y_test.npy'
def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_
# Loading dataset
# Each datapoint is a 8x8 image of a digit.
# mnist = input_data.read_data_sets('../dataset/MNIST_data', one_hot=True)
if __name__ == '__main__':

    if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
            x_test_savepath) and os.path.exists(y_test_savepath):
        print('-------------Load Datasets-----------------')
        x_train_save = np.load(x_train_savepath)
        y_train = np.load(y_train_savepath)
        x_test_save = np.load(x_test_savepath)
        y_test = np.load(y_test_savepath)
        x_train = np.reshape(x_train_save, (len(x_train_save), 32,32))
        x_test = np.reshape(x_test_save, (len(x_test_save), 32,32))
    else:
        print('-------------Generate Datasets-----------------')
        x_train, y_train = generateds(train_path, train_txt)
        x_test, y_test = generateds(test_path, test_txt)

        print('-------------Save Datasets-----------------')
        x_train_save = np.reshape(x_train, (len(x_train), -1))
        x_test_save = np.reshape(x_test, (len(x_test), -1))
        np.save(x_train_savepath, x_train_save)
        np.save(y_train_savepath, y_train)
        np.save(x_test_savepath, x_test_save)
        np.save(y_test_savepath, y_test)
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
# Splitting data
y_train = y_train.reshape(531,1)
x_train = x_train.reshape(531,1024)
x_test = x_test.reshape(161,1024)
y_test = y_test.reshape(161,1)
x_train = np.float64(x_train)
y_train = np.float64(y_train)
x_test = np.float64(x_test)
y_test = np.float64(y_test)
datasets = [x_train,y_train,x_test,y_test]

#X_train, X_test = X_train[::100], X_test[::100]
#Y_train, Y_test = Y_train[::100], Y_test[::100]
x_dim=datasets[0].shape[1]
y_dim=datasets[1].shape[1]
p_dim=int(np.sqrt(x_dim))

print(y_dim)
print(p_dim)
print(datasets[0][0])
print(datasets[1][0])
tf.reset_default_graph()
# # Training
select_case = 1

if select_case==1:
    classifier = DBN(
                 hidden_act_func='sigmoid',
                 output_act_func='softmax',
                 loss_func='cross_entropy', # gauss 激活函数会自动转换为 mse 损失函数
                 struct=[x_dim, 500,300, 100, 5],
                 lr=1e-2,
                 momentum=0.5,
                 use_for='classification',
                 bp_algorithm='rmsp',
                 epochs=10,
                 batch_size=32,
                 dropout=0.12,
                 units_type=['gauss','bin'],
                 rbm_lr=1e-3,
                 rbm_epochs=5,
                 cd_k=1)
if select_case==2:
    classifier = CNN(
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_func='cross_entropy',
                 use_for='classification',
                 lr=1e-3,
                 epochs=20,
                 img_shape=[p_dim,p_dim],
                 channels=[1, 6, 6, 64, y_dim], # 前几维给 ‘Conv’ ，后几维给 ‘Full connect’
                 layer_tp=['C','P','C','P'],
                 fsize=[[4,4],[3,3]],
                 ksize=[[2,2],[2,2]],
                 batch_size=32,
                 dropout=0.2)
if select_case==3:
    classifier = supervised_sAE(
                 output_func='softmax',
                 hidden_func='sigmoid',
                 loss_func='cross_entropy',
                 struct=[x_dim, 400, 200, 100, y_dim],
                 lr=1e-3,
                 use_for='classification',
                 epochs=10,
                 batch_size=32,
                 dropout=0.12,
                 ae_type='ae', # ae | dae | sae
                 act_type=['sigmoid','sigmoid'],# decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                 noise_type='mn', # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.5, # DAE：噪声损失系数 | SAE：稀疏损失系数
                 p=0.3, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                 ae_lr=1e-3,
                 ae_epochs=5,
                 pre_train=True)
run_sess(classifier,datasets,filename,load_saver='')
label_distribution = classifier.label_distribution