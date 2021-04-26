import tensorflow as tf
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

train_path = './tea_image_label/tea_train_jpg_531/'
train_txt = './tea_image_label/tea_train_jpg_531.txt'
x_train_savepath = './tea_image_label/tea_x_train.npy'
y_train_savepath = './tea_image_label/tea_y_train.npy'

test_path = './tea_image_label/tea_test_jpg_161/'
test_txt = './tea_image_label/tea_test_jpg_161.txt'
x_test_savepath = './tea_image_label/tea_x_test.npy'
y_test_savepath = './tea_image_label/tea_y_test.npy'


def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = np.array(img.convert('1'))  # 图片变为8位宽灰度值的np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_

if __name__ == '__main__':
#
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

    print(x_train.shape)
    print(x_test.shape)
    print(x_train[0])
    np.random.seed(116)
    np.random.shuffle(x_train)
    np.random.seed(116)
    np.random.shuffle(y_train)
    plt.figure()
    plt.imshow(x_train[0])
    plt.colorbar()
    plt.grid(True)
    plt.show()
    class_names = ['maohong','longjin','foshou','shuangfeng','qiqu']
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    checkpoint_save_path = "./checkpoint/mnist.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    # model.fit(x_train, y_train, batch_size=16,epochs=)
    # model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), validation_freq=1)
    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1,
                        callbacks=[cp_callback])
    print(model.trainable_variables)
    file = open('./weights.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')

        file.write(str(v.numpy()) + '\n')
    file.close()

    # model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1)
    # print('\nTest accuracy:', test_acc)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

