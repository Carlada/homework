import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.layers import Dense,Input,Dropout
from keras.models import Model
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import PIL
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

import pandas as pd
from keras.preprocessing.image import *
np.random.seed(2017)

def fun():

    X_train = []
    X_test = []
    for filenames in ["gap_RetNet50.h5"]:
        filename = filenames
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['train']))
            X_test.append(np.array(h['test']))
            y_train = np.array(h['label'])
    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)
    inputs = Input(X_train.shape[1:]) # shape=（2048*3，）
    x = Dropout(0.5)(inputs)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, x)

    model.compile(optimizer='adadelta',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, nb_epoch=8, validation_split=0.2, verbose=2)

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory('test', (224,224), shuffle=False, batch_size=1, class_mode=None)

    input_tensor = Input((224, 224, 3))
    x = input_tensor

    base_model = ResNet50(input_tensor=x, weights='imagenet', include_top=False)
    model2 = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    test = model2.predict_generator(test_generator, test_generator.samples)

    predict_y = model.predict(test, verbose=1)
    print(predict_y)

    fig2, ax2 = plt.subplots(figsize=(2, 2))

    images = []
    list = os.listdir('test/1') #列出文件夹下所有的目录与文件

    for i in range(0,len(list)):
        path = os.path.join('test/1',list[i])
        if os.path.isfile(path):
            images.append(path)


    # 绘制二维图
    for i in range(0,len(images)):
        image = mpimg.imread(images[i])

        if predict_y[i] > 0.5:
            print("狗")
        else:
            print("猫")

        plt.imshow(image)
        plt.show()



# y_pred = model.predict(X_test, verbose=1)
# print(y_pred.__len__())
# y_pred = y_pred.clip(min=0.005, max=0.995)
#
# df = pd.read_csv("F:\PythonWorkSpace\DeepLearningProject\sample_submission.csv")
#
# gen = ImageDataGenerator()
# test_generator = gen.flow_from_directory("F:\PythonWorkSpace\FinalDeepLearning\\test", (224, 224), shuffle=False,
#                                          batch_size=1, class_mode=None)
#
# for i, fname in enumerate(test_generator.filenames):
#     index = int(fname[fname.rfind('\\')+1:fname.rfind('.')])
#     df.set_value(index-1, 'label', y_pred[i])
#
# df.to_csv('pred.csv', index=None)
# df.head(10)

if __name__=="__main__":
    fun()