import os
import shutil
import numpy as np
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from sklearn.utils import shuffle

import h5py

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# 导出特征向量
def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]  # image的宽度
    height = image_size[1]  # image的高度
    input_tensor = Input((height, width, 3))  # 设置输入的tensor
    x = input_tensor

    # 对x执行lambda_func函数
    if lambda_func:
        x = Lambda(lambda_func)(x)

    # input_tensor 表示模型的tensor
    # include_top 表示是否包含模型顶部的全连接层
    # weights None 表示随机初始化，即不加载预训练权重 imagenet 表示加载权重
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    # GlobalAveragePooling2D() 全局平均值池化层 作为输出
    # 使用base_model的结果用作训练
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    # 创建多分类标签生成器
    gen = ImageDataGenerator()
    # 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
    # 路径， 图片大小， 是否打乱数据， batch数据的大小， class_mode 返回的标签数组的形式 None 不返回标签
    train_generator = gen.flow_from_directory("train/", image_size, shuffle=False, batch_size=16)
    test_generator = gen.flow_from_directory("data/test/", image_size, shuffle=False, batch_size=16, class_mode=None)

    # 从一个生成器上获取数据并进行预测（导出特征向量） generator：生成输入batch数据的生成器
    train = model.predict_generator(train_generator, 20000, verbose=1)
    test = model.predict_generator(test_generator, 12500, verbose=1)

    # h5py 一个 HDF5 文件是存储两类对象的容器
    with h5py.File("gap_%s.h5" % MODEL.func_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


# preprocess_input() 函数完成数据预处理的工作，数据预处理能够提高算法的运行效果。常用的预处理包括数据归一化和白化
# 使用三种模型 进行综合考虑
write_gap(ResNet50, (224, 224))
# write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
# write_gap(Xception, (299, 299), xception.preprocess_input)

