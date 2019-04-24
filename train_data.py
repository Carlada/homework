import os
import cv2
import shutil
import random
import numpy as np

# 数据预处理
train = 'data/train/'

# 分类train中的dog和cat
dogs = [i for i in os.listdir(train) if 'dog' in i]
cats = [i for i in os.listdir(train) if 'cat' in i]

# 分别输出dog和cat的训练集数量
print(len(dogs), len(cats))

# 训练集数量
train_count = (int)((len(dogs) + len(cats)) * 0.8 / 2)

random.shuffle(dogs)
random.shuffle(cats)

# 创建目录，如果存在就覆盖
def rmrf_mkdir(dirname) :
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


# 创建目录
rmrf_mkdir('train')
os.mkdir('train/cat')
os.mkdir('train/dog')
rmrf_mkdir('validation')
os.mkdir('validation/cat')
os.mkdir('validation/dog')

# 复制图片
for dog_file in dogs[:train_count]:
    shutil.copyfile('data/train/' + dog_file, 'train/dog/' + dog_file)

for cat_file in cats[:train_count]:
    shutil.copyfile('data/train/' + cat_file, 'train/cat/' + cat_file)

for dog_file in dogs[train_count:]:
    shutil.copyfile('data/train/' + dog_file, 'validation/dog/' + dog_file)

for cat_file in cats[train_count:]:
    shutil.copyfile('data/train/' + cat_file, 'validation/cat/' + cat_file)



# dogs = [i for i in os.listdir('train/dog/')]
# cats = [i for i in os.listdir('train/cat/')]
#
#
# rmrf_mkdir('train1')
# os.mkdir('train1/cat')
# os.mkdir('train1/dog')
# rmrf_mkdir('validation1')
# os.mkdir('validation1/cat')
# os.mkdir('validation1/dog')
# # 复制图片
# for dog_file in dogs[:2000]:
#     shutil.copyfile('train/dog/' + dog_file, 'train1/dog/' + dog_file)
#
# for cat_file in cats[:2000]:
#     shutil.copyfile('train/cat/' + cat_file, 'train1/cat/' + cat_file)
#
# dogs = [i for i in os.listdir('validation/dog/')]
# cats = [i for i in os.listdir('validation/cat/')]
# for dog_file in dogs[:1000]:
#     shutil.copyfile('validation/dog/' + dog_file, 'validation1/dog/' + dog_file)
#
# for cat_file in cats[:1000]:
#     shutil.copyfile('validation/cat/' + cat_file, 'validation1/cat/' + cat_file)
#
