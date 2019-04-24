from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# rmsprop = optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# horizontal_flip 进行随机水平翻转 zoom_range 随机缩放的幅度 shear_range 剪切强度 rescale 值将在执行其他处理前乘到整个图像上
train_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# rotation_range=20, width_shift_range=0.2, height_shift_range=0.2
validation_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory('train/', target_size=(224, 224), batch_size=32, class_mode='binary')
validation_generator = validation_gen.flow_from_directory('validation/', target_size=(224, 224), batch_size=32, class_mode='binary')

model.fit_generator(train_generator, steps_per_epoch=20000, epochs=5, validation_data=validation_generator, validation_steps=5000, verbose=1)

model.save('model.h5')
# 保存参数，载入参数e
model.save_weights('my_model_weights.h5')

