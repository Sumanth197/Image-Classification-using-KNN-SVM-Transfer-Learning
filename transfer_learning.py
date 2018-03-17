import keras
import pandas as pd
import numpy as np
from keras.models import Model
import cv2
import os
import numpy as np


def VGF16(img_shape, num_classes):

    model_16 = keras.applications.VGG16(include_top=False, weights='imagenet')
    keras_input = keras.layers.Input(shape=img_shape)
    output_vgg = model_16(keras_input)
    output_vgg = keras.layers.Flatten()(output_vgg)
    # x = keras.layers.Dense(2056, activation='relu')(output_vgg)
    # x = keras.layers.Dense(1028, activation='relu')(output_vgg)
    x = keras.layers.Dense(num_classes, activation='softmax')(output_vgg)
    # x = keras.layers.Dense(num_classes, activation='softmax')(output_vgg)
    model = Model(inputs=keras_input, outputs=x)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    # print model.layers[-1]
    # print (model.layers)
    for layer in model_16.layers:
        layer.trainable = False
    for layer in model_16.layers[-1:]:
        layer.trainable = True
    # model.layers[-1].trainable = False
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


import keras

model = VGF16((32, 32, 3), 8)
"""
 model = VGF16((32,32,3), 10)
 model.summary()

 x_train = x_train[0:100,:,:,:]
 y_train = y_train[0:100,:]
"""

train_images = []
for i in range(1, 1889):
    name = str(i) + '.jpg'
    path = os.path.join("/home/sumanth/Desktop/hw2_data/train",name)
    img = cv2.imread(path)
    img = cv2.resize(img,(32,32))
    train_images.append(img)

train_images = np.asarray(train_images).astype(int)

x_train = train_images

test_images = []
for i in range(1, 801):
    name = str(i) + '.jpg'
    path = os.path.join("/home/sumanth/Desktop/hw2_data/test/",name)
    img = cv2.imread(path)
    img = cv2.resize(img,(32,32))
    test_images.append(img)

test_images = np.asarray(test_images).astype(int)

x_test = test_images

train_labels = pd.read_csv('/home/sumanth/Desktop/hw2_data/train_labels.csv')
train_labels = train_labels.columns
train_labels = np.asarray(train_labels).astype('float32')
train_labels = train_labels.astype(int)

y_train = train_labels
y_train = np.asarray(y_train)
y_train[:]=[x-1 for x in y_train]

test_labels = pd.read_csv('/home/sumanth/Desktop/hw2_data/test_labels.csv')
test_labels = test_labels.columns
test_labels = np.asarray(test_labels).astype('float32')
test_labels = test_labels.astype(int)

y_test = test_labels
y_test = np.asarray(y_test)
y_test[:]=[x-1 for x in y_test]


from keras.utils.np_utils import to_categorical
y_test = to_categorical(y_test,8)
y_train = to_categorical(y_train,8)
print 'x_train:\n', len(x_train)
print 'x_test:\n', len(x_test)
print 'y_train:\n', len(y_train)
print 'y_test:\n' , len(y_test)
hist = model.fit(x_train, y_train, epochs=2, verbose=1)
pred = model.predict(x_train)
print (model.evaluate(x_train, y_train))
print (model.evaluate(x_test, y_test))

#y_test = test_labels
# y_train = train_labels
# data_set = all images in train
# x_test = all images in test
