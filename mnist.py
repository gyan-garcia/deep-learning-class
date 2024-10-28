# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:48:16 2024

@author: gyang
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense 
import matplotlib.pyplot as plt

def display_digit(data, labels, i):
    img = data[i]
    plt.title('Example %d. Label: %d' % (i, labels[i]))
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)
    plt.show()
    
#Download the mnist dataset from Keras


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshape our dataset to be one dimention only
x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


#Let's display some of the entries
display_digit(x_train, y_train, 400)


# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_test[0]


model = Sequential()
#model.add(Dense(16, input_dim=784, activation='relu'))
model.add(Dense(num_classes, input_dim=784, activation='softmax')) 


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
        
# Fit the model
model.fit(x_train.reshape(60000, 784), y_train, epochs=5, batch_size=32, validation_split=0.1)


# Score the model
score = model.evaluate(x_test.reshape(10000, 784), y_test, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])


layers = model.layers
weights = layers[0].get_weights()

f, axes = plt.subplots(2, 5, figsize=(10,4))
axes = axes.reshape(-1)
for i in range(len(axes)):
    a = axes[i]
    a.imshow((weights[0][0:784, i:i+1]).reshape(28, 28), cmap=plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(())
    a.set_yticks(())
plt.show()