# ç Francesco Pugliese
# general imports
import keras
import numpy as np
import pdb
import matplotlib.pyplot as plt

# keras imports
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import array_to_img, img_to_array
from keras.optimizers import SGD

import os

default_callbacks = []
limit = None
split = None
epochs = 100
training = True
classify = True
show_dataset = False

# Set CPU or GPU type
gpu = True
gpu_id = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
if gpu == False: 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else: 
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# Show the first image from the training set

if show_dataset == True: 
    plt.imshow(array_to_img(X_train[0]))
    plt.savefig("first_cifar10_mnist_train_image.jpg")
    print("First cifar10 mnist train image", y_train[0][0])
    plt.show(block = False)
    plt.pause(3)
    plt.close()

    # Show the first image from the test set
    plt.imshow(array_to_img(X_test[0]))
    plt.savefig("first_cifar10_mnist_test_image.jpg")
    print("First fashion mnist test image", y_test[0][0])
    plt.show(block = False)
    plt.pause(3)
    plt.close()

#pdb.set_trace()

# Normalization (testare come senza normalizzazione converge molto tardi a 95% rispetto alla normalizzazione)
X_train = X_train / 255.0
X_test = X_test / 255.0

if limit is not None: 
    X_train = X_train[0:limit]
    y_train = y_train[0:limit]
    X_test = X_test[0:limit]
    y_test = y_test[0:limit]
    

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
              
model.summary()

if training == True: 
    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    checkPoint=ModelCheckpoint("cifar10.cnn", save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    default_callbacks = default_callbacks+[checkPoint]

    #earlyStopping=EarlyStopping(monitor='val_loss', min_delta = 0.01, patience=10, 
    #                                                        verbose=0, mode='min') 
    #default_callbacks = default_callbacks+[earlyStopping]

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(X_train, y_train, validation_split = 0.2, epochs=epochs, batch_size=32, callbacks = default_callbacks, verbose = 2)

    score = model.evaluate(X_test, y_test, batch_size=32)
    print(score)

#if classify == True:
#    model.load_weights("cifar10.cnn")
    # load the image, pre-process it, and store it in the data list
#    image = cv2.imread("first_cifar10_mnist_test_image.jpg")
    
score = model.evaluate(X_test, y_test, batch_size=32)
print(score)
