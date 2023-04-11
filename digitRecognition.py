#
# @file digitReoognition.py
# @author Melih Altun @2023
#

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
from keras.layers import LeakyReLU
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import random
import os

#download train.csv from Kaggle (Digit Recognizer Competition) and place it under "./digits_data/"


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs Available: ', len(physical_devices))
if len(physical_devices) > 0:    # comment or set to false if GPU parrellization is not needed
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalizes Confusion Matrix")
    else:
        print("Confusion Matrix, without normalization")
    print(cm)
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# train test validation split
def splitData(dataArray, pctTrain, pctVal):
    N = dataArray.shape[0]
    pctTrain /= 100
    pctVal /= 100
    N_train = round(N*pctTrain)
    N_val = round(N*pctVal)
    N_test = N - N_train - N_val

    indices = np.arange(N)
    np.random.shuffle(indices)

    trainData = dataArray[indices[:N_train], :]
    valData = dataArray[indices[N_train:N_train+N_val], :]
    testData = dataArray[indices[N_train+N_val:N_train+N_val+N_test], :]

    trainLabels = trainData[:, 0]
    valLabels = valData[:, 0]
    testLabels = testData[:, 0]

    trainData = trainData[:, 1:]
    valData = valData[:, 1:]
    testData = testData[:, 1:]

    return trainLabels, trainData, valLabels, valData, testLabels, testData


def preProcessData(data, mean_all=0):
    data = data/255.0
    if mean_all == 0:
        mean_all = np.mean(data)
    data = data - mean_all
    return data, mean_all


def reshapeData(data, sz1, sz2):
    data = data.reshape(-1, sz1, sz2, 1)
    return data


if os.path.isdir('./digits_data/train_test_val') is False:
    dataAll = pd.read_csv('./digits_data/train.csv')
    dataAll = dataAll.to_numpy()
    trainLabels, trainData, valLabels, valData, testLabels, testData = splitData(dataAll, 90, 6)
    os.makedirs('./digits_data/train_test_val')
    pd.DataFrame(trainData).to_csv('./digits_data/train_test_val/trainData.csv', index=False, header=False)
    pd.DataFrame(trainLabels).to_csv('./digits_data/train_test_val/trainLabels.csv', index=False, header=False)
    pd.DataFrame(valData).to_csv('./digits_data/train_test_val/valData.csv', index=False, header=False)
    pd.DataFrame(valLabels).to_csv('./digits_data/train_test_val/valLabels.csv', index=False, header=False)
    pd.DataFrame(testData).to_csv('./digits_data/train_test_val/testData.csv', index=False, header=False)
    pd.DataFrame(testLabels).to_csv('./digits_data/train_test_val/testLabels.csv', index=False, header=False)
else:
    trainLabels = pd.read_csv('./digits_data/train_test_val/trainLabels.csv').to_numpy()
    trainData = pd.read_csv('./digits_data/train_test_val/trainData.csv').to_numpy()
    valLabels = pd.read_csv('./digits_data/train_test_val/valLabels.csv').to_numpy()
    valData = pd.read_csv('./digits_data/train_test_val/valData.csv').to_numpy()
    testLabels = pd.read_csv('./digits_data/train_test_val/testLabels.csv').to_numpy()
    testData = pd.read_csv('./digits_data/train_test_val/testData.csv').to_numpy()


batch_sz = 32
n_classes = 10
useLeakyRelu = True

trainData, meanData = preProcessData(trainData)
testData, _ = preProcessData(testData, meanData)
valData, _ = preProcessData(valData, meanData)

trainData = reshapeData(trainData, 28, 28)
valData = reshapeData(valData, 28, 28)
testData = reshapeData(testData, 28, 28)

trainLabels_enc = to_categorical(trainLabels, num_classes=n_classes)
valLabels_enc = to_categorical(valLabels, num_classes=n_classes)
#testLabels_enc = to_categorical(testLabels, num_classes=n_classes)

train_x = tf.convert_to_tensor(trainData)
train_y = tf.convert_to_tensor(trainLabels_enc)
val_x = tf.convert_to_tensor(valData)
val_y = tf.convert_to_tensor(valLabels_enc)
test_x = tf.convert_to_tensor(testData)
#test_y = tf.convert_to_tensor(testLabels_enc)

trainDataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
trainDataset = trainDataset.batch(batch_sz).prefetch(tf.data.AUTOTUNE)

valDataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
valDataset = valDataset.batch(batch_sz).prefetch(tf.data.AUTOTUNE)

#define model
if useLeakyRelu:
    model = Sequential([Conv2D(filters=8, kernel_size=(9, 9), padding='same', input_shape=(28, 28, 1)), LeakyReLU(alpha=0.1),
        MaxPool2D(pool_size=(2, 2), strides=1),
        Conv2D(filters=16, kernel_size=(7, 7), padding='same'), LeakyReLU(alpha=0.1),
        MaxPool2D(pool_size=(2, 2), strides=1),
        Conv2D(filters=32, kernel_size=(5, 5), padding='same'), LeakyReLU(alpha=0.1),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same'), LeakyReLU(alpha=0.1),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=32), LeakyReLU(alpha=0.1),
        Dense(units=48), LeakyReLU(alpha=0.1),
        Dense(units=16), LeakyReLU(alpha=0.1),
        Dense(units=10, activation='softmax')
    ])
else:
    model = Sequential([Conv2D(filters=8, kernel_size=(9, 9), activation='relu', padding='same', input_shape=(28, 28, 1)),
                        MaxPool2D(pool_size=(2, 2), strides=1),
                        Conv2D(filters=16, kernel_size=(7, 7), activation='relu', padding='same'),
                        MaxPool2D(pool_size=(2, 2), strides=1),
                        Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'),
                        MaxPool2D(pool_size=(2, 2), strides=2),
                        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
                        MaxPool2D(pool_size=(2, 2), strides=2),
                        Flatten(),
                        Dense(units=32, activation='relu'),
                        Dense(units=48, activation='relu'),
                        Dense(units=16, activation='relu'),
                        Dense(units=10, activation='softmax')])

# b=32, e=16 - f#16 9x9, mx s1, f#32 7x7, mx s1,  f#64 5x5, mx s2,  f#128 3x3, mx s2, flt, d32, d48, d16, d10 -> Tr .9959 Val .9833, Ts .9857
# b=32, e=16 - f#8 9x9, mx s1, f#16 7x7, mx s1,  f#32 5x5, mx s2,  f#64 3x3, mx s2, flt, d32, d48, d16, d10 -> Tr .9942 Val .9857 Ts .9875
# b=32, e=16 - leaky relu - f#8 9x9, mx s1, f#16 7x7, mx s1,  f#32 5x5, mx s2,  f#64 3x3, mx s2, flt, d32, d48, d16, d10 -> Tr .9935 Val .9857 Ts .9815
# b=32, e=24 - leaky relu - f#8 9x9, mx s1, f#16 7x7, mx s1,  f#32 5x5, mx s2,  f#64 3x3, mx s2, flt, d32, d48, d16, d10 -> Tr .9964 Val .9845 Ts .9899
model.summary()

checkpoint_filepath = './models/digitRec_model_checkpoint.h5'
os.makedirs('./models', exist_ok=True)
# callback function to save model after each epoch
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=trainDataset, validation_data=valDataset, epochs=24, callbacks=[model_checkpoint_callback])

#apply fitted model to test data
predictions = model.predict(x=test_x, verbose=0)
rounded_predictions = np.argmax(predictions, axis=-1)

cm = confusion_matrix(y_true=testLabels, y_pred=rounded_predictions)
cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

print("Test Accuracy = ")
print(np.sum(testLabels == rounded_predictions.reshape(len(rounded_predictions), 1))/len(testLabels))

dummy = 1
