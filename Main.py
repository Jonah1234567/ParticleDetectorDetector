import numpy as np
import os
import tensorboard
import cv2
import datetime
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from numpy import mean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import KFold
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD, Adam
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from Models import define_model_basic, define_model_vgg_16
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def load_dataset(edge_detect, threshold_one, threshold_two, height, width, img_folder, stop):
    print("hi")
    img_data_array = []
    ans_key = []
    i = 0

    if os.listdir(img_folder) != None:
        for dir1 in os.listdir(img_folder):
            print(i, "'s data samples have been processed")
            i += 1

            for file in os.listdir(os.path.join(img_folder, dir1)):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                if edge_detect==True:
                    image = cv2.Canny(image, threshold1=threshold_one, threshold2=threshold_two)
                image = cv2.resize(image, (height, width), interpolation=cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                image = np.resize(image, (32,32,1))
                img_data_array.append(image)
                ans_arr = np.zeros(7)
                ans_arr[int(dir1)] = 1
                ans_key.append(ans_arr)
            if stop == True:
                break
        x_here, y_here = np.array(img_data_array), np.array(ans_key)
        data_X, data_y = x_here.astype('float'), y_here.astype('float')
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=2)
        return X_train, X_test, y_train, y_test

def test_dataset(edge_detect, threshold_one, threshold_two, height, width, img_folder, stop):
    img_data_array = []
    vis = []
    ans_key = []
    i = 0

    if os.listdir(img_folder) != None:
        print(i, "'s data samples have been processed")
        i += 1

        for file in os.listdir(os.path.join(img_folder)):
            image_path = os.path.join(img_folder, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            vis.append(image)
            if edge_detect==True:
                image = cv2.Canny(image, threshold1=threshold_one, threshold2=threshold_two)
            vis.append(image)
            image = cv2.resize(image, (height, width), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            image = np.resize(image, (32,32,1))
            img_data_array.append(image)
            ans_arr = np.zeros(7)
            ans_arr[0] = 1
            ans_key.append(ans_arr)
    x_here, y_here = np.array(img_data_array), np.array(ans_key)
    data_X, data_y = x_here.astype('float'), y_here.astype('float')
    return data_X, data_y, vis


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(3, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        red_patch = mpatches.Patch(color='orange', label='Test')
        blue_patch = mpatches.Patch(color='blue', label='Train')
        plt.legend(handles=[red_patch, blue_patch])
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Loss")
        # plot accuracy
        pyplot.subplot(3, 1, 3)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        red_patch = mpatches.Patch(color='orange', label='Test')
        blue_patch = mpatches.Patch(color='blue', label='Train')
        plt.legend(handles=[red_patch, blue_patch])
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Accuracy (0-1)")
    pyplot.show()


def evaluate_model(dataX, dataY, n_folds, ep, bs, verb, opt, loss, dropout):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model_basic(True, True, dropout, opt, loss)
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model

        history = model.fit(trainX, trainY, epochs=ep, batch_size=bs, validation_data=(testX, testY), verbose=verb)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    print(scores)
    return scores, histories, model, np.mean(scores)


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()

def visual_test(model):
    test_set_X, test_set_Y, visuals = test_dataset(True, 500, 1000, 1920, 1080,
                                                   'C:\\Users\\jonah\\OneDrive\\Desktop\\Test Images 2', True)
    results = model.predict(test_set_X)
    print(len(results))
    for i in range(0, len(visuals), 2):
        print(i / 2)
        img_a = (cv2.resize(visuals[i], (640, 480), interpolation=cv2.INTER_AREA))
        img_b = cv2.cvtColor((cv2.resize(visuals[i + 1], (640, 480), interpolation=cv2.INTER_AREA)), cv2.COLOR_GRAY2RGB)
        print(np.shape(img_a))
        print(np.shape(img_b))
        Hori = np.concatenate((img_a, img_b), axis=1)
        print("Prediction, ", "number ", int(i / 2), "prediction:", np.argmax(results[int(i / 2)]))
        cv2.imshow(str(i / 2), Hori)
        cv2.waitKey(0)

def run_test_harness():
    # load dataset
    trainX, testX, trainY, testY = load_dataset(True, 100, 200, 32, 32, 'C:\\Users\\jonah\\OneDrive\\Desktop\\cloud chamber data final', True)
    # evaluate model
    scores, histories, model, percent_acc = evaluate_model(trainX, trainY, 5, 10, 32,0,SGD(learning_rate=0.1, momentum=0.9), 'categorical_crossentropy', dropout=0.5)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)
    visual_test(model)



# entry point, run the test harness
#run_test_harness()
