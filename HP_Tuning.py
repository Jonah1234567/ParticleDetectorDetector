
import numpy as np
import tensorboard
from tensorboard import notebook

import Main
from Models import define_model_basic
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, KLDivergence
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([16, 32]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_ACCURACY = 'accuracy'


def get_accuracy(dx, dy, n, e, b, v, o, lo, do):
    scores, histories, model, percent_acc = Main.evaluate_model(dataX=dx, dataY=dy, n_folds=n, ep=e, bs=b, verb=v,
                                                                opt=o,
                                                                loss=lo, dropout=do)
    return percent_acc


def run(run_dir, hparams_run, data_x, data_y):

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams_run)
        acc = get_accuracy(dx=data_x, dy=data_y, n=5, v=0, do=hparams_run[HP_DROPOUT], b=hparams_run[HP_BATCH_SIZE],
                           o=hparams_run[HP_OPTIMIZER], e=hparams_run[HP_EPOCHS])
        tf.summary.scalar(METRIC_ACCURACY, acc, step=10)


with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_BATCH_SIZE, HP_EPOCHS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

# load dataset
trainX, testX, trainY, testY = Main.load_dataset(True, 100, 200, 32, 32,'C:\\Users\\jonah\\OneDrive\\Desktop\\cloud chamber data final', True)
print(np.shape(trainX))
print(np.shape(trainY))
print(np.shape(testX))
print(np.shape(testY))
data_X = np.concatenate((trainX, testX))
data_Y = np.concatenate((trainY, testY))
session_num = 0

for batch_size in HP_BATCH_SIZE.domain.values:
    for dropout_rate in tf.linspace(HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value, 3):
        for optimizer in HP_OPTIMIZER.domain.values:
            for epoch in HP_EPOCHS.domain.values:
                hparams = {
                    HP_BATCH_SIZE: batch_size,
                    HP_DROPOUT: float("%.2f" % float(dropout_rate)),
                    # float("%.2f"%float(dropout_rate)) limits the decimal palces to 2
                    HP_OPTIMIZER: optimizer,
                }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams, data_X, data_Y)
            session_num += 1
notebook.display(port=6006, height=1000)