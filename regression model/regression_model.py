### Fully connected neural network with softmax loss
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.utils import plot_model
from time import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools
import utils as u
from classification import Classifier
from reg import Regression

##################################################

class Train:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.load_data()

    def load_data(self):
        """
        load train, test and validation data
        :return:
        """

        data_dir = self.config['data_dir']
        print(data_dir)
        x_train = np.load(data_dir + "x_train.npy")
        x_val = np.load(data_dir + "x_val.npy")
        x_test = np.load(data_dir + "x_test.npy")

        y_train = np.load(data_dir + "y_train.npy")
        y_val = np.load(data_dir + "y_val.npy")
        y_test = np.load(data_dir + "y_test.npy")

        print("data loaded successfully")
        print("training data : ", x_train.shape, x_train.dtype)
        print("training labels : ", y_train.shape, y_train.dtype)
        print("validation data : ", x_val.shape, x_val.dtype)
        print("validation labels : ", y_val.shape, y_val.dtype)
        print("test data : ", x_test.shape, x_test.dtype)
        print("test labels : ", y_test.shape, y_test.dtype)

        # #normalize data
        mean = np.mean(x_train, axis=0, keepdims=True)
        std = np.std(x_train, axis=0, keepdims=True)

        x_train -= mean
        x_val -= mean
        x_test -= mean

        x_train /= std
        x_val /= std
        x_test /= std

        # save data to trainer
        self.x_train = x_train
        self.y_train = y_train

        self.x_val = x_val
        self.y_val = y_val

        self.x_test = x_test
        self.y_test = y_test

        # sample data to test overfitting
        # s=np.random.choice(x_train.shape[0], 500)
        # self.xs_train=x_train[s]
        # self.ys_train=y_train[s]
        #
        # s2=np.random.choice(x_val.shape[0], int(0.7*500))
        # self.xs_val=x_val[s2]
        # self.ys_val=y_val[s2]


        pass

    def train(self):
        exp = self.config['exp']
        epochs = self.config['num_epochs']
        batch_size = self.config['batch_size']
        lr = self.config['learning_rate']

        checkpoint_dir = self.config['checkpoint_dir'] + "weights-best.hdf5"
        summ_dir = self.config['summ_dir']
        exp_dir = self.config['exp_dir'] + "model.h5"

        checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        tensorboard = TensorBoard(log_dir=summ_dir)

        adam = Adam(lr)

        self.model.compile(optimizer=adam, loss='mean_sqaured_error', metrics=['mse'])

        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), shuffle=True,
                       batch_size=batch_size, callbacks=[tensorboard, checkpoint], epochs=epochs)

        # self.model.load_weights(checkpoint_dir)
        #
        # scores = self.model.evaluate(self.x_val, self.y_val)
        # print("validation accuracy before saving")
        # print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

        print("saving model to : ", exp_dir)
        self.model.save(exp_dir)

    def load(self):
        #loads best model and resores graph
        exp_dir = self.config['exp_dir'] + "model.h5"
        checkpoint_file = self.config['checkpoint_dir'] + "weights-best.hdf5"

        print("model path : ", exp_dir)
        print("weights path : ", checkpoint_file)

        # load moadel and best checkpoint
        m_load = load_model(exp_dir)
        m_load.load_weights(checkpoint_file)

        print("loaded model and weights successfully")

        # validation accuracy
        #scores feha elaccuracy
        scores = m_load.evaluate(self.x_val, self.y_val)
        predictions_val = m_load.predict(self.x_val)

        print("labels_shape", self.y_val.shape)
        print("predicted_labels shape", predictions_val.shape)
        print("validation mse of loaded model")
        print("%s: %.2f%%" % (m_load.metrics_names[1], scores[1] * 100))

        print("-----------------------------------")

        # test acuracy
        scores = m_load.evaluate(self.x_test, self.y_test)
        predictions = m_load.predict(self.x_test)
        print("labels_shape", self.y_test.shape)
        print("predicted_labels shape", predictions.shape)
        print("test mse of loaded model")
        print("%s: %.2f%%" % (m_load.metrics_names[1], scores[1] * 100))

        print("--------------------------")
        print("wrong predcictions: ")

        return (self.y_test, predictions)




def main():
    # read args
    args = u.read_args()
    u.create_directories(args)

    #create classification model
    c=Regression(args)
    #if training flag is true build model and train it
    if args['train']:

        model = c.build()
        plot_model(model, to_file='regression.png' ,show_layer_names=False , show_shapes=False)
        operator = Train(model, args)
        operator.train()

    #if test is true, load best model and test it
    if args['test']:
        #load data only without creating model
        operator = Train(None, args)
        true, predicted = operator.load()

        plt.plot(true, color='red', label='true')
        plt.plot(predicted, color='blue')
        plt.show()


main()
