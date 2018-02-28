import pandas as pd
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import regularizers, losses
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
import os


##################################################


class lstm():
    # takes parameters needed for lstm
    def __init__(self, exp, timestep, data):
        """

        :param timestep: time step used in training (numpy array)
        :param data:numpy array of timeseries value
        """
        self.exp_dir = 'experiments/exp{}/'.format(exp)
        self.sum_dir = self.exp_dir + 'summaries/'
        self.timestep = timestep
        self.data = data
        # number of data samples
        self.n = data.shape[0]

        print("experiment directory: ", self.exp_dir)
        print('model initialized with time steps : {} , data observations : {}\n'.format(self.timestep, self.n))

        print('creating experiment directory\n')
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        if not os.path.exists(self.sum_dir):
            os.makedirs(self.sum_dir)

        print("preparing data for lstm")
        self.prepare_data()
        pass

    # prepare data shape for lstm (split, normalize, reshape)

    def prepare_data(self):

        # split into train and validate
        num_train = int(0.7 * self.n)
        train_data = self.data[0:num_train]
        test_data = self.data[num_train:]

        print("training sequence length: {} , test sequence length: {} ".format(num_train, self.n - num_train))

        # scale to values between 0-1
        self.mean = np.mean(train_data)
        self.std = np.mean(test_data)

        train_data -= self.mean
        test_data -= self.mean

        train_data /= self.std
        test_data /= self.std

        # reshape x into an array of shape(n, timesteps, 1 ) , y:(n, timesteps)
        # each value in the series and its next

        print("reshaping data using timestep")
        self.x_train, self.y_train = self.reshape_sequence(train_data)
        self.x_test = test_data[0:-1].reshape(-1, 1, 1)
        self.y_test = test_data[1:].reshape(-1, 1, 1)

        print("reshaped sequence for lstm ")
        print("x_train shape : {} , y_train shape : {} ".format(self.x_train.shape, self.y_train.shape))
        print("x_test shape : {} , y_test shape : {} \n".format(self.x_test.shape, self.y_test.shape))

        print("model initialization done ...........")

    def reshape_sequence(self, seq):
        """

        :param seq: sequence to be reshaped
        :return: x,y rehsaped based on self. timestep
        """

        n = seq.shape[0]
        window = self.timestep
        seq_len = (n - 1) - (n - 1) % window

        # y is shifted version of x
        x = self.data[0:seq_len]
        y = self.data[1:seq_len + 1]

        # reshape using time steps
        x = x.reshape(-1, window, 1)
        y = y.reshape(-1, window, 1)

        return x, y

    def difference(self):
        pass

    #########################################

    # build lstm architecture,
    def build_model(self, neurons, lr, batch_size, num_epochs):
        """
        construct stateful lstm model
        :param exp: experiment number
        :param neurons: number of neurons in each step
        :param lr: learning rate
        :param batch_size:
        :param num_epochs:
        :return:
        """

        # save parameters to reconstruct other model


        # reshape train data into (samples,timesteps,1)
        train_mask = self.x_train.shape[0] - (self.x_train.shape[0] % batch_size)

        test_mask = self.x_test.shape[0] - (self.x_test.shape[0] % batch_size)

        self.x_train = self.x_train[:train_mask]
        self.y_train = self.y_train[:train_mask]

        self.x_test = self.x_test[:test_mask]
        self.y_test = self.y_test[:test_mask]

        print("new data shapes after making it divisble by batch size")
        print("x_train: {}  , y_train: {} ".format(self.x_train.shape, self.y_train.shape))
        print("x_test: {}  , y_test: {} ".format(self.x_test.shape, self.y_test.shape))

        model = Sequential()
        model.add(LSTM(neurons[0], stateful=True,
                       batch_input_shape=(batch_size, self.x_train.shape[1], self.x_train.shape[2]),
                       return_sequences=True))

        model.add(LSTM(neurons[1], stateful=True, return_sequences=True))
        model.add(LSTM(neurons[2], stateful=True, return_sequences=True))

        adam = Adam(lr=lr)



        model.compile(loss='mean_squared_error', optimizer=adam)
        loss_hist = []
        # fit model
        for i in range(num_epochs):
            history = model.fit(self.x_train, self.y_train, validation_split=0,epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
            loss_hist += history.history['loss']
            model.reset_states()

        # plot history
        plt.plot(loss_hist)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper left')
        plt.show()

        # save parameters needed for predicting
        self.neurons = neurons
        self.opt = adam

        model.save_weights(self.exp_dir + 'weights')
        return model

    # build identical model and predict sequence one step at a time
    def predict(self, warmup, preds):
        """

        :param warmup: number of time steps to warm up h
        :param preds: number of timesteps to predict
        :return:
        """

        neurons = self.neurons

        # reconstruct model with timesteps=1, batchsize=1
        model = Sequential()
        model.add(LSTM(neurons[0], stateful=True, batch_input_shape=(1, 1, 1), return_sequences=True))

        model.add(LSTM(neurons[1], stateful=True, return_sequences=True))
        model.add(LSTM(neurons[2], stateful=True, return_sequences=True))

        adam = self.opt

        model.compile(loss='mean_squared_error', optimizer=adam)

        # restore trained weights
        model.load_weights(self.exp_dir + 'weights')
        print("training weights loaded successfully \n")
        predictions = []

        prev=np.zeros((1,1,1))

        for i in range(preds):
            p=None
            if i<=warmup:
                p = model.predict(self.x_test[i].reshape(1,1,1))
            else:
                #prev=prev.reshape(1,1,1)
                p=model.predict(prev)
                prev=p

            predictions+=[p.squeeze()]

        #plot results
        plt.plot(predictions)
        plt.plot(self.y_test[:preds].squeeze())
        plt.legend(['predicted', 'true'])
        plt.show()


data = pd.read_csv('interpolated_data.csv', index_col=0, parse_dates=True)[['value']].values
m = lstm(1, 100, data)
m.build_model([32, 32, 1], 0.001, 32, 500)
m.predict(50,500)
