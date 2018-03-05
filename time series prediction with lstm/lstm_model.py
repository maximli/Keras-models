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
        self.timestep = timestep
        self.data = data

        # number of data samples
        self.n = data.shape[0]

        print("experiment directory: ", self.exp_dir)
        print('model initialized with time steps : {} , data observations : {}\n'.format(self.timestep, self.n))


        print('creating experiment directory\n')
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.file= open(self.exp_dir+'expriment_params.txt', 'w')
        self.file.write('time steps : {} , full data length : {}\n'.format(self.timestep, self.n))

        print("preparing data for lstm")
        self.prepare_data()



    # prepare data shape for lstm (split, normalize, reshape)

    def prepare_data(self):

        # split into train and validate
        num_train = int(0.90 * self.n)
        train_data = self.data[0:num_train]
        test_data = self.data[num_train:]

        print("training sequence length: {} , test sequence length: {} ".format(num_train, self.n - num_train))

        # scale to values between 0-1
        self.mean = np.mean(train_data)
        self.std = np.std(train_data)

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
        s1="x_train shape : {} , y_train shape : {} \n".format(self.x_train.shape, self.y_train.shape)
        s2="x_test shape : {} , y_test shape : {} \n".format(self.x_test.shape, self.y_test.shape)
        print(s1)
        print(s2)

        self.file.write(s1)
        self.file.write(s2)



        print("model initialization done ...........")

    def reshape_sequence(self, seq):
        """

        :param seq: sequence to be reshaped
        :return: x,y rehsaped based on self. timestep
        """

        n = seq.shape[0]
        window = self.timestep
        seq_len = (n - 1) - ((n - 1) % window)

        # y is shifted version of x
        x = self.data[0:seq_len]
        y = self.data[1:seq_len + 1]

        # reshape using time steps
        x = x.reshape(-1, window, 1)
        y = y.reshape(-1, window, 1)

        return x, y



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
        self.file.write('neurons: {} \nlr: {} \nbatch size: {} \nepochs: {}\n'.format(neurons,lr,batch_size,num_epochs))

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

        #to make number of hidden layers variable
        for i in range(1,len(neurons)):
            model.add(LSTM(neurons[i], stateful=True, return_sequences=True))


        adam = Adam(lr=lr)



        model.compile(loss='mean_squared_error', optimizer=adam)

        # save parameters needed for predicting
        self.neurons = neurons
        self.opt = adam


        loss_hist = []
        # fit model
        for i in range(num_epochs):
            history = model.fit(self.x_train, self.y_train, validation_split=0,epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            print("epoch {} , loss {} ".format(i,history.history['loss']))
            loss_hist += history.history['loss']
            model.reset_states()

        # plot history
        plt.plot(loss_hist)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper left')
        plt.savefig(self.exp_dir+'trainingloss.png')
        plt.show()


        #save weights after training
        model.save_weights(self.exp_dir + 'weights')
        return model

    def rebuild_model(self):
        #build identical model with batch size =1
        neurons = self.neurons

        # reconstruct model with timesteps=1, batchsize=1
        model = Sequential()
        model.add(LSTM(neurons[0], stateful=True, batch_input_shape=(1, 1, 1), return_sequences=True))

        for i in range(1,len(neurons)):
            model.add(LSTM(neurons[i], stateful=True, return_sequences=True))

        adam = Adam(lr=self.lr)

        model.compile(loss='mean_squared_error', optimizer=adam)

        # restore trained weights
        model.load_weights(self.exp_dir + 'weights')
        print("training weights loaded successfully \n")

        return model


    def predict_training(self,model):
        predictions = []
        #reshape to one time stamp

        #reshape data to match new model
        x_train=self.x_train.reshape(-1,1,1)
        y_train= self.y_train.reshape(-1,1,1)

        #plot y_train
        plt.plot(y_train.squeeze(), 'b')
        axes = plt.gca()
        axes.set_ylim([-5,5])
        plt.title('y_train')
        plt.savefig(self.exp_dir+'training_data.png')
        plt.show()

        #predict next point from it previous
        print("try to predict training data")
        for i in range(len(x_train)):
            p = model.predict(x_train[i].reshape(1,1,1))
            predictions+=[p.squeeze()]

        s=self.std
        m=self.mean

        y=y_train.squeeze()
        true =np.array([((i*s)+m) for i in y])
        predicted=np.array([((i*s)+m) for i in predictions])




        error= predicted -true
        rmse=np.sqrt(np.mean(np.square(error)))
        mae = np.mean(np.abs(error))
        print('training data rmse: {} '.format(rmse))
        print('training data mae: {} '.format(mae))
        self.file.write('rmse on training data: {} '.format(rmse))
        self.file.write('mae on training data: {} '.format(mae))

        #plot training data and its prediction results
        plt.subplot(1,2,1)
        plt.plot(predicted, 'r')
        plt.title('predicted')

        plt.subplot(1,2,2)

        plt.plot(true, 'b')
        plt.title('true')
        plt.savefig(self.exp_dir+'training_predicted.png')
        plt.show()


        model.reset_states()


    # build identical model and predict sequence one step at a time
    def predict(self, warmup, preds,neurons, lr,train=False):
        """

        :param warmup: number of time steps to warm up h
        :param preds: number of timesteps to predict
        :return:
        """
        self.neurons=neurons
        self.lr=lr


        #build identical model and load its weights-- keras doesn't allow different batch size in train and test
        model=self.rebuild_model()


        #see training predictions
        if train:
            self.predict_training(model)


##########################################################################
        #predict validation:
        predictions=[]
        prev=np.zeros((1,1,1))
        print("start of prediction phase")
        for i in range(preds+warmup):
            p=None
            if i<warmup:
                p = model.predict(self.x_test[i].reshape(1,1,1))
                prev=p
            else:
                #prev=prev.reshape(1,1,1)
                p=model.predict(prev)
                prev=p

            predictions+=[p.squeeze()]

        #calculate rmse on test data
        m=self.mean
        s = self.std

        y=self.y_test[:preds+warmup].squeeze()

        true =np.array([((i*s)+m) for i in y])
        predicted=np.array([((i*s)+m) for i in predictions])


        error=predicted - true
        rmse=np.sqrt(np.mean(np.square(error)))
        self.file.write("warmup length: {} , predcictions: {}\n".format(warmup,preds))
        self.file.write('rmse on test data: {} '.format(rmse))
        print('rmse on test data: {} '.format(rmse))

        #plot results
        plt.plot(predicted)
        plt.plot(true)
        plt.legend(['predicted', 'true'])
        plt.savefig(self.exp_dir+'validationpredicted.png')
        plt.show()

        self.file.close()

        #return value of  rmse on validation data
        return rmse
###############################################

