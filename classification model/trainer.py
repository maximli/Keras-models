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
        #all features just for clarification
        feat = {0:'Avg_CQI',1:'Avg_UE_number',2:'CCE_Util',3:'Avg_MCS',4:'PRB_Util',
                5:'DL_Traffic_Volume',6:'MCS(0-9)',7:'MCS(10-16)',8:'MCS(17-28)',9:'DL_IBLER',
                10:'TM4',11:'Max_Pwr',12:'PRB_Avail',13:'Avg_Active_Users',14:'Max_Active_Users'
                }

        data_dir = self.config['data_dir']
        print("reading data from: ",data_dir)

        x_train = np.load(data_dir + "x_train.npy")
        x_val = np.load(data_dir + "x_val.npy")
        x_test = np.load(data_dir + "x_test.npy")

        y_train = np.load(data_dir + "y_train.npy")
        y_val = np.load(data_dir + "y_val.npy")
        y_test = np.load(data_dir + "y_test.npy")


        x_train,y_train=u.shuflle_data(x_train,y_train)
        x_val,y_val=u.shuflle_data(x_val,y_val)
        x_test,y_test=u.shuflle_data(x_test,y_test)

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

        #features that we will use
        idx=self.config['data_idx']

        #choose subset of data to overfit
        # idxt=np.random.choice(x_train.shape[0],100,replace=False)
        # idxv=np.random.choice(x_val.shape[0],10,replace=False)
        #
        # x_train=x_train[idxt,:]
        # y_train=y_train[idxt]
        #
        # x_val=x_val[idxv,:]
        # y_val=y_val[idxv]

        # save data to trainer
        self.x_train = x_train[:,idx]
        self.y_train = y_train

        self.x_val = x_val[:,idx]
        self.y_val = y_val

        self.x_test = x_test[:,idx]
        self.y_test = y_test



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

        self.model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("saving model to : ", exp_dir)
        self.model.save(exp_dir)

        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), shuffle=True,
                       batch_size=batch_size, callbacks=[tensorboard, checkpoint], epochs=epochs)

        self.model.load_weights(checkpoint_dir)

        # scores = self.model.evaluate(self.x_val, self.y_val) #contains loss and metric(accuracy)
        # print("best validation accuracy before saving")
        # print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))



        print("saving model to : ", exp_dir)
        self.model.save(exp_dir)


    def validate(self):
        #load model and its weights and run validation accuracy

        exp_dir = self.config['exp_dir'] + "model.h5"
        checkpoint_file = self.config['checkpoint_dir'] + "weights-best.hdf5"

        print("model path : ", exp_dir)
        print("weights path : ", checkpoint_file)

        # load moadel and best checkpoint
        m_load = load_model(exp_dir)
        m_load.load_weights(checkpoint_file)

        print("loaded model and weights successfully")

        # train accuracy

        training_acc=m_load.evaluate(self.x_train,self.y_train)
        print("training accuracy of loaded model")
        print("%s: %.2f%%" % (m_load.metrics_names[1], training_acc[1] * 100))

        #validation accuracy

        scores = m_load.evaluate(self.x_val, self.y_val)

        predictions = m_load.predict(self.x_val)
        predicted_validations = np.argmax(predictions, axis=1)
        print("labels_shape", self.y_val.shape)
        print("predicted_labels shape", predicted_validations.shape)
        print("validation accuracy of loaded model")
        print("%s: %.2f%%" % (m_load.metrics_names[1], scores[1] * 100))

        correct=np.equal(predicted_validations,self.y_val).astype(float)

        print('calculated accuracy', np.mean(correct))
        print("-----------------------------------")

    def test(self):
        #loads best model and resores graph
        exp_dir = self.config['exp_dir'] + "model.h5"
        checkpoint_file = self.config['checkpoint_dir'] + "weights-best.hdf5"

        print("model path : ", exp_dir)
        print("weights path : ", checkpoint_file)

        # load moadel and best checkpoint
        m_load = load_model(exp_dir)
        m_load.load_weights(checkpoint_file)

        print("loaded model and weights successfully")

        # test acuracy
        scores = m_load.evaluate(self.x_test, self.y_test)
        predictions = m_load.predict(self.x_test)
        predicted_labels = np.argmax(predictions, axis=1)
        print("labels_shape", self.y_test.shape)
        print("predicted_labels shape", predicted_labels.shape)
        print("test accuracy of loaded model")
        print("%s: %.2f%%" % (m_load.metrics_names[1], scores[1] * 100))

        print("--------------------------")


        return (self.y_test, predicted_labels)




def main():
    # read args
    args = u.read_args()
    u.create_directories(args)

    #create classification model
    c=Classifier(args)

    #if training flag is true build model and train it
    if args['train']:

        model = c.build()
        plot_model(model, to_file=args['exp_dir']+'modelimage'+'.png' ,show_layer_names=False , show_shapes=False)
        operator = Train(model, args)
        operator.train()
        operator.validate()

    #if test is true, load best model and test it
    if args['test']:
        #load data only without creating model
        operator = Train(None, args)
        operator.validate()
        true, predicted = operator.test()

        #plot confusion matrix
        class_names = ['0', '1']
        cf = confusion_matrix(true, predicted)
        plt.figure()
        u.plot_confusion_matrix(cf, classes=class_names, normalize=False,
                            title='Confusion matrix, without normalization')


main()
