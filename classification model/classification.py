from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.utils import plot_model


class Classifier:
    def __init__(self, config):
        # load configuration variables passed from file
        self.config = config

    def build(self):
        """
        any dropout, regularization can be foud in self.config
        reg= self.config['reg']
        dropout= self.config['dropout']

        :return:model object
        """
        input_dim=self.config['inp_shape']
        num_classes=self.config['num_classes']
        reg=self.config['reg']
        dropout=self.config['dropout']

        model = Sequential()
        model.add(Dense(128, input_shape=((input_dim,)),kernel_regularizer=regularizers.l2(reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add((Dense(1024,kernel_regularizer=regularizers.l2(reg))))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add((Dense(512,kernel_regularizer=regularizers.l2(reg))))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add((Dense(512,kernel_regularizer=regularizers.l2(reg))))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add((Dense(256,kernel_regularizer=regularizers.l2(reg))))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # model.add((Dense(512,kernel_regularizer=regularizers.l2(reg))))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        #
        # # model.add((Dense(1024,kernel_regularizer=regularizers.l2(reg))))
        # # model.add(BatchNormalization())
        # # model.add(Activation('relu'))
        #
        # model.add((Dense(512,kernel_regularizer=regularizers.l2(reg))))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        # model.add((Dense(256,kernel_regularizer=regularizers.l2(reg))))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        #
        # model.add((Dense(128,kernel_regularizer=regularizers.l2(reg))))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        #
        # model.add((Dense(64,kernel_regularizer=regularizers.l2(reg))))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        #
        # model.add(Dense(32,kernel_regularizer=regularizers.l2(reg)))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        model.add(Dense(num_classes,kernel_regularizer=regularizers.l2(reg)))
        model.add(Activation('softmax'))

        file_path=self.config['exp_dir']+'model_summary.txt'
        with open(file_path,'w+') as s:
            model.summary()
            model.summary(print_fn=lambda x: s.write(x + '\n'))

        return model

