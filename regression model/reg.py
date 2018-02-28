from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.utils import plot_model

class Regression:
    def __init__(self,config):
        self.config=config


    def build(self):
        model=Sequential()
        model = Sequential()
        model.add(Dense(32, input_shape=(4,)))
        model.add(Activation('relu'))

        model.add((Dense(64)))
        model.add(Activation('relu'))

        model.add((Dense(128)))
        model.add(Activation('relu'))

        model.add((Dense(64)))
        model.add(Activation('relu'))

        model.add(Dense(32))
        model.add(Activation('relu'))

        model.add(Dense(1))
        print(model.summary())
