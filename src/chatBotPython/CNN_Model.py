from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from .trainingData import TrainingData
import numpy as np


class CNN(TrainingData):

    def __init__(self):
        super(CNN, self).__init__()
        self._model = Sequential()
        self.create_model()

    def create_model(self):

        train_x, train_y = self.get_training_data()
        self.add_layers(train_x, train_y)
        sgd = self._SGD()
        self._model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self._hist = self._hist(train_x, train_y)


    def add_layers(self, train_x, train_y):

        self._model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(64, activation='relu'))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(len(train_y[0]), activation='softmax'))

    def _SGD(self):
        """ Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model """
        return SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    def _hist(self, train_x, train_y):
        return self._model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    def write_results(self, filename='chatbot.h5'):
        self._model.save(filename, self._hist)  # we will pickle this model to use in the future


