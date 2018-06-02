from __future__ import absolute_import, division, print_function

import numpy as np

import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
import pickle


class tf_basic_model:
    def __init__(self):
        self.model_filename = "./model/captcha_model.hdf5"
        self.model_lables_filename = "./model/model_labels.dat"

    def get_label_and_data(self, train_data):
        train_label = train_data[:, 0]
        train_data = np.delete(train_data, [0], axis=1)
        return train_label, train_data

    def pre_process_data(self, data):
        data = np.reshape(data, (data.shape[0], 28, 28))
        new_dim_data = np.expand_dims(data, axis=3)
        return np.array(new_dim_data, dtype='float') / 255.0

    def train_model(self, data, labels):
        (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
        lb = LabelBinarizer().fit(Y_train)
        Y_train = lb.transform(Y_train)
        Y_test = lb.transform(Y_test)
        with open(self.model_lables_filename, "wb") as f:
            pickle.dump(lb, f)

        model = Sequential()

        model.add(Conv2D(20, (5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500, activation="relu"))

        model.add(Dense(10, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

        model.save(self.model_filename)

    def predict_model(self, test_data):
        with open(self.model_lables_filename, "rb") as f:
            lb = pickle.load(f)

        model = load_model(self.model_filename)
        prediction = model.predict(test_data)
        prediction_results = lb.inverse_transform(prediction)

        return prediction_results.astype(int)

    def submit_prediction(self, predictions, filename=None):
        if filename is None:
            filename = 'submission'

        submission = pd.DataFrame()
        submission['Label'] = predictions
        submission['ImageId'] = submission.index + 1
        submission = submission.reindex(columns=['ImageId', 'Label'])
        submission.head()
        submission.to_csv('./data/' + filename + '.csv', index=False)
