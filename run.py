# from __future__ import absolute_import, division, print_function

import numpy as np
from helper import tf_basic_model

cnn_model = tf_basic_model()

train_input = np.genfromtxt('data/train.csv', delimiter=",", skip_header=1)
train_label, train_data = cnn_model.get_label_and_data(train_input)

train_data = cnn_model.pre_process_data(train_data)
cnn_model.train_model(train_data, train_label)

test_data = np.genfromtxt('data/test.csv', delimiter=",", skip_header=1)
test_data = cnn_model.pre_process_data(test_data)

predictions = cnn_model.predict_model(test_data)

cnn_model.submit_prediction(predictions)
