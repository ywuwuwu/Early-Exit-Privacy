from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import copy
import random
import numpy as np
import os
import collections


from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report
import tensorflow_privacy

from PrivacyMetrics import PrivacyMetrics
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
args = parser.parse_args([])

# set the value of num_classes manually
args.num_classes = 10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# # Split data into non-IID subsets for each client
# num_clients = 10
# client_data = {}
# for i in range(num_clients):
#     x_subset = x_train[i*5000:(i+1)*5000]
#     y_subset = y_train[i*5000:(i+1)*5000]
#     client_data[i] = (x_subset, y_subset)


# Split data into IID subsets for each client
num_clients = 10
client_data = {}
for i in range(num_clients):
    # Randomly select 500 samples for each class
    x_subset = []
    y_subset = []
    for class_label in range(10):
        x_class = x_train[y_train[:,0]==class_label]
        y_class = y_train[y_train[:,0]==class_label]
        idxs = tf.squeeze(tf.where(tf.equal(y_class[:,0], class_label)))
        idxs = tf.random.shuffle(idxs)[:500]
        x_subset.append(tf.gather(x_class, idxs))
        y_subset.append(tf.gather(y_class, idxs))
    x_subset = tf.concat(x_subset, axis=0)
    y_subset = tf.concat(y_subset, axis=0)
    client_data[i] = (x_subset, y_subset)

print('x_shape', np.shape(x_subset))
print('y_shape', np.shape(y_subset))
# for i in range(10):
#     client_data 
# label distribution

class CNNCifar(tf.keras.Model):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(6, (5, 5), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc2 = tf.keras.layers.Dense(84, activation='relu')
        self.fc3 = tf.keras.layers.Dense(args.num_classes)

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)  # cast input to float32
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Define client update function
def client_update(model, x, y):

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(x, y, epochs=1, verbose=1)

    weights = model.get_weights()
    loss = history.history['loss'][-1]
    accuracy = np.mean(history.history['accuracy'])
    return weights, loss, accuracy

# Define server aggregation function


def server_aggregate(global_model_weights, local_models_weights, num_clients):
    # Compute the weighted average of the local model weights
    global_weights = []
    for i in range(len(global_model_weights)):
        weight_sum = np.zeros_like(global_model_weights[i])
        for j in range(num_clients):
            weight_sum += local_models_weights[j][i]
        global_weight = weight_sum / num_clients
        global_weights.append(global_weight)
    return global_weights


# Define federated learning parameters
epochs_per_round = 1
num_rounds = 10
batch_size = 32
learning_rate = 0.1
loss_train = []


# Initialize global model parameters

global_model = CNNCifar(args)
global_model.build((None,) + x_subset.shape[1:])

privacy_callback = PrivacyMetrics(1)
all_reports = []

# Train the model using federated learning
for i in range(num_rounds):
    print("Round {}/{}".format(i+1, num_rounds))
    local_model_weight = []
    loss_locals = []
    global_weights = global_model.get_weights()
    for j in range(num_clients):
        # Get client data
        x, y = client_data[j]
        # Perform local training on client data and get updated model weights
        local_model = CNNCifar(args)
        # weights, loss, acc = client_update(local_model, x, y, learning_rate)
        local_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        local_model.build((None,) + x.shape[1:])
        local_model.set_weights(global_weights)
        history = local_model.fit(x, y, epochs=1, verbose=1)
        loss = history.history['loss'][-1]
        accuracy = np.mean(history.history['accuracy'])

        local_model_weight.append(local_model.get_weights())
        loss_locals.append(loss)
        
    # Aggregate local weights to get global weights
    global_weights = server_aggregate(global_weights, local_model_weight, num_clients)
    global_model.set_weights(global_weights)

    # privacy assessment
    privacy_callback.on_epoch_end(round) # round should be i

    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(i+1, loss_avg))
    loss_train.append(loss_avg)
    global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _, accuracy = global_model.evaluate(x_train, y_train, verbose=0)
    print('Round {:3d}, Test Accuracy {:.3f}'.format(i+1, accuracy))
# Evaluate trained model on test dataset

loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
global_model.save_weights('results/models/cnn_cifar10weights')

## privacy assessment
all_reports.extend(privacy_callback.attack_results)
results = AttackResultsCollection(all_reports)

privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
epoch_plot = privacy_report.plot_by_epochs(results, privacy_metrics=privacy_metrics)
epoch_plot.savefig('results/figures/tf_FL_privacy_report.png')