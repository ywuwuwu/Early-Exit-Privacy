import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple
from scipy import special
from sklearn import metrics
import os
import matplotlib.pyplot as plt
# Set verbosity.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report
# from PrivacyMetrics import PrivacyMetrics
import tensorflow_privacy

import argparse


from branchyAlex import AlexNet
from branchyAlex import branchy_AlexNet
from branchyAlex import branchy_AlexNet_inference

from branchyResnet import ResNet18

from trainer import training_loop



parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
args = parser.parse_args([])

# set the value of num_classes manually
args.num_classes = 10
args.threshold = 0.7

dataset = 'cifar10'
num_classes = 10
activation = 'relu'
num_conv = 3

batch_size= 50 
epochs_per_report = 1
total_epochs = 25 ## 25

lr = 0.001 # has to be 0.001 for cnn model, cant change it to 0.01

# Load the CIFAR-10 dataset

print('Loading the dataset.')
train_ds = tfds.as_numpy(
    tfds.load(dataset, split=tfds.Split.TRAIN, batch_size=-1))
test_ds = tfds.as_numpy(
    tfds.load(dataset, split=tfds.Split.TEST, batch_size=-1))
x_train = train_ds['image'].astype('float32') / 255.
y_train_indices = train_ds['label'][:, np.newaxis]
x_test = test_ds['image'].astype('float32') / 255.
y_test_indices = test_ds['label'][:, np.newaxis]

y_train_one_hot = tf.one_hot(y_train_indices[:, 0], depth=num_classes)
y_test_one_hot = tf.one_hot(y_test_indices[:, 0], depth=num_classes)

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train_indices, num_classes)
y_test = tf.keras.utils.to_categorical(y_test_indices, num_classes)

print('x_train', np.shape(x_train))
print('y_train', np.shape(y_train))

input_shape = x_train.shape[1:]

assert x_train.shape[0] % batch_size == 0, "The tensorflow_privacy optimizer doesn't handle partial batches"

all_reports = []
epochs_per_report = 1

class PrivacyMetrics(tf.keras.callbacks.Callback):
  def __init__(self, epochs_per_report, model_name, model):
    self.epochs_per_report = epochs_per_report
    self.model_name = model_name
    self.attack_results = []
    self.model = model# early exit model

  def on_epoch_end(self, epoch, logs=None):
    epoch = epoch+1

    if epoch % self.epochs_per_report != 0:
      return

    print(f'\nRunning privacy report for epoch: {epoch}\n')

# for early exit
    logits_train1 = self.model.predict(x_train, batch_size=batch_size)
    logits_test1 = self.model.predict(x_test, batch_size=batch_size)

    prob_train1 = special.softmax(logits_train1, axis=1)
    prob_test1 = special.softmax(logits_test1, axis=1)

    # Add metadata to generate a privacy report.
    privacy_report_metadata = PrivacyReportMetadata(
        # Show the validation accuracy on the plot
        # It's what you send to train_accuracy that gets plotted.
        accuracy_train=logs['val_accuracy'], 
        accuracy_test=logs['val_accuracy'],
        epoch_num=epoch,
        model_variant_label=self.model_name)

    # returns a dictionary containing the results of the attacks, including the attack success rate and ROC curve data.
    attack_results = mia.run_attacks(
        AttackInputData(
            labels_train=np.argmax(y_train_one_hot.numpy(), axis=-1),  # Convert to NumPy array and back to integer format
            labels_test=np.argmax(y_test_one_hot.numpy(), axis=-1),  # Convert to NumPy array and back to integer format,
            probs_train=prob_train1,
            probs_test=prob_test1),
        SlicingSpec(entire_dataset=True, by_class=True),
        attack_types=(AttackType.THRESHOLD_ATTACK,
                      AttackType.LOGISTIC_REGRESSION),
        privacy_report_metadata=privacy_report_metadata)

    self.attack_results.append(attack_results)


# Define optimizer
optimizer = tf.keras.optimizers.Adam()
# Define accuracy metric
metric = tf.keras.metrics.SparseCategoricalAccuracy()
metric_test = tf.keras.metrics.SparseCategoricalAccuracy()

# Create the BranchyNet Resnet model



# # Define optimizer
# optimizer = tf.keras.optimizers.Adam()
# # Define accuracy metric
# metric = tf.keras.metrics.SparseCategoricalAccuracy()
# metric_test = tf.keras.metrics.SparseCategoricalAccuracy()

# # Create the BranchyNet Alexnet model
# model_balex = branchy_AlexNet(args)
# model_balex_inference = branchy_AlexNet_inference(args)
# input_shape = (1,32, 32, 3)
# model_balex.build(input_shape)
# model_balex_inference.build(input_shape)
# callback = PrivacyMetrics(epochs_per_report, "branchy_alex", model_balex_inference)

# callback = training_loop(x_train, y_train, x_test,y_test,callback, model_balex,model_balex_inference,optimizer, metric, metric_test, batch_size = batch_size, total_epochs = total_epochs)

# all_reports.extend(callback.attack_results)



# ###
# ### change to another privacy metrics 
# ###
class PrivacyMetrics(tf.keras.callbacks.Callback):
  def __init__(self, epochs_per_report, model_name):
    self.epochs_per_report = epochs_per_report
    self.model_name = model_name
    self.attack_results = []

  def on_epoch_end(self, epoch, logs=None):
    epoch = epoch+1

    if epoch % self.epochs_per_report != 0:
      return

    print(f'\nRunning privacy report for epoch: {epoch}\n')

    logits_train = self.model.predict(x_train, batch_size=batch_size)
    logits_test = self.model.predict(x_test, batch_size=batch_size)

    prob_train = special.softmax(logits_train, axis=1)
    prob_test = special.softmax(logits_test, axis=1)

    # Add metadata to generate a privacy report.
    privacy_report_metadata = PrivacyReportMetadata(
        # Show the validation accuracy on the plot
        # It's what you send to train_accuracy that gets plotted.
        accuracy_train=logs['val_accuracy'], 
        accuracy_test=logs['val_accuracy'],
        epoch_num=epoch,
        model_variant_label=self.model_name)

    attack_results = mia.run_attacks(
        AttackInputData(
            labels_train=y_train_indices[:, 0],
            labels_test=y_test_indices[:, 0],
            probs_train=prob_train,
            probs_test=prob_test),
        SlicingSpec(entire_dataset=True, by_class=True),
        attack_types=(AttackType.THRESHOLD_ATTACK,
                      AttackType.LOGISTIC_REGRESSION),
        privacy_report_metadata=privacy_report_metadata)

    self.attack_results.append(attack_results)


# Alexnet =  AlexNet(args)
# print('Alexnet')
# Alexnet.compile(
#     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#     optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum = 0.9),
#     metrics=['accuracy'])

# callback = PrivacyMetrics(epochs_per_report, "AlexNet")
# history = Alexnet.fit(
#       x_train,
#       y_train,
#       batch_size=batch_size,
#       epochs=total_epochs,
#       validation_data=(x_test, y_test),
#       callbacks=[callback],
#       shuffle=True)
# all_reports.extend(callback.attack_results)

Resnet18 =  ResNet18(args)
print('Resnet')
Resnet18.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum = 0.9),
    metrics=['accuracy'])

callback = PrivacyMetrics(epochs_per_report, "Resnet18")
history = Resnet18.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=total_epochs,
      validation_data=(x_test, y_test),
      callbacks=[callback],
      shuffle=True)
all_reports.extend(callback.attack_results)


###
###
###

results = AttackResultsCollection(all_reports)
privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
epoch_plot = privacy_report.plot_by_epochs(
    results, privacy_metrics=privacy_metrics)
epoch_plot.savefig('results/branchy/epoch_plot.png')

privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
utility_privacy_plot = privacy_report.plot_privacy_vs_accuracy(
    results, privacy_metrics=privacy_metrics)

for axis in utility_privacy_plot.axes:
  axis.set_xlabel('Test accuracy')
utility_privacy_plot.savefig('results/figures/utility_privacy.png')
