# general imports
# privacy assessment: https://www.tensorflow.org/responsible_ai/privacy/tutorials/privacy_report
import numpy as np
from typing import Tuple
from scipy import special
from sklearn import metrics
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

# Set verbosity.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report
import tensorflow_privacy

import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
args = parser.parse_args([])

# set the value of num_classes manually
args.num_classes = 10

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

dataset = 'cifar10'
num_classes = 10
activation = 'relu'
num_conv = 3

batch_size=50
epochs_per_report = 2
total_epochs = 50

lr = 0.001

cnn_cifar =  CNNCifar(args)

cnn_cifar.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=['accuracy'])



def alexnet_cifar10(input_shape=(32, 32, 3), num_classes=10):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=num_classes, activation='softmax')
    ])
    return model
# Define the model
alexnet= alexnet_cifar10()

# Compile the model
alexnet.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=['accuracy'])


print('Loading the dataset.')
train_ds = tfds.as_numpy(
    tfds.load(dataset, split=tfds.Split.TRAIN, batch_size=-1))
test_ds = tfds.as_numpy(
    tfds.load(dataset, split=tfds.Split.TEST, batch_size=-1))
x_train = train_ds['image'].astype('float32') / 255.
y_train_indices = train_ds['label'][:, np.newaxis]
x_test = test_ds['image'].astype('float32') / 255.
y_test_indices = test_ds['label'][:, np.newaxis]

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train_indices, num_classes)
y_test = tf.keras.utils.to_categorical(y_test_indices, num_classes)

print('x_train', np.shape(x_train))
print('y_train', np.shape(y_train))

input_shape = x_train.shape[1:]

assert x_train.shape[0] % batch_size == 0, "The tensorflow_privacy optimizer doesn't handle partial batches"

def small_cnn(input_shape: Tuple[int],
              num_classes: int,
              num_conv: int,
              activation: str = 'relu') -> tf.keras.models.Sequential:
  """Setup a small CNN for image classification.

  Args:
    input_shape: Integer tuple for the shape of the images.
    num_classes: Number of prediction classes.
    num_conv: Number of convolutional layers.
    activation: The activation function to use for conv and dense layers.

  Returns:
    The Keras model.
  """
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Input(shape=input_shape))

  # Conv layers
  for _ in range(num_conv):
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=activation))
    model.add(tf.keras.layers.MaxPooling2D())

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(64, activation=activation))
  model.add(tf.keras.layers.Dense(num_classes))

  model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=['accuracy'])

  return model

model_2layers = small_cnn(
    input_shape, num_classes, num_conv=2, activation=activation)
model_3layers = small_cnn(
    input_shape, num_classes, num_conv=3, activation=activation)


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


all_reports = []
callback = PrivacyMetrics(epochs_per_report, "2 Layers")
history = model_2layers.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=total_epochs,
      validation_data=(x_test, y_test),
      callbacks=[callback],
      shuffle=True)

all_reports.extend(callback.attack_results)

callback = PrivacyMetrics(epochs_per_report, "3 Layers")
history = model_3layers.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=total_epochs,
      validation_data=(x_test, y_test),
      callbacks=[callback],
      shuffle=True)

all_reports.extend(callback.attack_results)

callback = PrivacyMetrics(epochs_per_report, "cnn_cifar")
history = cnn_cifar.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=total_epochs,
      validation_data=(x_test, y_test),
      callbacks=[callback],
      shuffle=True)

all_reports.extend(callback.attack_results)

callback = PrivacyMetrics(epochs_per_report, "alexnet")
history = alexnet.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=total_epochs,
      validation_data=(x_test, y_test),
      callbacks=[callback],
      shuffle=True)

all_reports.extend(callback.attack_results)


results = AttackResultsCollection(all_reports)
privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
epoch_plot = privacy_report.plot_by_epochs(
    results, privacy_metrics=privacy_metrics)
epoch_plot.savefig('results/figures/epoch_plot.png')

privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
utility_privacy_plot = privacy_report.plot_privacy_vs_accuracy(
    results, privacy_metrics=privacy_metrics)

for axis in utility_privacy_plot.axes:
  axis.set_xlabel('Validation accuracy')
utility_privacy_plot.savefig('results/figures/utility_privacy.png')
