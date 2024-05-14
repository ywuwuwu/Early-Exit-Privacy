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
from tensorflow.keras.optimizers.schedules import ExponentialDecay


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

import warnings
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse

from branchyAlex import AlexNet
from branchyAlex import branchy_AlexNet
from branchyAlex import branchy_AlexNet_inference

from trainer import training_loop
from trainer_fed import server_aggregate
# from trainer_fed import client_update

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

batch_size = 50 
epochs_per_report = 1
total_epochs = 50 # 25

lr = 0.001 # has to be 0.001 for cnn model, cant change it to 0.01
# # List available devices
# devices = tf.config.list_physical_devices()

# # Check if GPU is available
# gpu_devices = tf.config.list_physical_devices('GPU')
# print("GPU(s) available:", gpu_devices)
# tf.config.set_visible_devices(gpu_devices[1], 'GPU')
# print(f"Using GPU {1}: {gpu_devices[1].name}")

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
################

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
################

# class PrivacyMetrics(tf.keras.callbacks.Callback):
#   def __init__(self, epochs_per_report, model_name, model):
#     self.epochs_per_report = epochs_per_report
#     self.model_name = model_name
#     self.attack_results = []
#     self.model = model# early exit model

#   def on_epoch_end(self, epoch, logs=None):
#     epoch = epoch+1

#     if epoch % self.epochs_per_report != 0:
#       return

#     print(f'\nRunning privacy report for epoch: {epoch}\n')

# # for early exit
#     logits_train1 = self.model.predict(x_train, batch_size=batch_size)
#     logits_test1 = self.model.predict(x_test, batch_size=batch_size)

#     prob_train1 = special.softmax(logits_train1, axis=1)
#     prob_test1 = special.softmax(logits_test1, axis=1)

#     # Add metadata to generate a privacy report.
#     privacy_report_metadata = PrivacyReportMetadata(
#         # Show the validation accuracy on the plot
#         # It's what you send to train_accuracy that gets plotted.
#         accuracy_train=logs['val_accuracy'], 
#         accuracy_test=logs['val_accuracy'],
#         epoch_num=epoch,
#         model_variant_label=self.model_name)

#     # returns a dictionary containing the results of the attacks, including the attack success rate and ROC curve data.
#     attack_results = mia.run_attacks(
#         AttackInputData(
#             labels_train=np.argmax(y_train_one_hot.numpy(), axis=-1),  # Convert to NumPy array and back to integer format
#             labels_test=np.argmax(y_test_one_hot.numpy(), axis=-1),  # Convert to NumPy array and back to integer format,
#             probs_train=prob_train1,
#             probs_test=prob_test1),
#         SlicingSpec(entire_dataset=True, by_class=True),
#         attack_types=(AttackType.THRESHOLD_ATTACK,
#                       AttackType.LOGISTIC_REGRESSION),
#         privacy_report_metadata=privacy_report_metadata)

#     self.attack_results.append(attack_results)


# Define optimizer
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum = 0.9)
# # Define accuracy metric
# metric = tf.keras.metrics.SparseCategoricalAccuracy()
# metric_test = tf.keras.metrics.SparseCategoricalAccuracy()

# # Create the BranchyNet Alexnet model
# model_balex = branchy_AlexNet(args)
# model_balex_inference = branchy_AlexNet_inference(args) # gloabal model
# input_shape = (None, 32, 32, 3)
# model_balex.build(input_shape)
# model_balex_inference.build(input_shape)
# callback = PrivacyMetrics(epochs_per_report, "branchy_alex", model_balex_inference) # change it to global model_inference

# callback = training_loop(x_train, y_train, x_test,y_test,callback, model_balex,model_balex_inference,optimizer, metric, metric_test, batch_size = batch_size, total_epochs = total_epochs, lr = lr)

# all_reports.extend(callback.attack_results)

class PrivacyMetrics(tf.keras.callbacks.Callback):
  def __init__(self, epochs_per_report, model_name, model):
    self.epochs_per_report = epochs_per_report
    self.model_name = model_name
    self.attack_results = []
    self.model = model# global model without early exit

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

# Initialize global model parameters

global_model = AlexNet(args)
print('Alexnet')
global_model.build((None,) + x_subset.shape[1:])
loss_train = []
callback = PrivacyMetrics(epochs_per_report, "AlexNet", global_model)
metric_test = tf.keras.metrics.SparseCategoricalAccuracy()

for i in range(total_epochs):
    print("Round {}/{}".format(i+1, total_epochs))
    local_model_weight = []
    loss_locals = []
    metric_test.reset_states()
    test_loss = 0
    global_weights = global_model.get_weights()
    for j in range(num_clients):
      # Get client data
      x, y = client_data[j]
      # print('x = ', np.shape(x))
      # print('y = ', np.shape(y))
      y = np.argmax(y, axis=-1)
      # Perform local training on client data and get updated model weights
      local_model = AlexNet(args)
      local_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum = 0.9),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
      local_model.build((None,) + x.shape[1:])
      local_model.set_weights(global_weights)
      history = local_model.fit(x, y, epochs=1, verbose=1)
      loss = history.history['loss'][-1]
      accuracy = np.mean(history.history['accuracy'])

      local_model_weight.append(local_model.get_weights())
      loss_locals.append(loss)
    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(i+1, loss_avg))
    # loss_train.append(loss_avg)        
    # Aggregate local weights to get global weights
    global_weights = server_aggregate(global_weights, local_model_weight, num_clients)
    global_model.set_weights(global_weights)

###### global model evaluation

    # global_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum = 0.9),
    #                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                      metrics=['accuracy'])
    # y_test0 = np.argmax(y_test, axis=-1)
    # test_loss, test_accuracy = global_model.evaluate(x_test, y_test0, verbose=0)
    for batch in range(0, len(x_test), batch_size):
      x_test_batch = x_test[batch:batch+batch_size]
      y_test_batch = y_test[batch:batch+batch_size]
      global_model_output_test = global_model(x_test_batch)
      test_loss0 = tf.keras.losses.categorical_crossentropy(y_test_batch, global_model_output_test,  from_logits = True)
      for loss in test_loss0:
        test_loss += loss
      y_test_batch = tf.argmax(y_test_batch, axis=1)
      metric_test.update_state(y_test_batch, global_model_output_test)    

    # print('Round {:3d}, Test Accuracy {:.3f}'.format(i+1, test_accuracy))
    # privacy assessment
    logs = {'loss': test_loss.numpy()/ len(x_test), 'val_accuracy': metric_test.result().numpy()}
    print(logs)
    # Call the on_epoch_end method with the logs dictionary
    callback.on_epoch_end(i, logs=logs)

all_reports.extend(callback.attack_results)

results = AttackResultsCollection(all_reports)
privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
epoch_plot = privacy_report.plot_by_epochs(
    results, privacy_metrics=privacy_metrics)
epoch_plot.savefig('results/branchy/fed_alex_epoch_plot.pdf')

privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
utility_privacy_plot = privacy_report.plot_privacy_vs_accuracy(
    results, privacy_metrics=privacy_metrics)

for axis in utility_privacy_plot.axes:
  axis.set_xlabel('Test accuracy')
utility_privacy_plot.savefig('results/figures/fed_alex_utility_privacy.pdf')
