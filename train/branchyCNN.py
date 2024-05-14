from collections import defaultdict

import tensorflow as tf
import numpy as np
from scipy import special

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


class branchy_CNNCifar(tf.keras.Model):
    def __init__(self, args):
        super(branchy_CNNCifar, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(6, (5, 5), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc2 = tf.keras.layers.Dense(84, activation='relu')
        self.fc3 = tf.keras.layers.Dense(args.num_classes)

        # Define the branches
        self.branches = []

        branch1 = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dense(args.num_classes)])

        self.branches.append(branch1)

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)  # cast input to float32
        x = self.pool(self.conv1(x))
        # print('x_shape_conv1', np.shape(x))
        x = self.pool(self.conv2(x))
        # print('x_shape_conv2', np.shape(x))
        x = self.flatten(x)
        branch = self.branches[0]
        # print(branch)
        branch1_output = branch(x)
        # print('branch1_output', np.shape(branch1_output))
        
        x = self.fc1(x)
        # print('fc1', np.shape(x))
        x = self.fc2(x)
        # print('fc2', np.shape(x))
        x = self.fc3(x)
        # print('final_output', np.shape(x))

        exits = [branch1_output, x]
        return exits
        # return final_output, branch1_output

class branchy_CNNCifar_inference(tf.keras.Model):
    def __init__(self, args):
        super(branchy_CNNCifar_inference, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(6, (5, 5), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc2 = tf.keras.layers.Dense(84, activation='relu')
        self.fc3 = tf.keras.layers.Dense(args.num_classes)
        

        # Define the branches
        self.branches = []

        branch1 = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dense(args.num_classes)])

        self.branches.append(branch1)
        self.threshold = args.threshold

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)  # cast input to float32
        x = self.pool(self.conv1(x))
        # print('x_shape_conv1', np.shape(x))
        x = self.pool(self.conv2(x))
        # print('x_shape_conv2', np.shape(x))
        x = self.flatten(x)
        
        branch = self.branches[0]
        # print(branch)
        branch1_output = branch(x)
        probability = tf.reduce_max(tf.nn.softmax(branch1_output))


        output = tf.cond(probability > self.threshold,
                         true_fn=lambda: self.compute_branch1_output(branch1_output),
                         false_fn = lambda: self.compute_final_output(x))

        return output

        # return final_output, branch1_output
    def compute_branch1_output(self, branch1_output):
        return branch1_output

    def compute_final_output(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x