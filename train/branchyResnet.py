from collections import defaultdict

import tensorflow as tf
import numpy as np
from scipy import special

import tensorflow as tf
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
args = parser.parse_args([])

# set the value of num_classes manually
args.num_classes = 10
args.threshold = 0.7

import tensorflow as tf

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.residual_conv = tf.keras.layers.Conv2D(filters, (1, 1), strides=stride, padding='same')
            self.residual_bn = tf.keras.layers.BatchNormalization()
        else:
            self.residual_conv = None

        self.add = tf.keras.layers.Add()
        self.relu2 = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.residual_conv is not None:
            residual = self.residual_conv(inputs)
            residual = self.residual_bn(residual)
        else:
            residual = inputs

        x = self.add([x, residual])
        x = self.relu2(x)

        return x

class ResNet18(tf.keras.Model):
    def __init__(self, args, input_shape=(32, 32, 3), **kwargs):
        super().__init__(**kwargs)
        self.inputs_layer = tf.keras.Input(shape=input_shape)
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.block1 = ResNetBlock(64, 1)
        self.block2 = ResNetBlock(64, 1)
        self.block3 = ResNetBlock(128, 2)
        self.block4 = ResNetBlock(128, 1)
        self.block5 = ResNetBlock(256, 2)
        self.block6 = ResNetBlock(256, 1)
        self.block7 = ResNetBlock(512, 2)
        self.block8 = ResNetBlock(512, 1)
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(args.num_classes)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.pooling(x)
        x = self.output_layer(x)

        return x


class branchy_ResNet18_inference(tf.keras.Model):
    def __init__(self, args, input_shape=(32, 32, 3), **kwargs):
        super().__init__(**kwargs)
        self.inputs_layer = tf.keras.Input(shape=input_shape)
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.block1 = ResNetBlock(64, 1)
        self.block2 = ResNetBlock(64, 1)
        self.block3 = ResNetBlock(128, 2)
        self.block4 = ResNetBlock(128, 1)
        self.block5 = ResNetBlock(256, 2)
        self.block6 = ResNetBlock(256, 1)
        self.block7 = ResNetBlock(512, 2)
        self.block8 = ResNetBlock(512, 1)
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(args.num_classes)
        # Define the branches
        self.branches = []
        branch1 = tf.keras.Sequential([tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), activation='relu',padding='same'),
                                        tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.MaxPooling2D((2,2), padding='same'),
                                       tf.keras.layers.Flatten(),
                                       tf.keras.layers.Dropout(0.5),
                                       tf.keras.layers.Dense(args.num_classes)])
        self.branches.append(branch1)
        self.threshold = args.threshold

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

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
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.pooling(x)
        x = self.output_layer(x)
        return x

class branchy_ResNet18(tf.keras.Model):
    def __init__(self, args, input_shape=(32, 32, 3), **kwargs):
        super().__init__(**kwargs)
        self.inputs_layer = tf.keras.Input(shape=input_shape)
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.block1 = ResNetBlock(64, 1)
        self.block2 = ResNetBlock(64, 1)
        self.block3 = ResNetBlock(128, 2)
        self.block4 = ResNetBlock(128, 1)
        self.block5 = ResNetBlock(256, 2)
        self.block6 = ResNetBlock(256, 1)
        self.block7 = ResNetBlock(512, 2)
        self.block8 = ResNetBlock(512, 1)
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(args.num_classes)
        # Define the branches
        self.branches = []
        branch1 = tf.keras.Sequential([tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), activation='relu',padding='same'),
                                        tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.MaxPooling2D((2,2), padding='same'),
                                       tf.keras.layers.Flatten(),
                                       tf.keras.layers.Dropout(0.5),
                                       tf.keras.layers.Dense(args.num_classes)])
        self.branches.append(branch1)
        self.threshold = args.threshold

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        branch1 = self.branches[0]
        branch1_output = branch1(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.pooling(x)
        x = self.output_layer(x)
        exits = [branch1_output, x]
        return exits
