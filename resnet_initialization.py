#================================================
# Resnet 50 for initialization
#================================================
# UNAM IIMAS
# Author: Ivette Velez
# Tutor:  Caleb Rascon
# Co-tutor: Gibran Fuentes
#
# This model is programmed to verify if two signals are from the same speaker
# To run the model: 
# 1. Be sure that the arguments of the model are correct and that the name of the files have the audios required
# 2. To properly run the model it is possible to configure all the parameters on the this file (as default option) and just write:
#   python name_of_the_file
#
#
# ==============================================================================
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# Loading all the needed libraries
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import os
import sys
import time
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
from scipy import ndimage
from scipy import signal
import sklearn.preprocessing as pps
from sklearn.preprocessing import StandardScaler

from tensorflow.contrib import layers

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Some Graphic card have more than one GPU, to use just one you write the next line
# with 0 for the GPU 0 and 1 for the GPU 1
# Using just one GPU in case of GPU 
os.environ['CUDA_VISIBLE_DEVICES']= '0'

# Initializing the variable of the parameters
FLAGS = None

# Files for the model
FILE_TRAIN = 'train_speakers_initialization.txt'
FILE_VALID = 'valid_speakers.txt'
FILE_TEST = 'test_speakers.txt'
FILE_LIST = 'list_train_speakers.txt'

# Creating the classes for the multitask resnet
input_file  = open(FILE_LIST,'r')
permited_speakers = 100
list_speakers = []
for line in input_file:
  row = line.rstrip()
  list_speakers.append(row)

list_speakers = np.array(list_speakers, dtype = np.int).tolist()

# Parameter for the batch normalization with relu for the tf.layers.batch_normalization
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# ==============================================================================
# Constants on the model
# ==============================================================================
WINDOW = 1*16000 # with audios of librispeech it means 1 seconds
PART = 2
MS = 1.0/16000
NPERSEG = int(0.025/MS)
NOVERLAP = int(0.015/MS)
NFFT =NPERSEG
SIZE_FFT = 32
SIZE_COLS = int(np.ceil((WINDOW - NPERSEG)/(NPERSEG - NOVERLAP)))

# Indicating the amount of data to be generated per epoch for train, valid and test
TOTAL_DATA_TRAIN = 12277*2
TOTAL_DATA_VALID = TOTAL_DATA_TRAIN*0.1
TOTAL_DATA_TEST = TOTAL_DATA_TRAIN*0.1

# VAD threshold used to actually consider the audio as someone talking
VAD = 0.05

# ==============================================================================
# Configuration parameters for the architecture
# ==============================================================================
# Number of labels in the last layer, in this case is two for the results [0,1] or [1,0]
L_LABEL = 2
L_CLASS = permited_speakers

# Size of the entrance input data
IN_HEIGHT = SIZE_FFT
IN_WIDTH = SIZE_COLS

# Number of channels in the input data
CHANNELS = 1

# Amount of pool layers to calculate the final width and height after the convolutions
POOL_LAYERS = 3

# Calculating the width and height of the input data after all the pool layers, this is going to
# be used when the data is flatten for the last layers
WIDTH_AFTER_CONV = int(np.ceil(float(IN_WIDTH)/float(2**POOL_LAYERS)))
HEIGHT_AFTER_CONV = int(np.ceil(float(IN_HEIGHT)/float(2**POOL_LAYERS)))

# ==============================================================================
# Functions to split the string of the route and get the desired part
# ==============================================================================

# Function that splits and string to get the desire part
def get_part(part,string):
  aux = string.split('/')
  a = aux[len(aux)-part-1]
  return a

# function that gets the numeric part of a string
def get_int(string):
  return int(''.join([d for d in string if d.isdigit()]))

# ==============================================================================
# Configuration the architecture
# ==============================================================================

class resnet:

  def __init__(self):
    """ Creates the model """
    self.def_input()
    self.def_variable()
    self.def_params()
    self.def_model()
    self.def_output()
    self.def_loss()
    self.def_metrics()
    self.add_summaries()

  # Defining the conv2d operation with a stride of 1
  def conv2d(self, x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  # Defining the conv2d operation with a stride of 2
  def conv2ds2(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

  # Defining the max pool operation with a 2 x 2 kernel size and stride of 2
  def max_pool_2x2(self, x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Defining the average pool operation with a 7 x 7 kernel size and stride of 2
  def avg_pool_2x2(self,x):
    return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],strides=[1, 2, 2, 1], padding='SAME')

  # Defining the initializer for the parameters, it is used the Xavier initializer 
  def weight_variable(self,shape):
    initializer = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initializer)
    
  # Note: This model doesn't have BIAS beacuse it uses batch normalization

  def def_input(self):
    """ Defines inputs """
    with tf.name_scope('input'):

      # Defining the entrance of the model with the parameters defined at the begginig
      self.X1 = tf.placeholder(tf.float32, [None, IN_HEIGHT, IN_WIDTH, CHANNELS], name='X1')
      self.X2 = tf.placeholder(tf.float32, [None, IN_HEIGHT, IN_WIDTH, CHANNELS], name='X2')
      self.Y = tf.placeholder(tf.float32, [None, L_LABEL], name='Y')
      self.Ya = tf.placeholder(tf.float32, [None, L_CLASS], name='Ya')
      self.Yb = tf.placeholder(tf.float32, [None, L_CLASS], name='Yb')

      # Parameters fo the batch normalization paramater
      self.g_step = tf.contrib.framework.get_or_create_global_step()
      self.phase = tf.placeholder(tf.bool, name='phase')

  def def_variable(self):
    # Size of the batch, defined to be able to use it on the metrics section
    self.size_batch = float(FLAGS.batch_size)

  # Defining the parameters aka weights for the model
  # In this case the parameters are name with W for weight, b plus a number which means the number of the block
  # u plus a number which means the number of the unit of the block and cn plus a number which mean the convolutional
  # layer of the unit
  def def_params(self):

    self.weight = {}

    """ Defines model parameters """
    with tf.name_scope('params'):

      # Zero convolutional layer
      with tf.name_scope('conv0'):      
        self.weight["W_cn0"] = self.weight_variable([7,7,1, 64])

      # Block 1 --> 3 Units, the first unit has a shortcut

      # Block 1, unit 1
      with tf.name_scope('block1_unit1'):
        self.weight["W_b1_u1_cn0"] = self.weight_variable([1,1,64,256])
        self.weight["W_b1_u1_cn1"] = self.weight_variable([1,1,64,64])
        self.weight["W_b1_u1_cn2"] = self.weight_variable([3,3,64,64])
        self.weight["W_b1_u1_cn3"] = self.weight_variable([1,1,64,256])

      # Block 1, unit 2
      with tf.name_scope('block1_unit2'):
        self.weight["W_b1_u2_cn1"] = self.weight_variable([1,1,256,64])
        self.weight["W_b1_u2_cn2"] = self.weight_variable([3,3,64,64])
        self.weight["W_b1_u2_cn3"] = self.weight_variable([1,1,64,256])

      # Block 1, unit 3
      with tf.name_scope('block1_unit3'):
        self.weight["W_b1_u3_cn1"] = self.weight_variable([1,1,256,64])
        self.weight["W_b1_u3_cn2"] = self.weight_variable([3,3,64,64])
        self.weight["W_b1_u3_cn3"] = self.weight_variable([1,1,64,256])


      # Block 2 --> 4 Units, the first unit has a shortcut

      # Block 2, unit 1
      with tf.name_scope('block2_unit1'):
        self.weight["W_b2_u1_cn0"] = self.weight_variable([1,1,256, 512])
        self.weight["W_b2_u1_cn1"] = self.weight_variable([1,1,256, 128])
        self.weight["W_b2_u1_cn2"] = self.weight_variable([3,3,128, 128])
        self.weight["W_b2_u1_cn3"] = self.weight_variable([1,1,128, 512])

      # Block 2, unit 2
      with tf.name_scope('block2_unit2'):
        self.weight["W_b2_u2_cn1"] = self.weight_variable([1,1,512, 128])
        self.weight["W_b2_u2_cn2"] = self.weight_variable([3,3,128, 128])
        self.weight["W_b2_u2_cn3"] = self.weight_variable([1,1,128, 512])

      # Block 2, unit 3
      with tf.name_scope('block2_unit3'):
        self.weight["W_b2_u3_cn1"] = self.weight_variable([1,1,512, 128])
        self.weight["W_b2_u3_cn2"] = self.weight_variable([3,3,128, 128])
        self.weight["W_b2_u3_cn3"] = self.weight_variable([1,1,128, 512])

      # Block 2, unit 4
      with tf.name_scope('block2_unit4'):
        self.weight["W_b2_u4_cn1"] = self.weight_variable([1,1,512, 128])
        self.weight["W_b2_u4_cn2"] = self.weight_variable([3,3,128, 128])
        self.weight["W_b2_u4_cn3"] = self.weight_variable([1,1,128, 512])


      # Block 3 --> 6 Units, the first unit has a shortcut

      # Block 3, unit 1
      with tf.name_scope('block3_unit1'):
        self.weight["W_b3_u1_cn0"] = self.weight_variable([1,1,512, 1024])
        self.weight["W_b3_u1_cn1"] = self.weight_variable([1,1,512, 256])
        self.weight["W_b3_u1_cn2"] = self.weight_variable([3,3,256, 256])
        self.weight["W_b3_u1_cn3"] = self.weight_variable([1,1,256, 1024])

      # Block 3, unit 2
      with tf.name_scope('block3_unit2'):
        self.weight["W_b3_u2_cn1"] = self.weight_variable([1,1,1024, 256])
        self.weight["W_b3_u2_cn2"] = self.weight_variable([3,3,256, 256])
        self.weight["W_b3_u2_cn3"] = self.weight_variable([1,1,256, 1024])

      # Block 3, unit 3
      with tf.name_scope('block3_unit3'):
        self.weight["W_b3_u3_cn1"] = self.weight_variable([1,1,1024, 256])
        self.weight["W_b3_u3_cn2"] = self.weight_variable([3,3,256, 256])
        self.weight["W_b3_u3_cn3"] = self.weight_variable([1,1,256, 1024])

      # Block 3, unit 4
      with tf.name_scope('block3_unit4'):
        self.weight["W_b3_u4_cn1"] = self.weight_variable([1,1,1024, 256])
        self.weight["W_b3_u4_cn2"] = self.weight_variable([3,3,256, 256])
        self.weight["W_b3_u4_cn3"] = self.weight_variable([1,1,256, 1024])

      # Block 3, unit 5
      with tf.name_scope('block3_unit5'):
        self.weight["W_b3_u5_cn1"] = self.weight_variable([1,1,1024, 256])
        self.weight["W_b3_u5_cn2"] = self.weight_variable([3,3,256, 256])
        self.weight["W_b3_u5_cn3"] = self.weight_variable([1,1,256, 1024])

      # Block 3, unit 6
      with tf.name_scope('block3_unit6'):
        self.weight["W_b3_u6_cn1"] = self.weight_variable([1,1,1024, 256])
        self.weight["W_b3_u6_cn2"] = self.weight_variable([3,3,256, 256])
        self.weight["W_b3_u6_cn3"] = self.weight_variable([1,1,256, 1024])


      # Block 4 --> 3 Units, the first unit has a shortcut

      # Block 4, unit 1
      with tf.name_scope('block4_unit1'):
        self.weight["W_b4_u1_cn0"] = self.weight_variable([1,1,1024, 2048])
        self.weight["W_b4_u1_cn1"] = self.weight_variable([1,1,1024, 512])
        self.weight["W_b4_u1_cn2"] = self.weight_variable([3,3,512, 512])
        self.weight["W_b4_u1_cn3"] = self.weight_variable([1,1,512, 2048])

      # Block 4, unit 2
      with tf.name_scope('block4_unit2'):
        self.weight["W_b4_u2_cn1"] = self.weight_variable([1,1,2048, 512])
        self.weight["W_b4_u2_cn2"] = self.weight_variable([3,3,512, 512])
        self.weight["W_b4_u2_cn3"] = self.weight_variable([1,1,512, 2048])

      # Block 4, unit 3
      with tf.name_scope('block4_unit3'):
        self.weight["W_b4_u3_cn1"] = self.weight_variable([1,1,2048, 512])
        self.weight["W_b4_u3_cn2"] = self.weight_variable([3,3,512, 512])
        self.weight["W_b4_u3_cn3"] = self.weight_variable([1,1,512, 2048])


      # Fully connected
      with tf.name_scope('fc1'):# 30 x 71
        #self.weight["W_fc1"] = self.weight_variable([2 * 2048 * WIDTH_AFTER_CONV * HEIGHT_AFTER_CONV, L_LABEL])
        self.weight["W_fc1"] = self.weight_variable([2 * 2048 * WIDTH_AFTER_CONV * HEIGHT_AFTER_CONV, 2048])
        self.weight["W_fc2"] = self.weight_variable([2048, L_LABEL])
        self.weight["W_fc1a"] = self.weight_variable([2048 * WIDTH_AFTER_CONV * HEIGHT_AFTER_CONV, L_CLASS])
  
  
  # Defining the architecture of the model
  # In this case a resnet 50 is going to be used, the resnet 50 has blocks which are known as building blocks, this
  # blocks are composed by unit that have three layers one of a convolution with a kernel of 1x1, the second with a
  # convolution with a kernet of 3x3 and a third with a convolution of 1x1

  def def_model(self):
    """ Defines the model """
    with tf.name_scope('model'):

      with tf.name_scope('conv0a'):
        h_cn0a = self.conv2ds2(self.X1, self.weight["W_cn0"])
        h_cn0a = tf.layers.batch_normalization(inputs=h_cn0a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_cn0a = tf.nn.relu(h_cn0a)

      with tf.name_scope('pool0a'):
        h_pool1a = self.max_pool_2x2(h_cn0a)

      # Block 1, unit 1
      with tf.name_scope('block1_unit1a'):

        # Calculating the first shortcut
        shortcut_b1a = self.conv2d(h_pool1a, self.weight["W_b1_u1_cn0"])
        shortcut_b1a = tf.layers.batch_normalization(inputs=shortcut_b1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        h_b1_u1_cn1a = self.conv2d(h_pool1a, self.weight["W_b1_u1_cn1"])
        h_b1_u1_cn1a = tf.layers.batch_normalization(inputs=h_b1_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u1_cn1a = tf.nn.relu(h_b1_u1_cn1a)

        h_b1_u1_cn2a = self.conv2d(h_b1_u1_cn1a, self.weight["W_b1_u1_cn2"])
        h_b1_u1_cn2a = tf.layers.batch_normalization(inputs=h_b1_u1_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u1_cn2a = tf.nn.relu(h_b1_u1_cn2a)

        h_b1_u1_cn3a = self.conv2d(h_b1_u1_cn2a, self.weight["W_b1_u1_cn3"])
        h_b1_u1_cn3a = tf.layers.batch_normalization(inputs=h_b1_u1_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u1_cn3a = tf.add(h_b1_u1_cn3a, shortcut_b1a)
        h_b1_u1_cn3a = tf.nn.relu(h_b1_u1_cn3a)


      # Block 1, unit 2
      with tf.name_scope('block1_unit2a'):

        h_b1_u2_cn1a = self.conv2d(h_b1_u1_cn3a, self.weight["W_b1_u2_cn1"])
        h_b1_u2_cn1a = tf.layers.batch_normalization(inputs=h_b1_u2_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u2_cn1a = tf.nn.relu(h_b1_u2_cn1a)

        h_b1_u2_cn2a = self.conv2d(h_b1_u2_cn1a, self.weight["W_b1_u2_cn2"])
        h_b1_u2_cn2a = tf.layers.batch_normalization(inputs=h_b1_u2_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u2_cn2a = tf.nn.relu(h_b1_u2_cn2a)

        h_b1_u2_cn3a = self.conv2d(h_b1_u2_cn2a, self.weight["W_b1_u2_cn3"])
        h_b1_u2_cn3a = tf.layers.batch_normalization(inputs=h_b1_u2_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u2_cn3a = tf.add(h_b1_u2_cn3a, h_b1_u1_cn3a)
        h_b1_u2_cn3a = tf.nn.relu(h_b1_u2_cn3a)


      # Block 1, unit 3
      with tf.name_scope('block1_unit3a'):

        h_b1_u3_cn1a = self.conv2d(h_b1_u2_cn3a, self.weight["W_b1_u3_cn1"])
        h_b1_u3_cn1a = tf.layers.batch_normalization(inputs=h_b1_u3_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u3_cn1a = tf.nn.relu(h_b1_u3_cn1a)

        h_b1_u3_cn2a = self.conv2d(h_b1_u3_cn1a, self.weight["W_b1_u3_cn2"])
        h_b1_u3_cn2a = tf.layers.batch_normalization(inputs=h_b1_u3_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u3_cn2a = tf.nn.relu(h_b1_u3_cn2a)

        h_b1_u3_cn3a = self.conv2d(h_b1_u3_cn2a, self.weight["W_b1_u3_cn3"])
        h_b1_u3_cn3a = tf.layers.batch_normalization(inputs=h_b1_u3_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u3_cn3a = tf.add(h_b1_u3_cn3a, h_b1_u2_cn3a)
        h_b1_u3_cn3a = tf.nn.relu(h_b1_u3_cn3a)


      # Block 2, unit 1
      with tf.name_scope('block2_unit1a'):
        # Original way to go on a resnet50
        # Calculating the first shortcut
        # shortcut_b2a = self.conv2ds2(h_b1_u3_cn3a, self.weight["W_b2_u1_cn0"])
        # shortcut_b2a = tf.layers.batch_normalization(inputs=shortcut_b2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        # h_b2_u1_cn1a = self.conv2ds2(h_b1_u3_cn3a, self.weight["W_b2_u1_cn1"])
        # h_b2_u1_cn1a = tf.layers.batch_normalization(inputs=h_b2_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        # h_b2_u1_cn1a = tf.nn.relu(h_b2_u1_cn1a)

        # Modification in the resnet 50 due to excesive reduction of the input data through the blocks
        # The modification is to use the conv2d function insted of the conv2ds2

        # Calculating the first shortcut
        shortcut_b2a = self.conv2d(h_b1_u3_cn3a, self.weight["W_b2_u1_cn0"])
        shortcut_b2a = tf.layers.batch_normalization(inputs=shortcut_b2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        h_b2_u1_cn1a = self.conv2d(h_b1_u3_cn3a, self.weight["W_b2_u1_cn1"])
        h_b2_u1_cn1a = tf.layers.batch_normalization(inputs=h_b2_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u1_cn1a = tf.nn.relu(h_b2_u1_cn1a)

        h_b2_u1_cn2a = self.conv2d(h_b2_u1_cn1a, self.weight["W_b2_u1_cn2"])
        h_b2_u1_cn2a = tf.layers.batch_normalization(inputs=h_b2_u1_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u1_cn2a = tf.nn.relu(h_b2_u1_cn2a)

        h_b2_u1_cn3a = self.conv2d(h_b2_u1_cn2a, self.weight["W_b2_u1_cn3"])
        h_b2_u1_cn3a = tf.layers.batch_normalization(inputs=h_b2_u1_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u1_cn3a = tf.add(h_b2_u1_cn3a, shortcut_b2a)
        h_b2_u1_cn3a = tf.nn.relu(h_b2_u1_cn3a)

      
      # Block 2, unit 2
      with tf.name_scope('block2_unit2a'):

        h_b2_u2_cn1a = self.conv2d(h_b2_u1_cn3a, self.weight["W_b2_u2_cn1"])
        h_b2_u2_cn1a = tf.layers.batch_normalization(inputs=h_b2_u2_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u2_cn1a = tf.nn.relu(h_b2_u2_cn1a)

        h_b2_u2_cn2a = self.conv2d(h_b2_u2_cn1a, self.weight["W_b2_u2_cn2"])
        h_b2_u2_cn2a = tf.layers.batch_normalization(inputs=h_b2_u2_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u2_cn2a = tf.nn.relu(h_b2_u2_cn2a)

        h_b2_u2_cn3a = self.conv2d(h_b2_u2_cn2a, self.weight["W_b2_u2_cn3"])
        h_b2_u2_cn3a = tf.layers.batch_normalization(inputs=h_b2_u2_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u2_cn3a = tf.add(h_b2_u2_cn3a, h_b2_u1_cn3a)
        h_b2_u2_cn3a = tf.nn.relu(h_b2_u2_cn3a)


      # Block 2, unit 3
      with tf.name_scope('block2_unit3a'):

        h_b2_u3_cn1a = self.conv2d(h_b2_u2_cn3a, self.weight["W_b2_u3_cn1"])
        h_b2_u3_cn1a = tf.layers.batch_normalization(inputs=h_b2_u3_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u3_cn1a = tf.nn.relu(h_b2_u3_cn1a)

        h_b2_u3_cn2a = self.conv2d(h_b2_u3_cn1a, self.weight["W_b2_u3_cn2"])
        h_b2_u3_cn2a = tf.layers.batch_normalization(inputs=h_b2_u3_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u3_cn2a = tf.nn.relu(h_b2_u3_cn2a)

        h_b2_u3_cn3a = self.conv2d(h_b2_u3_cn2a, self.weight["W_b2_u3_cn3"])
        h_b2_u3_cn3a = tf.layers.batch_normalization(inputs=h_b2_u3_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u3_cn3a = tf.add(h_b2_u3_cn3a, h_b2_u2_cn3a)
        h_b2_u3_cn3a = tf.nn.relu(h_b2_u3_cn3a)


      # Block 2, unit 4
      with tf.name_scope('block2_unit4a'):

        h_b2_u4_cn1a = self.conv2d(h_b2_u3_cn3a, self.weight["W_b2_u4_cn1"])
        h_b2_u4_cn1a = tf.layers.batch_normalization(inputs=h_b2_u4_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u4_cn1a = tf.nn.relu(h_b2_u4_cn1a)

        h_b2_u4_cn2a = self.conv2d(h_b2_u4_cn1a, self.weight["W_b2_u4_cn2"])
        h_b2_u4_cn2a = tf.layers.batch_normalization(inputs=h_b2_u4_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u4_cn2a = tf.nn.relu(h_b2_u4_cn2a)

        h_b2_u4_cn3a = self.conv2d(h_b2_u4_cn2a, self.weight["W_b2_u4_cn3"])
        h_b2_u4_cn3a = tf.layers.batch_normalization(inputs=h_b2_u4_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u4_cn3a = tf.add(h_b2_u4_cn3a, h_b2_u3_cn3a)
        h_b2_u4_cn3a = tf.nn.relu(h_b2_u4_cn3a)


      # Block 3, unit 1
      with tf.name_scope('block3_unit1a'):
        # Original way to go on a resnet50
        # Calculating the first shortcut
        # shortcut_b3a = self.conv2ds2(h_b2_u4_cn3a, self.weight["W_b3_u1_cn0"])
        # shortcut_b3a = tf.layers.batch_normalization(inputs=shortcut_b3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        # h_b3_u1_cn1a = self.conv2ds2(h_b2_u4_cn3a, self.weight["W_b3_u1_cn1"])
        # h_b3_u1_cn1a = tf.layers.batch_normalization(inputs=h_b3_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        # h_b3_u1_cn1a = tf.nn.relu(h_b3_u1_cn1a)

        # Modification in the resnet 50 due to excesive reduction of the input data through the blocks
        # The modification is to use the conv2d function insted of the conv2ds2

        # Calculating the shortcut
        shortcut_b3a = self.conv2d(h_b2_u4_cn3a, self.weight["W_b3_u1_cn0"])
        shortcut_b3a = tf.layers.batch_normalization(inputs=shortcut_b3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        h_b3_u1_cn1a = self.conv2d(h_b2_u4_cn3a, self.weight["W_b3_u1_cn1"])
        h_b3_u1_cn1a = tf.layers.batch_normalization(inputs=h_b3_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u1_cn1a = tf.nn.relu(h_b3_u1_cn1a)

        h_b3_u1_cn2a = self.conv2d(h_b3_u1_cn1a, self.weight["W_b3_u1_cn2"])
        h_b3_u1_cn2a = tf.layers.batch_normalization(inputs=h_b3_u1_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u1_cn2a = tf.nn.relu(h_b3_u1_cn2a)

        h_b3_u1_cn3a = self.conv2d(h_b3_u1_cn2a, self.weight["W_b3_u1_cn3"])
        h_b3_u1_cn3a = tf.layers.batch_normalization(inputs=h_b3_u1_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u1_cn3a = tf.add(h_b3_u1_cn3a, shortcut_b3a)
        h_b3_u1_cn3a = tf.nn.relu(h_b3_u1_cn3a)

      
      # Block 3, unit 2
      with tf.name_scope('block3_unit2a'):

        h_b3_u2_cn1a = self.conv2d(h_b3_u1_cn3a, self.weight["W_b3_u2_cn1"])
        h_b3_u2_cn1a = tf.layers.batch_normalization(inputs=h_b3_u2_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u2_cn1a = tf.nn.relu(h_b3_u2_cn1a)

        h_b3_u2_cn2a = self.conv2d(h_b3_u2_cn1a, self.weight["W_b3_u2_cn2"])
        h_b3_u2_cn2a = tf.layers.batch_normalization(inputs=h_b3_u2_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u2_cn2a = tf.nn.relu(h_b3_u2_cn2a)

        h_b3_u2_cn3a = self.conv2d(h_b3_u2_cn2a, self.weight["W_b3_u2_cn3"])
        h_b3_u2_cn3a = tf.layers.batch_normalization(inputs=h_b3_u2_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u2_cn3a = tf.add(h_b3_u2_cn3a, h_b3_u1_cn3a)
        h_b3_u2_cn3a = tf.nn.relu(h_b3_u2_cn3a)


      # Block 3, unit 3
      with tf.name_scope('block3_unit3a'):

        h_b3_u3_cn1a = self.conv2d(h_b3_u2_cn3a, self.weight["W_b3_u3_cn1"])
        h_b3_u3_cn1a = tf.layers.batch_normalization(inputs=h_b3_u3_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u3_cn1a = tf.nn.relu(h_b3_u3_cn1a)

        h_b3_u3_cn2a = self.conv2d(h_b3_u3_cn1a, self.weight["W_b3_u3_cn2"])
        h_b3_u3_cn2a = tf.layers.batch_normalization(inputs=h_b3_u3_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u3_cn2a = tf.nn.relu(h_b3_u3_cn2a)

        h_b3_u3_cn3a = self.conv2d(h_b3_u3_cn2a, self.weight["W_b3_u3_cn3"])
        h_b3_u3_cn3a = tf.layers.batch_normalization(inputs=h_b3_u3_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u3_cn3a = tf.add(h_b3_u3_cn3a, h_b3_u2_cn3a)
        h_b3_u3_cn3a = tf.nn.relu(h_b3_u3_cn3a)


      # Block 3, unit 4
      with tf.name_scope('block3_unit4a'):

        h_b3_u4_cn1a = self.conv2d(h_b3_u3_cn3a, self.weight["W_b3_u4_cn1"])
        h_b3_u4_cn1a = tf.layers.batch_normalization(inputs=h_b3_u4_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u4_cn1a = tf.nn.relu(h_b3_u4_cn1a)

        h_b3_u4_cn2a = self.conv2d(h_b3_u4_cn1a, self.weight["W_b3_u4_cn2"])
        h_b3_u4_cn2a = tf.layers.batch_normalization(inputs=h_b3_u4_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u4_cn2a = tf.nn.relu(h_b3_u4_cn2a)

        h_b3_u4_cn3a = self.conv2d(h_b3_u4_cn2a, self.weight["W_b3_u4_cn3"])
        h_b3_u4_cn3a = tf.layers.batch_normalization(inputs=h_b3_u4_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u4_cn3a = tf.add(h_b3_u4_cn3a, h_b3_u3_cn3a)
        h_b3_u4_cn3a = tf.nn.relu(h_b3_u4_cn3a)


      # Block 3, unit 5
      with tf.name_scope('block3_unit5a'):

        h_b3_u5_cn1a = self.conv2d(h_b3_u4_cn3a, self.weight["W_b3_u5_cn1"])
        h_b3_u5_cn1a = tf.layers.batch_normalization(inputs=h_b3_u5_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u5_cn1a = tf.nn.relu(h_b3_u5_cn1a)

        h_b3_u5_cn2a = self.conv2d(h_b3_u5_cn1a, self.weight["W_b3_u5_cn2"])
        h_b3_u5_cn2a = tf.layers.batch_normalization(inputs=h_b3_u5_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u5_cn2a = tf.nn.relu(h_b3_u5_cn2a)

        h_b3_u5_cn3a = self.conv2d(h_b3_u5_cn2a, self.weight["W_b3_u5_cn3"])
        h_b3_u5_cn3a = tf.layers.batch_normalization(inputs=h_b3_u5_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u5_cn3a = tf.add(h_b3_u5_cn3a, h_b3_u4_cn3a)
        h_b3_u5_cn3a = tf.nn.relu(h_b3_u5_cn3a)


      # Block 3, unit 6
      with tf.name_scope('block3_unit6a'):

        h_b3_u6_cn1a = self.conv2d(h_b3_u5_cn3a, self.weight["W_b3_u6_cn1"])
        h_b3_u6_cn1a = tf.layers.batch_normalization(inputs=h_b3_u6_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u6_cn1a = tf.nn.relu(h_b3_u6_cn1a)

        h_b3_u6_cn2a = self.conv2d(h_b3_u6_cn1a, self.weight["W_b3_u6_cn2"])
        h_b3_u6_cn2a = tf.layers.batch_normalization(inputs=h_b3_u6_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u6_cn2a = tf.nn.relu(h_b3_u6_cn2a)

        h_b3_u6_cn3a = self.conv2d(h_b3_u6_cn2a, self.weight["W_b3_u6_cn3"])
        h_b3_u6_cn3a = tf.layers.batch_normalization(inputs=h_b3_u6_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u6_cn3a = tf.add(h_b3_u6_cn3a, h_b3_u5_cn3a)
        h_b3_u6_cn3a = tf.nn.relu(h_b3_u6_cn3a)


      # Block 4, unit 1
      with tf.name_scope('block4_unit1a'):

        # Original way to go on a resnet50
        # Calculating the first shortcut
        # shortcut_b4a = self.conv2ds2(h_b3_u6_cn3a, self.weight["W_b4_u1_cn0"])
        # shortcut_b4a = tf.layers.batch_normalization(inputs=shortcut_b4a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        # h_b4_u1_cn1a = self.conv2ds2(h_b3_u6_cn3a, self.weight["W_b4_u1_cn1"])
        # h_b4_u1_cn1a = tf.layers.batch_normalization(inputs=h_b4_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        # h_b4_u1_cn1a = tf.nn.relu(h_b4_u1_cn1a)

        # Modification in the resnet 50 due to excesive reduction of the input data through the blocks
        # The modification is to use the conv2d function insted of the conv2ds2

        # Calculating the shortcut
        shortcut_b4a = self.conv2d(h_b3_u6_cn3a, self.weight["W_b4_u1_cn0"])
        shortcut_b4a = tf.layers.batch_normalization(inputs=shortcut_b4a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        h_b4_u1_cn1a = self.conv2d(h_b3_u6_cn3a, self.weight["W_b4_u1_cn1"])
        h_b4_u1_cn1a = tf.layers.batch_normalization(inputs=h_b4_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u1_cn1a = tf.nn.relu(h_b4_u1_cn1a)

        h_b4_u1_cn2a = self.conv2d(h_b4_u1_cn1a, self.weight["W_b4_u1_cn2"])
        h_b4_u1_cn2a = tf.layers.batch_normalization(inputs=h_b4_u1_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u1_cn2a = tf.nn.relu(h_b4_u1_cn2a)

        h_b4_u1_cn3a = self.conv2d(h_b4_u1_cn2a, self.weight["W_b4_u1_cn3"])
        h_b4_u1_cn3a = tf.layers.batch_normalization(inputs=h_b4_u1_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u1_cn3a = tf.add(h_b4_u1_cn3a, shortcut_b4a)
        h_b4_u1_cn3a = tf.nn.relu(h_b4_u1_cn3a)


      # Block 4, unit 2
      with tf.name_scope('block4_unit2a'):

        h_b4_u2_cn1a = self.conv2d(h_b4_u1_cn3a, self.weight["W_b4_u2_cn1"])
        h_b4_u2_cn1a = tf.layers.batch_normalization(inputs=h_b4_u2_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u2_cn1a = tf.nn.relu(h_b4_u2_cn1a)

        h_b4_u2_cn2a = self.conv2d(h_b4_u2_cn1a, self.weight["W_b4_u2_cn2"])
        h_b4_u2_cn2a = tf.layers.batch_normalization(inputs=h_b4_u2_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u2_cn2a = tf.nn.relu(h_b4_u2_cn2a)

        h_b4_u2_cn3a = self.conv2d(h_b4_u2_cn2a, self.weight["W_b4_u2_cn3"])
        h_b4_u2_cn3a = tf.layers.batch_normalization(inputs=h_b4_u2_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u2_cn3a = tf.add(h_b4_u2_cn3a, h_b4_u1_cn3a)
        h_b4_u2_cn3a = tf.nn.relu(h_b4_u2_cn3a)


      # Block 4, unit 3
      with tf.name_scope('block4_unit3a'):

        h_b4_u3_cn1a = self.conv2d(h_b4_u2_cn3a, self.weight["W_b4_u3_cn1"])
        h_b4_u3_cn1a = tf.layers.batch_normalization(inputs=h_b4_u3_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u3_cn1a = tf.nn.relu(h_b4_u3_cn1a)

        h_b4_u3_cn2a = self.conv2d(h_b4_u3_cn1a, self.weight["W_b4_u3_cn2"])
        h_b4_u3_cn2a = tf.layers.batch_normalization(inputs=h_b4_u3_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u3_cn2a = tf.nn.relu(h_b4_u3_cn2a)

        h_b4_u3_cn3a = self.conv2d(h_b4_u3_cn2a, self.weight["W_b4_u3_cn3"])
        h_b4_u3_cn3a = tf.layers.batch_normalization(inputs=h_b4_u3_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u3_cn3a = tf.add(h_b4_u3_cn3a, h_b4_u2_cn3a)
        h_b4_u3_cn3a = tf.nn.relu(h_b4_u3_cn3a)

      with tf.name_scope('pool1a'):
        h_pool2a = self.avg_pool_2x2(h_b4_u3_cn3a)
        


      with tf.name_scope('conv0b'):
        h_cn0b = self.conv2ds2(self.X2, self.weight["W_cn0"])
        h_cn0b = tf.layers.batch_normalization(inputs=h_cn0b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_cn0b = tf.nn.relu(h_cn0b)

      with tf.name_scope('pool0b'):
        h_pool1b = self.max_pool_2x2(h_cn0b)

      # Block 1, unit 1
      with tf.name_scope('block1_unit1b'):

        # Calculating the first shortcut
        shortcut_b1b = self.conv2d(h_pool1b, self.weight["W_b1_u1_cn0"])
        shortcut_b1b = tf.layers.batch_normalization(inputs=shortcut_b1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        h_b1_u1_cn1b = self.conv2d(h_pool1b, self.weight["W_b1_u1_cn1"])
        h_b1_u1_cn1b = tf.layers.batch_normalization(inputs=h_b1_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u1_cn1b = tf.nn.relu(h_b1_u1_cn1b)

        h_b1_u1_cn2b = self.conv2d(h_b1_u1_cn1b, self.weight["W_b1_u1_cn2"])
        h_b1_u1_cn2b = tf.layers.batch_normalization(inputs=h_b1_u1_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u1_cn2b = tf.nn.relu(h_b1_u1_cn2b)

        h_b1_u1_cn3b = self.conv2d(h_b1_u1_cn2b, self.weight["W_b1_u1_cn3"])
        h_b1_u1_cn3b = tf.layers.batch_normalization(inputs=h_b1_u1_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u1_cn3b = tf.add(h_b1_u1_cn3b, shortcut_b1b)
        h_b1_u1_cn3b = tf.nn.relu(h_b1_u1_cn3b)


      # Block 1, unit 2
      with tf.name_scope('block1_unit2b'):

        h_b1_u2_cn1b = self.conv2d(h_b1_u1_cn3b, self.weight["W_b1_u2_cn1"])
        h_b1_u2_cn1b = tf.layers.batch_normalization(inputs=h_b1_u2_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u2_cn1b = tf.nn.relu(h_b1_u2_cn1b)

        h_b1_u2_cn2b = self.conv2d(h_b1_u2_cn1b, self.weight["W_b1_u2_cn2"])
        h_b1_u2_cn2b = tf.layers.batch_normalization(inputs=h_b1_u2_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u2_cn2b = tf.nn.relu(h_b1_u2_cn2b)

        h_b1_u2_cn3b = self.conv2d(h_b1_u2_cn2b, self.weight["W_b1_u2_cn3"])
        h_b1_u2_cn3b = tf.layers.batch_normalization(inputs=h_b1_u2_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u2_cn3b = tf.add(h_b1_u2_cn3b, h_b1_u1_cn3b)
        h_b1_u2_cn3b = tf.nn.relu(h_b1_u2_cn3b)


      # Block 1, unit 3
      with tf.name_scope('block1_unit3b'):

        h_b1_u3_cn1b = self.conv2d(h_b1_u2_cn3b, self.weight["W_b1_u3_cn1"])
        h_b1_u3_cn1b = tf.layers.batch_normalization(inputs=h_b1_u3_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u3_cn1b = tf.nn.relu(h_b1_u3_cn1b)

        h_b1_u3_cn2b = self.conv2d(h_b1_u3_cn1b, self.weight["W_b1_u3_cn2"])
        h_b1_u3_cn2b = tf.layers.batch_normalization(inputs=h_b1_u3_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u3_cn2b = tf.nn.relu(h_b1_u3_cn2b)

        h_b1_u3_cn3b = self.conv2d(h_b1_u3_cn2b, self.weight["W_b1_u3_cn3"])
        h_b1_u3_cn3b = tf.layers.batch_normalization(inputs=h_b1_u3_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b1_u3_cn3b = tf.add(h_b1_u3_cn3b, h_b1_u2_cn3b)
        h_b1_u3_cn3b = tf.nn.relu(h_b1_u3_cn3b)


      # Block 2, unit 1
      with tf.name_scope('block2_unit1b'):

        # Original way to go on a resnet50
        # Calculating the first shortcut
        # shortcut_b2b = self.conv2ds2(h_b1_u3_cn3b, self.weight["W_b2_u1_cn0"])
        # shortcut_b2b = tf.layers.batch_normalization(inputs=shortcut_b2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        # h_b2_u1_cn1b = self.conv2ds2(h_b1_u3_cn3b, self.weight["W_b2_u1_cn1"])
        # h_b2_u1_cn1b = tf.layers.batch_normalization(inputs=h_b2_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        # h_b2_u1_cn1b = tf.nn.relu(h_b2_u1_cn1b)


        # Modification in the resnet 50 due to excesive reduction of the input data through the blocks
        # The modification is to use the conv2d function insted of the conv2ds2

        # Calculating the first shortcut
        shortcut_b2b = self.conv2d(h_b1_u3_cn3b, self.weight["W_b2_u1_cn0"])
        shortcut_b2b = tf.layers.batch_normalization(inputs=shortcut_b2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        h_b2_u1_cn1b = self.conv2d(h_b1_u3_cn3b, self.weight["W_b2_u1_cn1"])
        h_b2_u1_cn1b = tf.layers.batch_normalization(inputs=h_b2_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u1_cn1b = tf.nn.relu(h_b2_u1_cn1b)

        h_b2_u1_cn2b = self.conv2d(h_b2_u1_cn1b, self.weight["W_b2_u1_cn2"])
        h_b2_u1_cn2b = tf.layers.batch_normalization(inputs=h_b2_u1_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u1_cn2b = tf.nn.relu(h_b2_u1_cn2b)

        h_b2_u1_cn3b = self.conv2d(h_b2_u1_cn2b, self.weight["W_b2_u1_cn3"])
        h_b2_u1_cn3b = tf.layers.batch_normalization(inputs=h_b2_u1_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u1_cn3b = tf.add(h_b2_u1_cn3b, shortcut_b2b)
        h_b2_u1_cn3b = tf.nn.relu(h_b2_u1_cn3b)

      
      # Block 2, unit 2
      with tf.name_scope('block2_unit2b'):

        h_b2_u2_cn1b = self.conv2d(h_b2_u1_cn3b, self.weight["W_b2_u2_cn1"])
        h_b2_u2_cn1b = tf.layers.batch_normalization(inputs=h_b2_u2_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u2_cn1b = tf.nn.relu(h_b2_u2_cn1b)

        h_b2_u2_cn2b = self.conv2d(h_b2_u2_cn1b, self.weight["W_b2_u2_cn2"])
        h_b2_u2_cn2b = tf.layers.batch_normalization(inputs=h_b2_u2_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u2_cn2b = tf.nn.relu(h_b2_u2_cn2b)

        h_b2_u2_cn3b = self.conv2d(h_b2_u2_cn2b, self.weight["W_b2_u2_cn3"])
        h_b2_u2_cn3b = tf.layers.batch_normalization(inputs=h_b2_u2_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u2_cn3b = tf.add(h_b2_u2_cn3b, h_b2_u1_cn3b)
        h_b2_u2_cn3b = tf.nn.relu(h_b2_u2_cn3b)


      # Block 2, unit 3
      with tf.name_scope('block2_unit3b'):

        h_b2_u3_cn1b = self.conv2d(h_b2_u2_cn3b, self.weight["W_b2_u3_cn1"])
        h_b2_u3_cn1b = tf.layers.batch_normalization(inputs=h_b2_u3_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u3_cn1b = tf.nn.relu(h_b2_u3_cn1b)

        h_b2_u3_cn2b = self.conv2d(h_b2_u3_cn1b, self.weight["W_b2_u3_cn2"])
        h_b2_u3_cn2b = tf.layers.batch_normalization(inputs=h_b2_u3_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u3_cn2b = tf.nn.relu(h_b2_u3_cn2b)

        h_b2_u3_cn3b = self.conv2d(h_b2_u3_cn2b, self.weight["W_b2_u3_cn3"])
        h_b2_u3_cn3b = tf.layers.batch_normalization(inputs=h_b2_u3_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u3_cn3b = tf.add(h_b2_u3_cn3b, h_b2_u2_cn3b)
        h_b2_u3_cn3b = tf.nn.relu(h_b2_u3_cn3b)


      # Block 2, unit 4
      with tf.name_scope('block2_unit4b'):

        h_b2_u4_cn1b = self.conv2d(h_b2_u3_cn3b, self.weight["W_b2_u4_cn1"])
        h_b2_u4_cn1b = tf.layers.batch_normalization(inputs=h_b2_u4_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u4_cn1b = tf.nn.relu(h_b2_u4_cn1b)

        h_b2_u4_cn2b = self.conv2d(h_b2_u4_cn1b, self.weight["W_b2_u4_cn2"])
        h_b2_u4_cn2b = tf.layers.batch_normalization(inputs=h_b2_u4_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u4_cn2b = tf.nn.relu(h_b2_u4_cn2b)

        h_b2_u4_cn3b = self.conv2d(h_b2_u4_cn2b, self.weight["W_b2_u4_cn3"])
        h_b2_u4_cn3b = tf.layers.batch_normalization(inputs=h_b2_u4_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b2_u4_cn3b = tf.add(h_b2_u4_cn3b, h_b2_u3_cn3b)
        h_b2_u4_cn3b = tf.nn.relu(h_b2_u4_cn3b)


      # Block 3, unit 1
      with tf.name_scope('block3_unit1b'):

        # Original way to go on a resnet50
        # Calculating the first shortcut
        # shortcut_b3b = self.conv2ds2(h_b2_u4_cn3b, self.weight["W_b3_u1_cn0"])
        # shortcut_b3b = tf.layers.batch_normalization(inputs=shortcut_b3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        # h_b3_u1_cn1b = self.conv2ds2(h_b2_u4_cn3b, self.weight["W_b3_u1_cn1"])
        # h_b3_u1_cn1b = tf.layers.batch_normalization(inputs=h_b3_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        # h_b3_u1_cn1b = tf.nn.relu(h_b3_u1_cn1b)

        # Modification in the resnet 50 due to excesive reduction of the input data through the blocks
        # The modification is to use the conv2d function insted of the conv2ds2

        # Calculating the shortcut
        shortcut_b3b = self.conv2d(h_b2_u4_cn3b, self.weight["W_b3_u1_cn0"])
        shortcut_b3b = tf.layers.batch_normalization(inputs=shortcut_b3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        h_b3_u1_cn1b = self.conv2d(h_b2_u4_cn3b, self.weight["W_b3_u1_cn1"])
        h_b3_u1_cn1b = tf.layers.batch_normalization(inputs=h_b3_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u1_cn1b = tf.nn.relu(h_b3_u1_cn1b)

        h_b3_u1_cn2b = self.conv2d(h_b3_u1_cn1b, self.weight["W_b3_u1_cn2"])
        h_b3_u1_cn2b = tf.layers.batch_normalization(inputs=h_b3_u1_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u1_cn2b = tf.nn.relu(h_b3_u1_cn2b)

        h_b3_u1_cn3b = self.conv2d(h_b3_u1_cn2b, self.weight["W_b3_u1_cn3"])
        h_b3_u1_cn3b = tf.layers.batch_normalization(inputs=h_b3_u1_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u1_cn3b = tf.add(h_b3_u1_cn3b, shortcut_b3b)
        h_b3_u1_cn3b = tf.nn.relu(h_b3_u1_cn3b)

      
      # Block 3, unit 2
      with tf.name_scope('block3_unit2b'):

        h_b3_u2_cn1b = self.conv2d(h_b3_u1_cn3b, self.weight["W_b3_u2_cn1"])
        h_b3_u2_cn1b = tf.layers.batch_normalization(inputs=h_b3_u2_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u2_cn1b = tf.nn.relu(h_b3_u2_cn1b)

        h_b3_u2_cn2b = self.conv2d(h_b3_u2_cn1b, self.weight["W_b3_u2_cn2"])
        h_b3_u2_cn2b = tf.layers.batch_normalization(inputs=h_b3_u2_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u2_cn2b = tf.nn.relu(h_b3_u2_cn2b)

        h_b3_u2_cn3b = self.conv2d(h_b3_u2_cn2b, self.weight["W_b3_u2_cn3"])
        h_b3_u2_cn3b = tf.layers.batch_normalization(inputs=h_b3_u2_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u2_cn3b = tf.add(h_b3_u2_cn3b, h_b3_u1_cn3b)
        h_b3_u2_cn3b = tf.nn.relu(h_b3_u2_cn3b)


      # Block 3, unit 3
      with tf.name_scope('block3_unit3b'):

        h_b3_u3_cn1b = self.conv2d(h_b3_u2_cn3b, self.weight["W_b3_u3_cn1"])
        h_b3_u3_cn1b = tf.layers.batch_normalization(inputs=h_b3_u3_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u3_cn1b = tf.nn.relu(h_b3_u3_cn1b)

        h_b3_u3_cn2b = self.conv2d(h_b3_u3_cn1b, self.weight["W_b3_u3_cn2"])
        h_b3_u3_cn2b = tf.layers.batch_normalization(inputs=h_b3_u3_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u3_cn2b = tf.nn.relu(h_b3_u3_cn2b)

        h_b3_u3_cn3b = self.conv2d(h_b3_u3_cn2b, self.weight["W_b3_u3_cn3"])
        h_b3_u3_cn3b = tf.layers.batch_normalization(inputs=h_b3_u3_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u3_cn3b = tf.add(h_b3_u3_cn3b, h_b3_u2_cn3b)
        h_b3_u3_cn3b = tf.nn.relu(h_b3_u3_cn3b)


      # Block 3, unit 4
      with tf.name_scope('block3_unit4b'):

        h_b3_u4_cn1b = self.conv2d(h_b3_u3_cn3b, self.weight["W_b3_u4_cn1"])
        h_b3_u4_cn1b = tf.layers.batch_normalization(inputs=h_b3_u4_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u4_cn1b = tf.nn.relu(h_b3_u4_cn1b)

        h_b3_u4_cn2b = self.conv2d(h_b3_u4_cn1b, self.weight["W_b3_u4_cn2"])
        h_b3_u4_cn2b = tf.layers.batch_normalization(inputs=h_b3_u4_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u4_cn2b = tf.nn.relu(h_b3_u4_cn2b)

        h_b3_u4_cn3b = self.conv2d(h_b3_u4_cn2b, self.weight["W_b3_u4_cn3"])
        h_b3_u4_cn3b = tf.layers.batch_normalization(inputs=h_b3_u4_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u4_cn3b = tf.add(h_b3_u4_cn3b, h_b3_u3_cn3b)
        h_b3_u4_cn3b = tf.nn.relu(h_b3_u4_cn3b)


      # Block 3, unit 5
      with tf.name_scope('block3_unit5b'):

        h_b3_u5_cn1b = self.conv2d(h_b3_u4_cn3b, self.weight["W_b3_u5_cn1"])
        h_b3_u5_cn1b = tf.layers.batch_normalization(inputs=h_b3_u5_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u5_cn1b = tf.nn.relu(h_b3_u5_cn1b)

        h_b3_u5_cn2b = self.conv2d(h_b3_u5_cn1b, self.weight["W_b3_u5_cn2"])
        h_b3_u5_cn2b = tf.layers.batch_normalization(inputs=h_b3_u5_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u5_cn2b = tf.nn.relu(h_b3_u5_cn2b)

        h_b3_u5_cn3b = self.conv2d(h_b3_u5_cn2b, self.weight["W_b3_u5_cn3"])
        h_b3_u5_cn3b = tf.layers.batch_normalization(inputs=h_b3_u5_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u5_cn3b = tf.add(h_b3_u5_cn3b, h_b3_u4_cn3b)
        h_b3_u5_cn3b = tf.nn.relu(h_b3_u5_cn3b)


      # Block 3, unit 6
      with tf.name_scope('block3_unit6b'):

        h_b3_u6_cn1b = self.conv2d(h_b3_u5_cn3b, self.weight["W_b3_u6_cn1"])
        h_b3_u6_cn1b = tf.layers.batch_normalization(inputs=h_b3_u6_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u6_cn1b = tf.nn.relu(h_b3_u6_cn1b)

        h_b3_u6_cn2b = self.conv2d(h_b3_u6_cn1b, self.weight["W_b3_u6_cn2"])
        h_b3_u6_cn2b = tf.layers.batch_normalization(inputs=h_b3_u6_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u6_cn2b = tf.nn.relu(h_b3_u6_cn2b)

        h_b3_u6_cn3b = self.conv2d(h_b3_u6_cn2b, self.weight["W_b3_u6_cn3"])
        h_b3_u6_cn3b = tf.layers.batch_normalization(inputs=h_b3_u6_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b3_u6_cn3b = tf.add(h_b3_u6_cn3b, h_b3_u5_cn3b)
        h_b3_u6_cn3b = tf.nn.relu(h_b3_u6_cn3b)


      # Block 4, unit 1
      with tf.name_scope('block4_unit1b'):

        # Original way to go on a resnet50
        # Calculating the first shortcut
        # shortcut_b4b = self.conv2ds2(h_b3_u6_cn3b, self.weight["W_b4_u1_cn0"])
        # shortcut_b4b = tf.layers.batch_normalization(inputs=shortcut_b4b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        # h_b4_u1_cn1b = self.conv2ds2(h_b3_u6_cn3b, self.weight["W_b4_u1_cn1"])
        # h_b4_u1_cn1b = tf.layers.batch_normalization(inputs=h_b4_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        # h_b4_u1_cn1b = tf.nn.relu(h_b4_u1_cn1b)

        # Modification in the resnet 50 due to excesive reduction of the input data through the blocks
        # The modification is to use the conv2d function insted of the conv2ds2

        # Calculating the shortcut
        shortcut_b4b = self.conv2d(h_b3_u6_cn3b, self.weight["W_b4_u1_cn0"])
        shortcut_b4b = tf.layers.batch_normalization(inputs=shortcut_b4b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)

        h_b4_u1_cn1b = self.conv2d(h_b3_u6_cn3b, self.weight["W_b4_u1_cn1"])
        h_b4_u1_cn1b = tf.layers.batch_normalization(inputs=h_b4_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u1_cn1b = tf.nn.relu(h_b4_u1_cn1b)

        h_b4_u1_cn2b = self.conv2d(h_b4_u1_cn1b, self.weight["W_b4_u1_cn2"])
        h_b4_u1_cn2b = tf.layers.batch_normalization(inputs=h_b4_u1_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u1_cn2b = tf.nn.relu(h_b4_u1_cn2b)

        h_b4_u1_cn3b = self.conv2d(h_b4_u1_cn2b, self.weight["W_b4_u1_cn3"])
        h_b4_u1_cn3b = tf.layers.batch_normalization(inputs=h_b4_u1_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u1_cn3b = tf.add(h_b4_u1_cn3b, shortcut_b4b)
        h_b4_u1_cn3b = tf.nn.relu(h_b4_u1_cn3b)


      # Block 4, unit 2
      with tf.name_scope('block4_unit2b'):

        h_b4_u2_cn1b = self.conv2d(h_b4_u1_cn3b, self.weight["W_b4_u2_cn1"])
        h_b4_u2_cn1b = tf.layers.batch_normalization(inputs=h_b4_u2_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u2_cn1b = tf.nn.relu(h_b4_u2_cn1b)

        h_b4_u2_cn2b = self.conv2d(h_b4_u2_cn1b, self.weight["W_b4_u2_cn2"])
        h_b4_u2_cn2b = tf.layers.batch_normalization(inputs=h_b4_u2_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u2_cn2b = tf.nn.relu(h_b4_u2_cn2b)

        h_b4_u2_cn3b = self.conv2d(h_b4_u2_cn2b, self.weight["W_b4_u2_cn3"])
        h_b4_u2_cn3b = tf.layers.batch_normalization(inputs=h_b4_u2_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u2_cn3b = tf.add(h_b4_u2_cn3b, h_b4_u1_cn3b)
        h_b4_u2_cn3b = tf.nn.relu(h_b4_u2_cn3b)


      # Block 4, unit 3
      with tf.name_scope('block4_unit3b'):

        h_b4_u3_cn1b = self.conv2d(h_b4_u2_cn3b, self.weight["W_b4_u3_cn1"])
        h_b4_u3_cn1b = tf.layers.batch_normalization(inputs=h_b4_u3_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u3_cn1b = tf.nn.relu(h_b4_u3_cn1b)

        h_b4_u3_cn2b = self.conv2d(h_b4_u3_cn1b, self.weight["W_b4_u3_cn2"])
        h_b4_u3_cn2b = tf.layers.batch_normalization(inputs=h_b4_u3_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u3_cn2b = tf.nn.relu(h_b4_u3_cn2b)

        h_b4_u3_cn3b = self.conv2d(h_b4_u3_cn2b, self.weight["W_b4_u3_cn3"])
        h_b4_u3_cn3b = tf.layers.batch_normalization(inputs=h_b4_u3_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.phase, fused=True)
        h_b4_u3_cn3b = tf.add(h_b4_u3_cn3b, h_b4_u2_cn3b)
        h_b4_u3_cn3b = tf.nn.relu(h_b4_u3_cn3b)

      with tf.name_scope('pool1b'):
        h_pool2b = self.avg_pool_2x2(h_b4_u3_cn3b)


      # Fully connected
      with tf.name_scope('fc1'):
        h_concat = tf.concat([h_pool2a, h_pool2b], axis=3)
        h_concat_flat = tf.reshape(h_concat, [-1,2 * 2048 * HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV])

        Y_fc1 = tf.nn.relu(tf.matmul(h_concat_flat, self.weight["W_fc1"]))

        self.Y_logt = tf.matmul(Y_fc1, self.weight["W_fc2"])
        self.Y_pred = tf.nn.softmax(self.Y_logt, name='Y_pred')

        h_pool2a_flat = tf.reshape(h_pool2a, [-1,2048 * HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV])
        h_pool2b_flat = tf.reshape(h_pool2b, [-1,2048 * HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV])

        self.Y_logta = tf.matmul(h_pool2a_flat, self.weight["W_fc1a"])
        self.Y_preda = tf.nn.softmax(self.Y_logta, name='Y_preda')

        self.Y_logtb = tf.matmul(h_pool2b_flat, self.weight["W_fc1a"])
        self.Y_predb = tf.nn.softmax(self.Y_logtb, name='Y_predb')


  # Defining the output, the output it is collapsed to a value of 0,1 or 1,0 using the argmax function
  # Initially the value was the result of a softmax which make the result of the fully connected layer
  # and transform the result into real values in the range of (0,1) that add up to 1
  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output'):
      self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
      self.label_true = tf.argmax(self.Y, 1, name='label_true')

      self.label_preda = tf.argmax(self.Y_preda, 1, name='label_preda')
      self.label_truea = tf.argmax(self.Ya, 1, name='label_truea')

      self.label_predb = tf.argmax(self.Y_predb, 1, name='label_predb')
      self.label_trueb = tf.argmax(self.Yb, 1, name='label_trueb')


  # Defining the loss used, in the this case is used a crossed entropy and as a regularizer it is used
  # the l2 norm. This is done to avoid overfitting of the model.
  # Officialy the regularizer should be calculated with the weights of the whole model, however
  # for this kind of model the weights of the last or second last layer is enough
  # Another approach is the use the rms of the predicted output vs the desired output
  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss'):

      _WEIGHT_DECAY = 0.01
      
      #cross entropy
      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
      self.loss = tf.reduce_mean(self.cross_entropy)
      regularizer = tf.nn.l2_loss(self.weight["W_fc1"])
      self.loss = tf.reduce_mean(self.loss + _WEIGHT_DECAY * regularizer)

      self.cross_entropya = tf.nn.softmax_cross_entropy_with_logits(labels=self.Ya, logits=self.Y_logta)
      self.lossa = tf.reduce_mean(self.cross_entropya)
      regularizera = tf.nn.l2_loss(self.weight["W_fc1a"])
      self.lossa = tf.reduce_mean(self.lossa + _WEIGHT_DECAY * regularizera)

      self.cross_entropyb = tf.nn.softmax_cross_entropy_with_logits(labels=self.Yb, logits=self.Y_logtb)
      self.lossb = tf.reduce_mean(self.cross_entropyb)
      regularizerb = tf.nn.l2_loss(self.weight["W_fc1a"])
      self.lossb = tf.reduce_mean(self.lossb + _WEIGHT_DECAY * regularizerb)

  # Calculating the accuracy per batch of the system
  # This let us know if the model is actually improving, or if it is not learning anything at all
  # There are two metrics here, the number of positive results and the percentaje of positive results
  # in the batch, the first is to have the accumulated result since the beginning, the second is to
  # watch how it is improving the model 
  def def_metrics(self):
    """ Adds metrics """
    with tf.name_scope('metrics'):
      cmp_labels = tf.equal(self.label_true, self.label_pred)
      self.accuracy = tf.reduce_sum(tf.cast(cmp_labels, tf.float32), name='accuracy')
      self.acc_batch = (self.accuracy/self.size_batch)*100
      #self.accuracy = tf.reduce_mean(tf.cast(cmp_labels, tf.float32), name='accuracy')

      cmp_labelsa = tf.equal(self.label_truea, self.label_preda)
      self.accuracya = tf.reduce_sum(tf.cast(cmp_labelsa, tf.float32), name='accuracy_a')
      self.acc_batcha = (self.accuracya/self.size_batch)*100

      cmp_labelsb = tf.equal(self.label_trueb, self.label_predb)
      self.accuracyb = tf.reduce_sum(tf.cast(cmp_labelsb, tf.float32), name='accuracy_b')
      self.acc_batchb = (self.accuracyb/self.size_batch)*100

  # This is to plot the results and to see how is the lost and the accuracy actually working
  # The accuracy that it is drawn, it is the accuracy per batch
  def add_summaries(self):
    """ Adds summaries for Tensorboard """
    # defines a namespace for the summaries
    with tf.name_scope('summaries'):
      # adds a plot for the loss
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('loss_a', self.lossa)
      tf.summary.scalar('loss_b', self.lossb)

      #tf.summary.scalar('accuracy', self.accuracy)
      tf.summary.scalar('accuracy', self.acc_batch)
      tf.summary.scalar('accuracy_a', self.acc_batcha)
      tf.summary.scalar('accuracy_b', self.acc_batchb)
      # groups summaries
      self.summary = tf.summary.merge_all()

  def train(self):

    # Creating a folder where to save the parameters of the model aka weights
    file_path = str(sys.argv[0]) +'_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs)

    # Creating a file to write the loss and acurracy
    output_file = open(file_path+'_results.txt', 'w')

    """ Trains the model """
    # setup minimize function, this is the optimizer of the function to be minimized according to the loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss)
    optimizera = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.lossa)
    optimizerb = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.lossb)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    # opens session
    with tf.Session() as sess:
      
      # writers for TensorBorad, this is to make the tensorboard graphs
      train_writer = tf.summary.FileWriter('graphs/' +str(sys.argv[0]) +'_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      valid_writer = tf.summary.FileWriter('graphs/' +str(sys.argv[0]) +'_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      test_writer = tf.summary.FileWriter('graphs/' +str(sys.argv[0]) +'_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      train_writer.add_graph(sess.graph)

      extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

      # initialize variables (params)
      sess.run(init_op)

      # The save let us save the parameters of the model
      saver = tf.train.Saver()      

      # In case we need to restore the weights of another model we should do it like this:
      # saver.restore(sess, "resnet_V20_Xavier_GRAD_SPEC_1S_WM.py_0.001_10/final_weights.ckpt")      
      # Note: the model must have exactly the same architecture or it is not going to be loaded, in case 
      # you just need some of the weights, you must save some especific parameters and load those 
      # specific paramaters

      # Initializing the step for train, validation and test
      # This is done to not show the results in every iteration but every certain amount of spets
      step_train = 1
      step_valid = 1
      step_test = 1
      acc_train = 0
      acc_valid = 0
      acc_test = 0
      
      # Compute for the desired number of epochs.
      for n_epochs in range(FLAGS.num_epochs):

        # This is to define which data you run, in the last epoch you sweep all the train, valid and test data
        # in the rest of the epochs you just sweep train and valid
        if n_epochs == FLAGS.num_epochs - 1:
          k_limit = 3
        else:
          k_limit = 2

        # This is to run in k=0 the train in k = 1 the test and in the last epoch in k = 3 the test
        # It is done in a cycle because, the code is always the same the only thing that changes 
        # is the location of the files
        for step_file in range(0,k_limit):

          # Audio
          if step_file == 0:
            total_data = TOTAL_DATA_TRAIN
            input_file  = open(FILE_TRAIN,'r')

          elif step_file == 1:      
            total_data = TOTAL_DATA_VALID
            input_file  = open(FILE_VALID,'r')

          else:
            total_data = TOTAL_DATA_TEST
            input_file  = open(FILE_TEST,'r')

          # Determing the rows of the file to write
          #if total_data < ROWS_FILE:
          if total_data < FLAGS.batch_size:
            rows_tf = total_data
          else:
            rows_tf = FLAGS.batch_size

          # Moving audio routes to a matrix
          matrix = []
          for line in input_file:
            row = line.rstrip()
            matrix.append(row)

          # Permutating the data to do not have the data in the same order in every epoch
          permutation_m = np.random.permutation(len(matrix))
          matrix = np.array(matrix)
          matrix = matrix[permutation_m]
          matrix = matrix.tolist()

          data_per_row = int(np.ceil(float(total_data)/float(2*len(matrix))))
          rows = len(matrix)

          # Creating the vector that are going to be fill in the next steps
          X1 = []
          X2 = []
          Y = []
          Ya = []
          Yb = []

          i = 0
          total = 0
          j = 0

          # Doing the cycle while I still have data to do
          while i < total_data:

            chosen_audio_1 = matrix[j]
            audio_1,samplerate = sf.read(chosen_audio_1)
            fixed_class = str(get_int(get_part(PART,chosen_audio_1))).zfill(4)

            if audio_1.shape[0]>WINDOW and np.sqrt(np.mean(np.abs(audio_1))) > VAD:        

              # number of times each audio is going to be used
              for k in range(0,data_per_row):

                flag_model = False

                # All the audios will be mixed with one of its own and one different
                if j+1 <rows and str(get_int(get_part(PART,matrix[j+1]))).zfill(4) == fixed_class:
                  chosen_audio_2 = matrix[j+1]
                elif j-1 >=0 and str(get_int(get_part(PART,matrix[j-1]))).zfill(4) == fixed_class:
                  chosen_audio_2 = matrix[j-1]
                else:
                  chosen_audio_2 = matrix[j]

                audio_2,samplerate = sf.read(chosen_audio_2)

                if audio_2.shape[0]>WINDOW and np.sqrt(np.mean(np.abs(audio_2))) > VAD:
                  
                  # Comparing with a data of the same speaker
                  while True:   

                      # Getting a random index for each audio
                      index_a1 = random.randrange(0,audio_1.shape[0]-WINDOW,1)
                      index_a2 = random.randrange(0,audio_2.shape[0]-WINDOW,1)

                      # Analyzing the VAD o the signal using the rms
                      if np.sqrt(np.mean(np.abs(audio_1[index_a1:index_a1+WINDOW]))) > VAD and np.sqrt(np.mean(np.abs(audio_2[index_a2:index_a2+WINDOW]))) > VAD:

                        f, t, Sxx1 = signal.spectrogram(audio_1[index_a1:index_a1+WINDOW], samplerate,  window=('hamming'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1)
                        f, t, Sxx2 = signal.spectrogram(audio_2[index_a2:index_a2+WINDOW], samplerate,  window=('hamming'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1)

                        Hxx1 = StandardScaler().fit_transform(Sxx1)
                        Hxx2 = StandardScaler().fit_transform(Sxx2)

                        data_audio_1 = np.reshape(Hxx1[0:SIZE_FFT,:],(SIZE_FFT,SIZE_COLS,1))
                        data_audio_2 = np.reshape(Hxx2[0:SIZE_FFT,:],(SIZE_FFT,SIZE_COLS,1))

                        # Filling the matrixes with the data
                        X1.append(data_audio_1)
                        X2.append(data_audio_2)
                        Y.append([0,1])

                        # Adding the Ya and Yb 
                        Y_Aux_a = np.zeros(L_CLASS) 
                        Y_Aux_b = np.zeros(L_CLASS) 

                        if step_file == 0:
                          if list_speakers.index(get_int(get_part(PART,chosen_audio_1))) < permited_speakers:
                            Y_Aux_a[list_speakers.index(get_int(get_part(PART,chosen_audio_1)))] = 1
                          if list_speakers.index(get_int(get_part(PART,chosen_audio_2))) < permited_speakers:
                            Y_Aux_b[list_speakers.index(get_int(get_part(PART,chosen_audio_2)))] = 1

                        Ya.append(Y_Aux_a)
                        Yb.append(Y_Aux_b)

                        total+=1
                        i+=1
                        break;

                  if total>= rows_tf:
                    flag_model = True

                # Comparing with data of a different speaker
                while flag_model == False:
                  chosen_audio_2 = matrix[random.randrange(0,rows,1)]

                  if str(get_int(get_part(PART,chosen_audio_2))).zfill(4)!=fixed_class:
                    
                    audio_2,samplerate = sf.read(chosen_audio_2)

                    if audio_2.shape[0]>WINDOW:

                      # Getting a random index for each audio
                      index_a1 = random.randrange(0,audio_1.shape[0]-WINDOW,1)
                      index_a2 = random.randrange(0,audio_2.shape[0]-WINDOW,1)

                      # Analyzing the VAD o the signal using the rms
                      if np.sqrt(np.mean(np.abs(audio_1[index_a1:index_a1+WINDOW]))) > VAD and np.sqrt(np.mean(np.abs(audio_2[index_a2:index_a2+WINDOW]))) > VAD:

                        f, t, Sxx1 = signal.spectrogram(audio_1[index_a1:index_a1+WINDOW], samplerate,  window=('hamming'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1)
                        f, t, Sxx2 = signal.spectrogram(audio_2[index_a2:index_a2+WINDOW], samplerate,  window=('hamming'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1)

                        Hxx1 = StandardScaler().fit_transform(Sxx1)
                        Hxx2 = StandardScaler().fit_transform(Sxx2)

                        data_audio_1 = np.reshape(Hxx1[0:SIZE_FFT,:],(SIZE_FFT,SIZE_COLS,1))
                        data_audio_2 = np.reshape(Hxx2[0:SIZE_FFT,:],(SIZE_FFT,SIZE_COLS,1))

                        # Filling the matrixes with the data
                        X1.append(data_audio_1)
                        X2.append(data_audio_2)
                        Y.append([1,0])

                        # Adding the Ya and Yb 
                        Y_Aux_a = np.zeros(L_CLASS) 
                        Y_Aux_b = np.zeros(L_CLASS) 

                        if step_file == 0:
                          if list_speakers.index(get_int(get_part(PART,chosen_audio_1))) < permited_speakers:
                            Y_Aux_a[list_speakers.index(get_int(get_part(PART,chosen_audio_1)))] = 1
                          if list_speakers.index(get_int(get_part(PART,chosen_audio_2))) < permited_speakers:
                            Y_Aux_b[list_speakers.index(get_int(get_part(PART,chosen_audio_2)))] = 1

                        Ya.append(Y_Aux_a)
                        Yb.append(Y_Aux_b)

                        total+=1
                        i+=1
                        break;

                if total>= rows_tf:
                    flag_model = True

                # If there is enough data for a batch
                if flag_model == True:

                  X1_array = np.array(X1)
                  X2_array = np.array(X2)
                  Y_array = np.array(Y)
                  Ya_array = np.array(Ya)
                  Yb_array = np.array(Yb)

                  permutation = np.random.permutation(X1_array.shape[0])
                  X1_array = X1_array[permutation,:]
                  X2_array = X2_array[permutation,:]
                  Y_array = Y_array[permutation]
                  Ya_array = Ya_array[permutation]
                  Yb_array = Yb_array[permutation]            


                  # Running the apropiate model
                  # Train
                  if step_file == 0:

                    # evaluation with train data, we feed the data to the network
                    feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array, self.Ya : Ya_array, self.Yb : Yb_array, self.phase:1}

                    # Evaluation for verification
                    fetches = [optimizer, self.loss, self.accuracy, self.summary,  extra_update_ops]
                    _,train_loss, train_acc, train_summary,_ = sess.run(fetches, feed_dict=feed_dict)

                    # Evaluation for clasification A
                    fetches = [optimizera, self.lossa, self.accuracya, self.summary, extra_update_ops]
                    _,train_lossa, train_acca, train_summary,_ = sess.run(fetches, feed_dict=feed_dict)

                    # Evaluation for clasification B
                    fetches = [optimizerb, self.lossb, self.accuracyb, self.summary, extra_update_ops]
                    _,train_lossb, train_accb, train_summary,_ = sess.run(fetches, feed_dict=feed_dict)

                    train_writer.add_summary(train_summary, step_train)

                    acc_train = acc_train + train_acc

                    # Printing the results every 100 batch
                    if step_train % 100 == 0:
                      msg = "I{:3d} loss_train: ({:6.8f}), acc_train(batch, global, a, b): ({:6.8f},{:6.8f},{:6.8f},{:6.8f})"
                      msg = msg.format(step_train, train_loss, train_acc/FLAGS.batch_size, acc_train/(FLAGS.batch_size*step_train), train_acca, train_accb)
                      print(msg)
                      output_file.write(msg + '\n')

                    step_train += 1

                  # Validation
                  elif step_file == 1:
                    
                    # evaluation with Validation data
                    feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array, self.Ya : Ya_array, self.Yb : Yb_array, self.phase:0}
                    fetches = [self.loss, self.accuracy, self.summary]
                    valid_loss, valid_acc, valid_summary = sess.run(fetches, feed_dict=feed_dict)
                    valid_writer.add_summary(valid_summary, step_train)

                    acc_valid = acc_valid + valid_acc

                    if step_valid % 100 == 0:
                      msg = "I{:3d} loss_val: ({:6.8f}), acc_val(batch, global): ({:6.8f},{:6.8f})"
                      msg = msg.format(step_valid, valid_loss, valid_acc/FLAGS.batch_size, acc_valid/(FLAGS.batch_size*step_valid))
                      print(msg)
                      output_file.write(msg + '\n')

                    step_valid += 1

                  # Test
                  else:

                    # evaluation with Test data
                    feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array, self.Ya : Ya_array, self.Yb : Yb_array, self.phase:0}
                    fetches = [self.loss, self.accuracy, self.summary]
                    test_loss, test_acc, test_summary = sess.run(fetches, feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step_train)

                    acc_test = acc_test + test_acc

                    if step_test % 100 == 0:
                      msg = "I{:3d} loss_test: ({:6.8f}), acc_test(batch, global): ({:6.8f},{:6.8f})"
                      msg = msg.format(step_test, test_loss, test_acc/FLAGS.batch_size, acc_test/(FLAGS.batch_size*step_test))
                      print(msg)
                      output_file.write(msg + '\n')

                    step_test += 1

                  total = 0
                  X1 = []
                  X2 = []
                  Y = []
                  Ya = []
                  Yb = []
            
            j+=1
            

      os.mkdir(file_path)
      
      # Saving the weights just once (no in every epoch) 
      save_path = saver.save(sess, str(file_path+'/final_weights.ckpt') )


def run():

  print('Inicio resnet')

  # defines our model
  model = resnet()

  print('termino resnet') 

  # trains our model
  model.train()


def main(args):
  run()
  return 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=10,
      help='Number of epochs to run trainer.'
  )

  parser.add_argument(
      '--batch_size',
      type=int,
      default=10,
      help='Batch size.'
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)