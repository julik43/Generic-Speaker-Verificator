# -*- coding: utf-8 -*-

# """ Resnet
# UNAM IIMAS
# Author: Ivette Velez
# Tutor:  Caleb Rascon
# Co-tutor: Gibran Fuentes
# To run the model: python verificador_resnet
# """

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

"""Train and Eval the MNIST network.
This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/programmers_guide/reading_data#reading_from_files
for context.
YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import os
import sys
import time
import glob
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
import sounddevice as sd
from scipy import ndimage
from scipy import signal
import sklearn.preprocessing as pps
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib import layers

from python_speech_features import mfcc
from python_speech_features import logfbank

#from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Using just one GPU in case of GPU 
#os.environ['CUDA_VISIBLE_DEVICES']= '0'
os.environ['CUDA_VISIBLE_DEVICES']= '0'

# Basic model parameters as external flags.
FLAGS = None

# To don't have problems with the model the number of permited speakers is left
# with the number that the machine was trained
permited_speakers = 100

# Lenght of the audio to be recorded
lengt_audio = 1
sample_rate = 16000

# Parameter for the batch normalization with relu
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# Constants on the model
WINDOW = 1*16000 # with audios of librispeech it means 1 seconds
PART = 2
MS = 1.0/16000
NPERSEG = int(0.025/MS)
NOVERLAP = int(0.015/MS)
#NFFT =1024
NFFT =NPERSEG
#SIZE_FFT = int(NFFT/2)
#SIZE_FFT = int(NFFT/4)
SIZE_FFT = 32
SIZE_COLS = int(np.ceil((WINDOW - NPERSEG)/(NPERSEG - NOVERLAP)))

print(SIZE_COLS)
print(SIZE_FFT)

VAD = 0.05

# Audio
L_LABEL = 2
#L_CLASS = len(list_speakers)
L_CLASS = permited_speakers
IN_HEIGHT = SIZE_FFT
IN_WIDTH = SIZE_COLS
CHANNELS = 1
POOL_LAYERS = 3

WIDTH_AFTER_CONV = int(np.ceil(float(IN_WIDTH)/float(2**POOL_LAYERS)))
HEIGHT_AFTER_CONV = int(np.ceil(float(IN_HEIGHT)/float(2**POOL_LAYERS)))

print(WIDTH_AFTER_CONV)
print(HEIGHT_AFTER_CONV)

# Function that splits and string to get the desire part
def get_part(part,string):
  aux = string.split('/')
  a = aux[len(aux)-part-1]
  return a

# function that gets the numeric part of a string
def get_int(string):
  return int(''.join([d for d in string if d.isdigit()]))

def get_class(string):
  # Finding the name of the class
  class_index = string.rfind('/')
  fixed_class = string[class_index+1:len(string)]

  class_index = fixed_class.rfind('.')
  complete_class = fixed_class[0:class_index]

  class_index = fixed_class.rfind('_')
  if class_index >= 0:
    fixed_class = fixed_class[0:class_index]

  class_index = fixed_class.find('-')
  if class_index >= 0:
    fixed_class = fixed_class[0:class_index]

  return fixed_class

def get_number_audio(string):

  class_index = string.rfind('/')
  fixed_class = string[class_index+1:len(string)]
  class_index = fixed_class.rfind('.')
  complete_class = fixed_class[0:class_index]

  class_index = complete_class.rfind('_')
  if class_index >= 0:
    number_of_audio = complete_class[class_index+1:]

  return number_of_audio

class resnet_v1:

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

  def conv2d(self, x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def conv2ds2(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

  # def max_pool_2x1(self, x):
  #   """max_pool_2x1 downsamples a feature map by 2X."""
  #   return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

  def max_pool_2x2(self, x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def avg_pool_2x2(self,x):
    return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],strides=[1, 2, 2, 1], padding='SAME')


  def weight_variable(self,shape):
    initializer = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initializer)
    
  def bias_variable(self,shape):
    initializer = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initializer)


  def def_input(self):
    """ Defines inputs """
    with tf.name_scope('input'):

      # Defining the entrance of the model
      self.X1 = tf.placeholder(tf.float32, [None, IN_HEIGHT, IN_WIDTH, CHANNELS], name='X1')
      self.X2 = tf.placeholder(tf.float32, [None, IN_HEIGHT, IN_WIDTH, CHANNELS], name='X2')
      self.Y = tf.placeholder(tf.float32, [None, L_LABEL], name='Y')
      self.Ya = tf.placeholder(tf.float32, [None, L_CLASS], name='Ya')
      self.Yb = tf.placeholder(tf.float32, [None, L_CLASS], name='Yb')


      self.g_step = tf.contrib.framework.get_or_create_global_step()
      self.phase = tf.placeholder(tf.bool, name='phase')

  def def_variable(self):
    self.size_batch = float(FLAGS.batch_size)

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

        # Calculating the first shortcut
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

        # Calculating the first shortcut
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

        # Calculating the first shortcut
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

        # Calculating the first shortcut
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

        # self.Y_logt = tf.matmul(h_concat_flat, self.weight["W_fc1"])
        # self.Y_pred = tf.nn.softmax(self.Y_logt, name='Y_pred')

        Y_fc1 = tf.nn.relu(tf.matmul(h_concat_flat, self.weight["W_fc1"]))

        self.Y_logt = tf.matmul(Y_fc1, self.weight["W_fc2"])
        self.Y_pred = tf.nn.softmax(self.Y_logt, name='Y_pred')

        h_pool2a_flat = tf.reshape(h_pool2a, [-1,2048 * HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV])
        h_pool2b_flat = tf.reshape(h_pool2b, [-1,2048 * HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV])


        self.Y_logta = tf.matmul(h_pool2a_flat, self.weight["W_fc1a"])
        self.Y_preda = tf.nn.softmax(self.Y_logta, name='Y_preda')

        self.Y_logtb = tf.matmul(h_pool2b_flat, self.weight["W_fc1a"])
        self.Y_predb = tf.nn.softmax(self.Y_logtb, name='Y_predb')


    # # Concat layer to go from convolutional layer to fully connected
    # with tf.name_scope('concat1'):
    #   h_concat1 = tf.concat([h_pool2a, h_pool2b], axis=3)

  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output'):
      self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
      self.label_true = tf.argmax(self.Y, 1, name='label_true')

      self.label_preda = tf.argmax(self.Y_preda, 1, name='label_preda')
      self.label_truea = tf.argmax(self.Ya, 1, name='label_truea')

      self.label_predb = tf.argmax(self.Y_predb, 1, name='label_predb')
      self.label_trueb = tf.argmax(self.Yb, 1, name='label_trueb')

  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss'):

      _WEIGHT_DECAY = 0.01

      # self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=self.Y, logits=self.Y_logt)
      # self.loss = tf.losses.get_total_loss()

      #cross entropy
      #self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)*0.00001
      #self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)*0.01
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

    # Creating a folder where to save the parameter
    path_heatmap = str(sys.argv[0]) + '_' 
    Y_heatmap = []

    if os.path.exists('resultados_equilibrio') == False:
      os.mkdir('resultados_equilibrio')
    
    # setup minimize function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss)
    optimizera = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.lossa)
    optimizerb = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.lossb)

    #optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    # opens session
    with tf.Session() as sess:
      
      # writers for TensorBorad
      train_writer = tf.summary.FileWriter('graphs/resnet_verificator_train' )
      valid_writer = tf.summary.FileWriter('graphs/resnet_verificator_valid' )
      test_writer = tf.summary.FileWriter('graphs/resnet_verificator_test')
      train_writer.add_graph(sess.graph)

      extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

      # initialize variables (params)
      # sess.run(tf.global_variables_initializer())
      sess.run(init_op)
      # print('Llego 0')
      # sys.exit(0)

      saver = tf.train.Saver()
      saver.restore(sess, "/home/ar/ivette_thesis/models_aira/resnet_librispeech_noisy/resnet_step2.py_0.001_10/9weights.ckpt")

      for num_loc in range(1,11):
        
        for num_audios in xrange(1,11):

          # Creating a folder where to save the parameter
          file_path = str(sys.argv[0]) + '_loc_' + str(num_loc)+ '_audios_' + str(num_audios) + '_'

          Y_true = []
          Y_pred = []

          # Initializing the step for train and validation
          step_train = 1
          step_valid = 1
          step_test = 1
          acc_train = 0
          acc_valid = 0
          acc_test = 0

          # Determing the while condition depending on the way of working automatic or manual
          if FLAGS.automatic == True:
            flag_condition = True
          else:
            flag_condition = False

          # Asking it the automatic mode is activated
          if FLAGS.automatic == True:
            # loading the file with the audios to be verified

            # Adquiring the data for the database
            database_aux = glob.glob( os.path.join(FLAGS.database, '*.npy') )          

            database = []

            # Looking for all the classes
            all_classes = []
            for audio in xrange(0,len(database_aux)):
              chosen_audio_1 = database_aux[audio]
              fixed_class = get_class(chosen_audio_1)
              all_classes.append(fixed_class)

            all_classes = np.unique(np.array(all_classes))
            
            # The classes chosen are selected depending on the number
            chosen_classes = all_classes[0:num_loc]

            for audio in xrange(0,len(database_aux)):                

              chosen_audio_1 = database_aux[audio]
              fixed_class = get_class(chosen_audio_1)
              number_audio = get_number_audio(chosen_audio_1)

              if int(number_audio) <= num_audios and (fixed_class in chosen_classes) == True:
              #if int(number_audio) < num_audios and (fixed_class in chosen_classes) == True:
                database.append(database_aux[audio])



            # loading the file with the audios to be verified
            input_file  = open(FLAGS.verification_file,'r')

            aux_ver_audio = []
            for line in input_file:
              row = line.rstrip()
              aux_ver_audio.append(row)

            ver_audio = []

            for audio in xrange(0,len(aux_ver_audio)):                

              chosen_audio_1 = aux_ver_audio[audio]
              fixed_class = get_class(chosen_audio_1)

              if (fixed_class in chosen_classes) == True:
                ver_audio.append(aux_ver_audio[audio])

            input_file  = open(FLAGS.unk_file,'r')

            for line in input_file:
              row = line.rstrip()
              ver_audio.append(row)

            ver_audio = np.array(ver_audio)
            

          i_ver_audio = 0

          # For the verificator there is not any need of sweeping all the data, but to enter the data for the new
          # speaker and verify the information given
          while True:

            #print('entro al while')

            X1 = []
            X2 = []
            Y = []
            Ya = []
            Yb = []
            Y_Aux_a = []
            Y_Aux_b = []
            Y_complete_a = []
            Y_complete_b = []        
            total_ver_ypred = []
            total_ver_lpred = []


            if flag_condition == True:
              chosen = ver_audio[i_ver_audio]
              recording, samplerate = sf.read(ver_audio[i_ver_audio])
              new_class = get_class(chosen)

              # Verifying if there is any data to be processed
              if np.sqrt(np.mean(np.abs(recording))) <= VAD:            
                #print('The audio selected does not pass the VAD threshold, please repeat the action')
                flag_VAD = False
              else:
                flag_VAD = True


            # Verifying if the VAD threshold is passed
            if flag_VAD ==  True:

              init_time = time.time()

              audio = 0

              # Creating the data with the known audios this is going to be stored in the X1
              #for audio in xrange(0,len(database)):
              while audio < len(database):

                # Selecting the audio
                chosen_audio_1 = database[audio]
                data_audio_1 = np.load(chosen_audio_1)
                fixed_class = get_class(chosen_audio_1)

                X1.append(data_audio_1)
                Ya.append(np.zeros(L_CLASS))
                Y_Aux_a.append(fixed_class)
                Y_complete_a.append(fixed_class)
                Y.append([0,0])


                # Selecting the audio
                # chosen_audio_2 = database[audio]
                # audio_2,samplerate = sf.read(chosen_audio_2)
                if flag_condition == True:
                  chosen_audio_2 = ver_audio[i_ver_audio]
                  audio_2 = recording
                  fixed_class = get_class(chosen_audio_2)

                if audio_2.shape[0]>WINDOW and np.sqrt(np.mean(np.abs(audio_2))) > VAD:       

                  # Comparing with a data of the same speaker
                  while True:   
                  
                    # Getting a random index for each audio
                    index_a2 = random.randrange(0,audio_2.shape[0]-WINDOW,1) 

                    # Analyzing the VAD o the signal using the rms
                    if np.sqrt(np.mean(np.abs(audio_2[index_a2:index_a2+WINDOW]))) > VAD:

                      f, t, Sxx2 = signal.spectrogram(audio_2[index_a2:index_a2+WINDOW], samplerate,  window=('hamming'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1)
                      Hxx2 = StandardScaler().fit_transform(Sxx2)
                      data_audio_2 = np.reshape(Hxx2[0:SIZE_FFT,:],(SIZE_FFT,SIZE_COLS,1))
                      
                      # Filling the matrixes with the data
                      X2.append(data_audio_2)
                      Yb.append(np.zeros(L_CLASS))
                      Y_Aux_b.append(fixed_class)
                      Y_complete_b.append(fixed_class)

                      audio+=1
                      break;
              

              name_speaker = ''

              # If there are no audios to compare the audio is stored
              if len(database) == 0:

                if flag_condition == False:
                  #print('Storing the new audio because there is not any known speakers')
                  print('There is not any known speakers')
                
              # If there are audios to compare 
              else:

                # Sending the data according to the size of the batch

                # The model can not work with more than a certain amount of data, just the batch size, 
                # so it has to be cutted depeding on the size of the batch
                amount_audio = 0

                while amount_audio < len(database):

                  if amount_audio + FLAGS.batch_size > len(database):

                    X1_array = np.array(X1[amount_audio:len(database)])
                    X2_array = np.array(X2[amount_audio:len(database)])
                    Y_array = np.array(Y[amount_audio:len(database)])
                    Ya_array = np.array(Ya[amount_audio:len(database)])
                    Yb_array = np.array(Yb[amount_audio:len(database)])
                  
                  else:
                    limit_array = amount_audio+FLAGS.batch_size
                    X1_array = np.array(X1[amount_audio:limit_array])
                    X2_array = np.array(X2[amount_audio:limit_array])
                    Y_array = np.array(Y[amount_audio:limit_array])
                    Ya_array = np.array(Ya[amount_audio:limit_array])
                    Yb_array = np.array(Yb[amount_audio:limit_array])

                  #init_time = time.time()

                  feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array, self.Ya : Ya_array, self.Yb : Yb_array, self.phase:0}
                  #fetches = [self.loss, self.accuracy, self.Y_pred, self.label_pred]
                  fetches = [self.Y_pred, self.label_pred]
                  ver_ypred, ver_lpred = sess.run(fetches, feed_dict=feed_dict)

                  # Updating the total results
                  total_ver_ypred = total_ver_ypred + ver_ypred.tolist()
                  total_ver_lpred = total_ver_lpred + ver_lpred.tolist()

                  amount_audio = amount_audio + FLAGS.batch_size

              end_time = time.time()


              # Storing the audio 
              if flag_condition == True:

                # Naming the classes to get the average
                unique_classes = np.unique(np.array(Y_Aux_a))

                # Getting the values per class
                class_value = np.zeros((num_loc))

                for number_class in range(0,num_loc):

                  for row_pred in xrange(0,len(total_ver_ypred)):

                    if Y_Aux_a[row_pred] == unique_classes[number_class]:
                      class_value[number_class] = class_value[number_class] + total_ver_ypred[row_pred][1]

                  class_value[number_class] = class_value[number_class]/num_audios
                  

                number_audio = 0
                name_speaker = new_class

                if (name_speaker in Y_Aux_a) == False:
                  name_speaker = 'unk'


                if len(database) > 0:

                  # Initializing the value of the class
                  value_class = 'unk'
                  value_y_pred = 0

                  # Choosing only the class with the highest score above 0.5
                  for row_pred in xrange(0,class_value.shape[0]):
                    
                    if class_value[row_pred] > 0.5 and class_value[row_pred] > value_y_pred:

                      value_class = unique_classes[row_pred]
                      value_y_pred = class_value[row_pred]


                  Y_true.append(name_speaker)
                  Y_pred.append(value_class)

              
              # if there is a  list of the audios to verify, we mus sweep it
              if flag_condition == True:
                if i_ver_audio == ver_audio.shape[0] - 1:

                  # print('Y_true: ', Y_true)
                  # print('Y_pred: ', Y_pred)

                  s_precision, s_recall, s_f1, _ = precision_recall_fscore_support(Y_true, Y_pred, average='micro')
                  s_accuracy = accuracy_score(Y_true, Y_pred)

                  print('Locutores: ', num_loc, ' audios: ', num_audios)
                  print('Precision: ', s_precision)
                  print('Recall: ', s_recall)
                  print('F1: ', s_f1)
                  print('Accuracy: ', s_accuracy)

                  Y_heatmap.append([num_loc, num_audios, s_accuracy])

                  np.save(str('resultados_equilibrio/'+file_path+'Y_true'), Y_true)
                  np.save(str('resultados_equilibrio/'+file_path+'Y_pred'), Y_pred)


                  break
                else:
                  i_ver_audio+= 1

      np.save(str(path_heatmap+'Y_heatmap'), Y_heatmap)

def run():

  # defines our model
  model = resnet_v1()

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
      default=0.001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=10,
      help='Batch size.'
  )
  parser.add_argument(
      '--database',
      type=str,
      default='/home/ar/ivette_thesis/data_bases/database_32_libriaira',
      help='Directory with the data to be tested and stored.'
  )
  parser.add_argument(
      '--automatic',
      type=bool,
      #default=False,
      default=True,
      help='Parameter that specifies if the process is automatic (True) or manual (False)'
  )
  parser.add_argument(
      '--verification_file',
      type=str,
      default='verification_file.txt',
      help='List of audio files to be verified'
  )
  parser.add_argument(
      '--unk_file',
      type=str,
      default='file_unknown.txt',
      help='List of unknown audio files to be verified'
  )
  parser.add_argument(
      '--loc',
      type=int,
      default=0,
      help='List of audio files to be verified'
  )
  parser.add_argument(
      '--audios',
      type=int,
      default=0,
      help='List of audio files to be verified'
  )


  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



# if __name__ == '__main__':
#   import sys
#   sys.exit(main(sys.argv))
