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

# Using just one GPU in case of GPU 
os.environ['CUDA_VISIBLE_DEVICES']= '0'

# Basic model parameters as external flags.
FLAGS = None

# Files for the models
FILE_TRAIN = 'train_speakers.txt'
FILE_VALID = 'valid_speakers.txt'
FILE_TEST = 'test_speakers.txt'

# ==============================================================================
# Constants on the model
# ==============================================================================
WINDOW = 1*16000 # with audios of librispeech it means 1 seconds
PART = 2
MS = 1.0/16000
NPERSEG = int(0.025/MS)
NOVERLAP = int(0.015/MS)
NFFT =1024
SIZE_FFT = int(NFFT/4)
SIZE_COLS = int(np.ceil((WINDOW - NPERSEG)/(NPERSEG - NOVERLAP)))

# Indicating the amount of data to be generated per epoch for train, valid and test
TOTAL_DATA_TRAIN = 800000
TOTAL_DATA_VALID = TOTAL_DATA_TRAIN*0.1
TOTAL_DATA_TEST = TOTAL_DATA_TRAIN*0.1

# VAD threshold used to actually consider the audio as someone talking
VAD = 0.05

# ==============================================================================
# Configuration parameters for the architecture
# ==============================================================================
# Number of labels in the last layer, in this case is two for the results [0,1] or [1,0]
L_LABEL = 2

# Size of the entrance input data
IN_HEIGHT = SIZE_FFT
IN_WIDTH = SIZE_COLS

# Number of channels in the input data
CHANNELS = 1

# Amount of pool layers to calculate the final width and height after the convolutions
POOL_LAYERS = 2

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

class cnn7:

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

  # Defining the max pool operation with a 2 x 2 kernel size and stride of 2
  def max_pool_2x2(self, x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  # Defining the initializer for the parameters, it is used the Xavier initializer 
  def weight_variable(self,shape):
    initializer = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initializer)
    
  # Defining the initializer for the bias, it is used the Xavier initializer 
  def bias_variable(self,shape):
    initializer = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initializer)


  def def_input(self):
    """ Defines inputs """
    with tf.name_scope('input'):

      # Defining the entrance of the model with the parameters defined at the begginig
      self.X1 = tf.placeholder(tf.float32, [None, IN_HEIGHT, IN_WIDTH, CHANNELS], name='X1')
      self.X2 = tf.placeholder(tf.float32, [None, IN_HEIGHT, IN_WIDTH, CHANNELS], name='X2')
      self.Y = tf.placeholder(tf.float32, [None, L_LABEL], name='Y')

  def def_variable(self):
    # Size of the batch, defined to be able to use it on the metrics section
    self.size_batch = FLAGS.batch_size

  # Defining the parameters aka weights for the model
  # In this case the parameters are name with W for weight and b for the bias, cn for convolutional plus a number
  # for the number of convolutional it is, and fc for fully connected plus a number
  def def_params(self):
    """ Defines model parameters """
    with tf.name_scope('params'):

      # First convolutional layer
      with tf.name_scope('conv1'):
        self.W_cn1 = self.weight_variable([3, 3, 1, 64])
        self.b_cn1 = self.bias_variable([64])

      # Second convolutional layer
      with tf.name_scope('conv2'):
        self.W_cn2 = self.weight_variable([3, 3, 64, 64])
        self.b_cn2 = self.bias_variable([64])

      # Third convolutional layer
      with tf.name_scope('conv3'):
        self.W_cn3 = self.weight_variable([3, 3, 64, 128])
        self.b_cn3 = self.bias_variable([128])

      # Fourth Convolutional layer
      with tf.name_scope('conv4'):
        self.W_cn4 = self.weight_variable([3, 3, 128, 128])
        self.b_cn4 = self.bias_variable([128])

      # First fully connected layer      
      with tf.name_scope('fc1'):
        self.W_fc1 = self.weight_variable([HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV * 256, 1024])
        self.b_fc1 = self.bias_variable([1024])

      # Second fully connected layer      
      with tf.name_scope('fc2'):
        self.W_fc2 = self.weight_variable([1024, 1024])
        self.b_fc2 = self.bias_variable([1024])

      # Third fully connected layer
      with tf.name_scope('fc3'):
        self.W_fc3 = self.weight_variable([1024, L_LABEL])
        self.b_fc3 = self.bias_variable([L_LABEL])        

  # Defining the architecture of the model
  # In this case it is an architecture inspired in a VGG 16, it has 4 convolutional layers and
  # 3 fully connected layers.
  def def_model(self):
    """ Defines the model """
    W_cn1 = self.W_cn1
    b_cn1 = self.b_cn1
    W_cn2 = self.W_cn2
    b_cn2 = self.b_cn2
    W_cn3 = self.W_cn3
    b_cn3 = self.b_cn3
    W_cn4 = self.W_cn4
    b_cn4 = self.b_cn4
    W_fc1 = self.W_fc1
    b_fc1 = self.b_fc1
    W_fc2 = self.W_fc2
    b_fc2 = self.b_fc2
    W_fc3 = self.W_fc3
    b_fc3 = self.b_fc3
  
    # First convolutional layers for the first signal
    with tf.name_scope('conv1a'):
      h_cn1a = tf.nn.relu(self.conv2d(self.X1, W_cn1) + b_cn1)

    # First convolutional layers for the second signal
    with tf.name_scope('conv1b'):
      h_cn1b = tf.nn.relu(self.conv2d(self.X2, W_cn1) + b_cn1)

    # Second convolutional layers for the first signal
    with tf.name_scope('conv2a'):
      h_cn2a = tf.nn.relu(self.conv2d(h_cn1a, W_cn2) + b_cn2)

    # Second convolutional layers for the second signal
    with tf.name_scope('conv2b'):
      h_cn2b = tf.nn.relu(self.conv2d(h_cn1b, W_cn2) + b_cn2)

    # First pooling layer for the first signal
    with tf.name_scope('pool1a'):
      #h_pool1a = self.max_pool_2x1(h_cn2a)
      h_pool1a = self.max_pool_2x2(h_cn2a)

    # First pooling layer for the second signal
    with tf.name_scope('pool1b'):
      h_pool1b = self.max_pool_2x2(h_cn2b)


    # Third convolutional layers for the first signal
    with tf.name_scope('conv3a'):
      h_cn3a = tf.nn.relu(self.conv2d(h_pool1a, W_cn3) + b_cn3)

    # Third convolutional layers for the second signal
    with tf.name_scope('conv3b'):
      h_cn3b = tf.nn.relu(self.conv2d(h_pool1b, W_cn3) + b_cn3)

    # Fourth convolutional layers for the first signal
    with tf.name_scope('conv4a'):
      h_cn4a = tf.nn.relu(self.conv2d(h_cn3a, W_cn4) + b_cn4)

    # Fourth convolutional layers for the second signal
    with tf.name_scope('conv4b'):
      h_cn4b = tf.nn.relu(self.conv2d(h_cn3b, W_cn4) + b_cn4)

    # Second pooling layer for the first signal
    with tf.name_scope('pool2a'):
      h_pool2a = self.max_pool_2x2(h_cn4a)

    # Second pooling layer for the second signal
    with tf.name_scope('pool2b'):
      h_pool2b = self.max_pool_2x2(h_cn4b)

    # Concat layer to go from convolutional layer to fully connected
    with tf.name_scope('concat1'):
      h_concat1 = tf.concat([h_pool2a, h_pool2b], axis=3)

    # First fully connected layer
    with tf.name_scope('fc1'):
      h_concat1_flat = tf.reshape(h_concat1, [-1, HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV *256]) 
      h_mat =  tf.matmul(h_concat1_flat, W_fc1)
      h_fc1 = tf.nn.relu(h_mat + b_fc1)

    # Second fully connected layer
    with tf.name_scope('fc2'):
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # Third fully connected layer
    with tf.name_scope('fc3'):
      self.Y_logt = tf.matmul(h_fc2, W_fc3) + b_fc3
      self.Y_pred = tf.nn.softmax(self.Y_logt, name='Y_pred')

  # Defining the output, the output it is collapsed to a value of 0,1 or 1,0 using the argmax function
  # Initially the value was the result of a softmax which make the result of the fully connected layer
  # and transform the result into real values in the range of (0,1) that add up to 1
  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output'):
      self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
      self.label_true = tf.argmax(self.Y, 1, name='label_true')

  # Defining the loss used, in the this case is used a crossed entropy and as a regularizer it is used
  # the l2 norm. This is done to avoid overfitting of the model.
  # Officialy the regularizer should be calculated with the weights of the whole model, however
  # for this kind of model the weights of the last or second last layer is enough
  # Another approach is the use the rms of the predicted output vs the desired output
  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss'):

      #cross entropy
      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
      self.loss = tf.reduce_mean(self.cross_entropy)
      
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

  # This is to plot the results and to see how is the lost and the accuracy actually working
  # The accuracy that it is drawn, it is the accuracy per batch
  def add_summaries(self):
    """ Adds summaries for Tensorboard """
    # defines a namespace for the summaries
    with tf.name_scope('summaries'):
      # adds a plot for the loss
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('accuracy', self.acc_batch)
      # groups summaries
      self.summary = tf.summary.merge_all()

  def train(self):

    # Creating a folder where to save the parameter
    file_path = str(sys.argv[0]) +'_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs)
    
    if os.path.exists(file_path) == False:
      os.mkdir(file_path)

    # Creating a file to write the loss and acurracy
    output_file = open(file_path+'_results.txt', 'w')

    """ Trains the model """
    # setup minimize function, this is the optimizer of the function to be minimized according to the loss
    grad = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    
    # setup minimize function
    optimizer = grad.minimize(self.loss)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    # opens session
    with tf.Session() as sess:
      
      # writers for TensorBorad
      train_writer = tf.summary.FileWriter('graphs/' +str(sys.argv[0]) +'_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      valid_writer = tf.summary.FileWriter('graphs/' +str(sys.argv[0]) +'_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      test_writer = tf.summary.FileWriter('graphs/' +str(sys.argv[0]) +'_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      train_writer.add_graph(sess.graph)

      # initialize variables (params)
      sess.run(init_op)

      saver = tf.train.Saver()     

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

          data_per_row = int(np.ceil(float(total_data)/float(2*len(matrix))))
          rows = len(matrix)

          # Creating the vector that are going to be fill in the next steps
          X1 = []
          X2 = []
          Y = []

          i = 0
          total = 0
          j = 0

          # Doing the cycle while I still have data to do
          while i < total_data:

            chosen_audio_1 = matrix[j]
            audio_1,samplerate = sf.read(chosen_audio_1)
            fixed_class = str(get_int(get_part(PART,chosen_audio_1))).zfill(4)

            if audio_1.shape[0]>WINDOW and np.sqrt(np.mean(np.abs(audio_1))) > VAD:        

              # number of times each audios is going to be used
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

                  permutation = np.random.permutation(X1_array.shape[0])
                  X1_array = X1_array[permutation,:]
                  X2_array = X2_array[permutation,:]
                  Y_array = Y_array[permutation]


                  # Running the apropiate model
                  # Train
                  if step_file == 0:

                    # evaluation with train data
                    feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array}
                    fetches = [optimizer, self.loss, self.accuracy, self.summary]
                    _,train_loss, train_acc, train_summary = sess.run(fetches, feed_dict=feed_dict)
                    train_writer.add_summary(train_summary, step_train)

                    acc_train = acc_train + train_acc

                    # Printing the results every 100 batch
                    if step_train % 50 == 0:
                      msg = "I{:3d} loss_train: ({:6.8f}), acc_train(batch, global): ({:6.8f},{:6.8f})"
                      msg = msg.format(step_train, train_loss, train_acc/FLAGS.batch_size, acc_train/(FLAGS.batch_size*step_train))
                      print(msg)
                      output_file.write(msg + '\n')

                    step_train += 1

                  # Validation
                  elif step_file == 1:
                    
                    # evaluation with Validation data
                    feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array}
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
                    feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array}
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
            
            j+=1
            
        # Saving the weights just once (no in every epoch) 
        save_path = saver.save(sess, str(file_path+'/'+ str(n_epochs) +'weights.ckpt') )
        

def run():

  # defines our model
  model = cnn7()

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
      default=15,
      help='Number of epochs to run trainer.'
  )

  parser.add_argument(
      '--batch_size',
      type=int,
      default=50,
      help='Batch size.'
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
