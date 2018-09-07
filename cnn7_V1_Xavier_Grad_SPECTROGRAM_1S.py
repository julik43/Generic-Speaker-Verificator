# """ Simple convolutional neural network.
# UNAM IIMAS
# AUthor: Ivette Velez
# Tutor:  Caleb Rascon
# Co-tutor: Gibran Fuentes
# To run the model: python cnn6_V1_VGG16.py --learning_rate 0.002 --num_epochs 20 --train_dir dataset --batch_size 100
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
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
import stempeg as smp
from scipy import ndimage
from scipy import signal
import sklearn.preprocessing as pps
from sklearn.preprocessing import StandardScaler

#from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib import layers

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Using just one GPU in case of GPU 
# os.environ['CUDA_VISIBLE_DEVICES']= '1'

# Basic model parameters as external flags.
FLAGS = None

FILE= 'audios.txt'
FILE_TRAIN = 'train_speakers.txt'
FILE_VALID = 'valid_speakers.txt'
FILE_TEST = 'test_speakers.txt'

# Constants on the model
WINDOW = 1*16000 # with audios of librispeech it means 1 seconds
PART = 2
MS = 1.0/16000
NPERSEG = int(0.025/MS)
NOVERLAP = int(0.015/MS)
NFFT = NPERSEG
SIZE_FFT = 32
SIZE_COLS = int(np.ceil((WINDOW - NPERSEG)/(NPERSEG - NOVERLAP)))

print(SIZE_COLS)
print(SIZE_FFT)

# TOTAL_DATA_TRAIN = 1092009*2*2
TOTAL_DATA_TRAIN = 656427*2*2
# TOTAL_DATA_VALID = TOTAL_DATA_TRAIN*0.1
TOTAL_DATA_VALID = 36237*2*2
# TOTAL_DATA_TEST = TOTAL_DATA_TRAIN*0.1
TOTAL_DATA_TEST = 36237*2*2
VAD = 0.08

L_LABEL = 2
IN_HEIGHT = SIZE_FFT
IN_WIDTH = SIZE_COLS
CHANNELS = 1
POOL_LAYERS = 2

WIDTH_AFTER_CONV = int(np.ceil(float(IN_WIDTH)/float(2**POOL_LAYERS)))
HEIGHT_AFTER_CONV = int(np.ceil(float(IN_HEIGHT)/float(2**POOL_LAYERS)))

print(WIDTH_AFTER_CONV)
print(HEIGHT_AFTER_CONV)

# audio_possibilities = ['.flac', '_car-winupb2_SNR-5.flac', '_car-winupb2_SNR-10.flac', '_car-winupb2_SNR-15.flac', '_street-city2_SNR-5.flac', '_street-city2_SNR-10.flac', '_street-city2_SNR-15.flac', '_street-kg2_SNR-5.flac', '_street-kg2_SNR-10.flac', '_street-kg2_SNR-15.flac']
audio_possibilities = ['.m4a']

# Function that splits and string to get the desire part
def get_part(part,string):
  aux = string.split('/')
  a = aux[len(aux)-part-1]
  return a

# function that gets the numeric part of a string
def get_int(string):
  return int(''.join([d for d in string if d.isdigit()]))

def get_class(direction):

  # Verifying the extension of the audio file
  extension = direction[(len(direction)-3) : len(direction)]  

  if extension == 'wav':
    fixed_class = get_part(1,direction)
  else:
    fixed_class = get_part(2,direction)

  # print(str(direction + ' extension:' + str(extension) + ' clase:' + str(fixed_class)))
  return fixed_class

def random_audio(direction):
  # For the Librispeech corpus there is a noisy version with 3 types of noises in 3 SNR,
  # so the idea is to choose one of the 10 posible options
  audio_index = direction.rfind('.flac')
  audio_route = direction[0:audio_index]

  possibility = random.randrange(0,len(audio_possibilities),1)

  return str(audio_route+audio_possibilities[possibility])

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

  def conv2d(self, x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self, x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

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

  def def_variable(self):
    self.size_batch = FLAGS.batch_size

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
        #self.W_fc1 = self.weight_variable([HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV * 1024, 4096])
        #self.W_fc1 = self.weight_variable([HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV * 512, 1024])
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
      #h_concat1 = tf.concat([h_pool5a, h_pool5b], axis=3)
      #h_concat1 = tf.concat([h_pool4a, h_pool4b], axis=3)
      #h_concat1 = tf.concat([h_pool3a, h_pool3b], axis=3)
      h_concat1 = tf.concat([h_pool2a, h_pool2b], axis=3)

    # First fully connected layer
    with tf.name_scope('fc1'):
      #h_concat1_flat = tf.reshape(h_concat1, [-1, HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV * 1024])  
      #h_concat1_flat = tf.reshape(h_concat1, [-1, HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV * 512])  
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

  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output'):
      self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
      self.label_true = tf.argmax(self.Y, 1, name='label_true')

  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss'):

      # self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=self.Y, logits=self.Y_logt)
      # self.loss = tf.losses.get_total_loss()

      #cross entropy
      #self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)*0.00001
      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
      self.loss = tf.reduce_mean(self.cross_entropy)
      

  def def_metrics(self):
    """ Adds metrics """
    with tf.name_scope('metrics'):
      cmp_labels = tf.equal(self.label_true, self.label_pred)
      self.accuracy = tf.reduce_sum(tf.cast(cmp_labels, tf.float32), name='accuracy')
      self.acc_batch = (self.accuracy/self.size_batch)*100

  def add_summaries(self):
    """ Adds summaries for Tensorboard """
    # defines a namespace for the summaries
    with tf.name_scope('summaries'):
      # adds a plot for the loss
      tf.summary.scalar('loss', self.loss)
      #tf.summary.scalar('accuracy', self.accuracy)
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
    # creates optimizer
    grad = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    #grad = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    
    # setup minimize function
    optimizer = grad.minimize(self.loss)

    #optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    # opens session
    with tf.Session() as sess:
      
      # writers for TensorBorad
      train_writer = tf.summary.FileWriter('graphs/cnn7_v1_train_XS_XA_Grad_1S_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      valid_writer = tf.summary.FileWriter('graphs/cnn7_v1_valid_XS_XA_Grad_1S_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      test_writer = tf.summary.FileWriter('graphs/cnn7_v1_test_XS_XA_Grad_1S_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      train_writer.add_graph(sess.graph)

      # initialize variables (params)
      # sess.run(tf.global_variables_initializer())
      sess.run(init_op)

      saver = tf.train.Saver()     
      # saver.restore(sess, "/home/ar/ivette_thesis/models_aira/V7_32_dimex_noisy_plus_librispeech/cnn7_V1_Xavier_Grad_SPECTROGRAM_1S.py_0.01_15/14weights.ckpt") 

      # Initializing the step for train and validation
      step_train = 1
      step_valid = 1
      step_test = 1
      acc_train = 0
      acc_valid = 0
      acc_test = 0
      
      # Compute for the desired number of epochs.
      for n_epochs in range(FLAGS.num_epochs):

        if n_epochs == FLAGS.num_epochs - 1:
          k_limit = 3
        else:
          k_limit = 2


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
            # Getting the name of the class
            matrix.append(row)

          # print(len(matrix))

          data_per_row = int(np.ceil(float(total_data)/float(2*len(matrix))))
          rows = len(matrix)

          print('Data per row: '+str(data_per_row))
          print('Rows TF: '+str(rows_tf))
          print('Total data: '+str(total_data))

          X1 = []
          X2 = []
          Y = []

          i = 0
          total = 0
          j = 0

          # for i in xrange(0,len(matrix)):
          #   chosen_audio_1 = matrix[i]
          #   audio_1,samplerate = sf.read(chosen_audio_1)
          #   fixed_class = str(get_class(chosen_audio_1))

          #   # if audio_1.shape[0] <= WINDOW:
          #   if audio_1.shape[0]>WINDOW and np.sqrt(np.mean(np.abs(audio_1))) <= VAD:        
          #     print(str(chosen_audio_1) + ' shape:' + str(audio_1.shape[0]))

          # exit(0)

          # print('total data ' + str(total_data))
          # print(len(matrix))
          # # exit(0)

          while (i < total_data and j < len(matrix)):

            chosen_audio_1 = matrix[j]

            audio_1,samplerate = sf.read(chosen_audio_1)
            # audio_1,samplerate = smp.read_stems(chosen_audio_1)
            fixed_class = str(get_class(chosen_audio_1))

            # print(chosen_audio_1)
            # print('Clase: ' + str(fixed_class))

            # fixed_class = str(get_int(get_part(PART,chosen_audio_1))).zfill(4)
            if audio_1.shape[0] <= WINDOW:
              # print(chosen_audio_1)
              # print(audio_1.shape[0])
              audio_1_aux = np.zeros((WINDOW + 1))
              audio_1_aux[0:audio_1.shape[0]] = audio_1
              audio_1 = audio_1_aux

              # print(audio_1.shape[0])
              # print('Menor que ventana')
              # sys.exit(0)


            if audio_1.shape[0]>WINDOW and np.sqrt(np.mean(np.abs(audio_1))) > VAD:        

              # number of times each audios is going to be used
              for k in range(0,data_per_row):

                flag_model = False                

                # All the audios will be mixed with one of its own and one different
                if j+1 <rows and str(get_class(matrix[j+1])) == fixed_class:
                  chosen_audio_2 = matrix[j+1]
                elif j-1 >=0 and str(get_class(matrix[j-1])) == fixed_class:
                  chosen_audio_2 = matrix[j-1]
                else:
                  chosen_audio_2 = matrix[j]

                audio_2,samplerate = sf.read(chosen_audio_2)
                # audio_2,samplerate = smp.read_stems(chosen_audio_2)

                if audio_2.shape[0] <= WINDOW:
                  print(audio_2.shape[0])
                  audio_2_aux = np.zeros((WINDOW + 1))
                  audio_2_aux[0:audio_2.shape[0]] = audio_2
                  audio_2 = audio_2_aux


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

                        #print(fixed_class, str(get_int(get_part(PART,chosen_audio_2))).zfill(4)) 

                        # Filling the matrixes with the data
                        X1.append(data_audio_1)
                        X2.append(data_audio_2)
                        Y.append([0,1])
                        total+=1
                        i+=1
                        break;

                  if total>= rows_tf:
                    flag_model = True

                # print(total)

                # Comparing with data of a different speaker
                #while True:
                while flag_model == False:
                  chosen_audio_2 = matrix[random.randrange(0,rows,1)]

                  if str(get_class(chosen_audio_2))!=fixed_class:
                    
                    audio_2,samplerate = sf.read(chosen_audio_2)
                    # audio_2,samplerate = smp.read_stems(chosen_audio_2)

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

                # print(total)

                if total>= rows_tf:
                    flag_model = True

                # print(str(flag_model))

                # If the file must be written
                if flag_model == True:

                  #print('Entro al train')

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
                    fetches = [optimizer, self.loss, self.accuracy, self.summary ]
                    _,train_loss, train_acc, train_summary = sess.run(fetches, feed_dict=feed_dict)
                    train_writer.add_summary(train_summary, step_train)

                    acc_train = acc_train + train_acc

                    # Printing the results every 100 batch
                    if step_train % 50 == 0:
                    # if True:
                      msg = "I{:3d} loss_train: ({:6.8f}), acc_train(batch, global): ({:6.8f},{:6.8f})"
                      msg = msg.format(step_train, train_loss, train_acc/FLAGS.batch_size, acc_train/(FLAGS.batch_size*step_train))
                      print(msg)
                      output_file.write(msg + '\n')

                    step_train += 1

                  # Validation
                  elif step_file == 1:
                    
                    # evaluation with train data
                    feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array}
                    fetches = [self.loss, self.accuracy, self.summary]
                    valid_loss, valid_acc, valid_summary = sess.run(fetches, feed_dict=feed_dict)
                    valid_writer.add_summary(valid_summary, step_train)

                    acc_valid = acc_valid + valid_acc

                    if step_valid % 50 == 0:
                      msg = "I{:3d} loss_val: ({:6.8f}), acc_val(batch, global): ({:6.8f},{:6.8f})"
                      msg = msg.format(step_valid, valid_loss, valid_acc/FLAGS.batch_size, acc_valid/(FLAGS.batch_size*step_valid))
                      print(msg)
                      output_file.write(msg + '\n')

                    step_valid += 1

                  # Test
                  else:

                    # evaluation with train data
                    feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array}
                    fetches = [self.loss, self.accuracy, self.summary]
                    test_loss, test_acc, test_summary = sess.run(fetches, feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step_train)

                    acc_test = acc_test + test_acc

                    if step_test % 50 == 0:
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
      default=60,
      help='Batch size.'
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



# if __name__ == '__main__':
#   import sys
#   sys.exit(main(sys.argv))
