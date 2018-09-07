# -*- coding: utf-8 -*-

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
import glob
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
from scipy import ndimage
from scipy import signal
import sklearn.preprocessing as pps
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

#from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib import layers

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Using just one GPU in case of GPU 
os.environ['CUDA_VISIBLE_DEVICES']= '1'

# Basic model parameters as external flags.
FLAGS = None

# Lenght of the audio to be recorded
lengt_audio = 1
sample_rate = 16000

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
    path_heatmap = str(sys.argv[0]) + '_' 
    Y_heatmap = []

    if os.path.exists('resultados_equilibrio') == False:
      os.mkdir('resultados_equilibrio')
    
    """ Trains the model """
    # creates optimizer
    grad = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)    
    optimizer = grad.minimize(self.loss)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    # opens session
    with tf.Session() as sess:
      
      # writers for TensorBorad
      train_writer = tf.summary.FileWriter('graphs/verificator_train')
      valid_writer = tf.summary.FileWriter('graphs/verificator_valid')
      test_writer = tf.summary.FileWriter('graphs/verificator_test')
      train_writer.add_graph(sess.graph)

      # initialize variables (params)
      sess.run(init_op)

      saver = tf.train.Saver()
      saver.restore(sess, "/home/ar/ivette_thesis/models_aira/V7_32_voxceleb_epoch_11/10weights.ckpt") 

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

              # Number speaker since 1
              if int(number_audio) <= num_audios and (fixed_class in chosen_classes) == True:

              # number speaker since 0
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

          # print('Len database: ' + str(len(database)))
          # print('Ver audio shape ' + str(ver_audio.shape))
          # exit(0)

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
                      #Yb.append(np.zeros(L_CLASS))
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
                    #Ya_array = np.array(Ya[amount_audio:len(database)])
                    #Yb_array = np.array(Yb[amount_audio:len(database)])
                  
                  else:
                    limit_array = amount_audio+FLAGS.batch_size
                    X1_array = np.array(X1[amount_audio:limit_array])
                    X2_array = np.array(X2[amount_audio:limit_array])
                    Y_array = np.array(Y[amount_audio:limit_array])
                    #Ya_array = np.array(Ya[amount_audio:limit_array])
                    #Yb_array = np.array(Yb[amount_audio:limit_array])

                  #init_time = time.time()

                  #feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array, self.Ya : Ya_array, self.Yb : Yb_array, self.phase:0}
                  feed_dict = {self.X1: X1_array, self.X2: X2_array, self.Y : Y_array}
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
      '--batch_size',
      type=int,
      default=50,
      help='Batch size.'
  )
  parser.add_argument(
      '--database',
      type=str,
      default='/data_bases/database_32_libriaira',
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
