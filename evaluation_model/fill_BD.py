# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
import random
import sys
from scipy import signal
from sklearn.preprocessing import StandardScaler

# Function that splits and string to get the desire part
def get_part(part,string):
  aux = string.split('/')
  a = aux[len(aux)-part-1]
  return a

# function that gets the numeric part of a string
def get_int(string):
  return int(''.join([d for d in string if d.isdigit()]))

name_file = 'speakers.txt'
input_file  = open(name_file,'r')


VAD = 0.08
WINDOW= 16000 
MS = 1.0/16000
NPERSEG = int(0.025/MS)
NOVERLAP = int(0.015/MS)
NFFT = NPERSEG # Para 32
SIZE_FFT = 32 # Para 32

#NFFT =1024 # Para 256
#SIZE_FFT = int(NFFT/4) # For 256

SIZE_COLS = int(np.ceil((float(WINDOW) - NPERSEG)/(NPERSEG - NOVERLAP)))
print SIZE_COLS

ver_audio = []
for line in input_file:
  row = line.rstrip()
  name = str(get_part(0,row))
  class_index = name.rfind('.')
  name = name[0:class_index]
  class_index = name.find('-')
  name = name[0:class_index]
  ver_audio.append([row, name])

ver_audio = np.array(ver_audio)

n_class = ''
num_audio = 1

for i in xrange(0,ver_audio.shape[0]):

  chosen_audio = ver_audio[i][0]
  # print chosen_audio
  name = ver_audio[i][1]

  audio,samplerate = sf.read(chosen_audio)

  if n_class != name:
    num_audio = 1
    n_class = name

  if audio.shape[0] <= WINDOW:
    audio_aux = np.zeros((WINDOW + 1))
    audio_aux[0:audio.shape[0]] = audio
    audio = audio_aux

  if np.sqrt(np.mean(np.abs(audio))) < VAD:
    audio = audio * 4


  if audio.shape[0]>WINDOW and np.sqrt(np.mean(np.abs(audio))) > VAD:

    while True:

      # Getting a random index for each audio
      index = random.randrange(0,audio.shape[0]-WINDOW,1)

      # Analyzing the VAD o the signal using the rms
      if np.sqrt(np.mean(np.abs(audio[index:index+WINDOW+1]))) > VAD:

        f, t, Sxx = signal.spectrogram(audio[index:index+WINDOW], samplerate,  window=('hamming'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1)
        Hxx = StandardScaler().fit_transform(Sxx)
        data_audio = np.reshape(Hxx[0:SIZE_FFT,:],(SIZE_FFT,SIZE_COLS,1))

        np.save('data_bases/database_32_libriaira/'+ name+'_'+str(num_audio), data_audio)
        num_audio = num_audio + 1

        break;


  else:
    print i
    print audio.shape[0]>WINDOW
    print np.sqrt(np.mean(np.abs(audio))) > VAD