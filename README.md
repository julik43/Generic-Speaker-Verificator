# Generic-Speaker-Verificator

This repository has been updated to: https://github.com/julik43/Online-Identification-of-New-Speakers

To use this model you need:

1. Download a database	

	For this work the database used was LibriSpeech available in: http://www.openslr.org/12/

2. To run any of the models 

	You need four files:
	
	* train_speakers.txt --> a list of audio files for training, with the proper path
	
	* valid_speakers.txt --> a list of audio files for validation, with the proper path

	* test_speakers.txt --> a list of audio files for test, with the proper path


# To replicate the results found

Update the correct path to the files you are going to use, LibriSpeech or VoxCeleb.

For this project, VoxCeleb database was changed fromo m4a format to flac format using the bash code "change_m4a_to_flac.sh".

Note: it is important to mention that this code localize the audios of the third level of folders from the path given.

# ResNet trained with 32 frequencies and Librispeech

1. Remember to update the path of the audios you are working with in the files of train, validation and test.

2. Be sure that the parameters of the resnet_initialization.py are configured as follows:

FILE_TRAIN = 'librispeech_train_speakers_initialization'

FILE_VALID = 'librispeech_valid_speakers.txt'

FILE_TEST = 'librispeech_test_speakers.txt'

FILE_LIST = 'librispeech_list_train_speakers'

permited_speakers = 100

WINDOW = 1*16000

MS = 1.0/16000

NPERSEG = int(0.025/MS)

NOVERLAP = int(0.015/MS)

NFFT =NPERSEG

SIZE_FFT = 32

TOTAL_DATA_TRAIN = 12277*2

TOTAL_DATA_VALID = TOTAL_DATA_TRAIN*0.1

TOTAL_DATA_TEST = TOTAL_DATA_TRAIN*0.1

VAD = 0.05


3. Be sure that the parameters of the resnet.py are configured as follows:

FILE_TRAIN = 'librispeech_train_speakers'

FILE_VALID = 'librispeech_valid_speakers.txt'

FILE_TEST = 'librispeech_test_speakers.txt'

FILE_LIST = 'librispeech_list_train_speakers'

permited_speakers = 100

WINDOW = 1*16000

MS = 1.0/16000

NPERSEG = int(0.025/MS)

NOVERLAP = int(0.015/MS)

NFFT =NPERSEG

SIZE_FFT = 32

TOTAL_DATA_TRAIN = 800000

TOTAL_DATA_VALID = TOTAL_DATA_TRAIN*0.1

TOTAL_DATA_TEST = TOTAL_DATA_TRAIN*0.1

VAD = 0.05

Configure the path of the saver with the correct path to the resnet initialization final weights.

saver.restore(sess, "resnet_initialization.py_0.01_10/final_weights.ckpt")

4. Run the models.

4.1. Run the resnet initialization model --> python resnet_initialization.py

4.2. Run the resnet model --> python resnet.py --learning_rate 0.01
	
	Note: Remember to configure the path of the saver

4.3 Run the resnet model with a smaller learning rate --> python resnet.py --learning_rate 0.001

	Note: Remember to configure the path of the saver with the weights of the model of 4.2. 

Results:

train: 0.95858375

validation: 0.93198125

test: 0.92392500




# Module VGG trained with 32 frequencies and LibriSpeech

1. Remember to update the path of the audios you are working with in the files of train, validation and test.

2. Be sure that the parameters of the module_vgg.py are configured as follows:

FILE_TRAIN = 'librispeech_train_speakers'

FILE_VALID = 'librispeech_valid_speakers.txt'

FILE_TEST = 'librispeech_test_speakers.txt'

WINDOW = 1*16000

MS = 1.0/16000

NPERSEG = int(0.025/MS)

NOVERLAP = int(0.015/MS)

NFFT =NPERSEG

SIZE_FFT = 32

TOTAL_DATA_TRAIN = 800000

TOTAL_DATA_VALID = TOTAL_DATA_TRAIN*0.1

TOTAL_DATA_TEST = TOTAL_DATA_TRAIN*0.1

VAD = 0.05


3. Run the model.

3.1. Run the resnet model --> python module_vgg.py 


Results:

train: 0.92258267

validation: 0.89274333

test: 0.92220000




# Module VGG trained with 256 frequencies and LibriSpeech

1. Remember to update the path of the audios you are working with in the files of train, validation and test.

2. Be sure that the parameters of the module_vgg.py are configured as follows:

FILE_TRAIN = 'librispeech_train_speakers'

FILE_VALID = 'librispeech_valid_speakers.txt'

FILE_TEST = 'librispeech_test_speakers.txt'

WINDOW = 1*16000

MS = 1.0/16000

NPERSEG = int(0.025/MS)

NOVERLAP = int(0.015/MS)

NFFT =NPERSEG

SIZE_FFT = 256

TOTAL_DATA_TRAIN = 800000

TOTAL_DATA_VALID = TOTAL_DATA_TRAIN*0.1

TOTAL_DATA_TEST = TOTAL_DATA_TRAIN*0.1

VAD = 0.05


3. Run the model.

3.1. Run the resnet model --> python module_vgg.py 


Results:

train: 0.93572700

validation: 0.89579833

test: 0.91178750




# Module VGG trained with 32 frequencies and VoxCeleb

1. Remember to update the path of the audios you are working with in the files of train, validation and test.

2. Be sure that the parameters of the module_vgg.py are configured as follows:

FILE_TRAIN = 'voxceleb_train_speakers'

FILE_VALID = 'voxceleb_valid_speakers.txt'

FILE_TEST = 'voxceleb_test_speakers.txt'

WINDOW = 1*16000

MS = 1.0/16000

NPERSEG = int(0.025/MS)

NOVERLAP = int(0.015/MS)

NFFT =NPERSEG

SIZE_FFT = 32

TOTAL_DATA_TRAIN = 656427*2*2

TOTAL_DATA_VALID = 36237*2*2

TOTAL_DATA_TEST = 36237*2*2

VAD = 0.08


3. Run the model.

3.1. Run the resnet model --> python module_vgg.py --num_epochs 8


Results:

train: 0.87228221

validation: 0.84642973









