# About

## Abstract

This repository presents a Convolutional Neural Network (CNN) model that can identify sounds from 50 different categories such as nature sounds, human non-speech sounds, urban sounds, etc. with an accuracy of **70%**. The model analyzes the processed spectrum of a wav file to predict its category. The wav file is processed and transformed into a spectrum of size (11,220), and the model assigns a label to each spectrum.

This repository includes 8 files. The `implementation.ipynb` file contains the model's implementation code, the `custom.ipynb` file is used to load the model and run it on custom wav files, the `wav2spectrogram.ipynb` file reads a wav file and converts it into a spectrogram, the `wav2spectrum.ipynb` file reads a wav file and processes it into a spectrum, the `requirements.txt` file lists the requirements needed to run the model, the `categories.csv` file contains the classification categories, and the `instructions.txt` file offers guidance for a better understanding of this repository.


## Detailed

The model is trained on [esc50]("https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50") dataset which contains 2002 wav files of 50 different categories including nature sound, animal sound, human non speech sound, urban sound, domestic sound, bird sound, etc. The wav file is converted into spectrum of shape (11,220) and the model is trained on this spectrum. Below is the step by step explaination of respository.


### The model : 

The proposed Convolutional Neural Netowrk (CNN) based model has 6 convolutional layers, 3 pooling layers, 2 fully connected layers, 1 dropout layer with selu activation. The model taks spectrum of wav file in the form of numpy array of size (11,220) as an input and gives probability for each class in vector of size (50,1). Below is the screenshot of model's summary.

![CNN](https://github.com/Akash8292/Environment_Sound_Detection/assets/97883391/9171d537-60c7-4579-9899-0f315baaed90 | width=100)



### Wav to Spectrum : 

Before delving into creating spectrum, let's first grasp the structure of a wav file and how Python interprets it. Similar to how an image comprises pixels, a wav file consists of samples. The sample rate signifies the quantity of samples employed per second to capture analog audio and encode it into digital audio. In this scenario, the sample rate equals 44100 Hz, and every 5-second wav file encompasses 220500 samples. The transition from wav to spectrum progresses as follows: Wav -> Waveform -> Spectrogram -> Spectrum.


Now, let's explore wav to waveform. The waveform depicts the amplitude distribution over time of a wave file. When we read a wavefile, it is in the form of a numpy array of size (220500,1) where the initial 440100 samples correspond to the first second, the subsequent 440100 samples represent the second second, and so forth. By dividing the array by the sample rate, we obtain an array of shape (44100,5). Each value at a specific index signifies the amplitude. Plotting the array generates the waveform. Attached is a screenshot displaying the waveform.


![waveform](https://github.com/Akash8292/Environment_Sound_Detection/assets/97883391/56961499-7b4e-4880-84bc-c06f41d5cd50)

The spectrogram is the colour map of distribution of frequency (in Hz) vs time (in s) where the color represents the intensity (in dB). By using `librosa` library's `spectrogram` method we the frequency, time and intensity. Here is the plot of spectrogram.

![spectrogram](https://github.com/Akash8292/Environment_Sound_Detection/assets/97883391/03000415-9438-44f2-a765-11d739a739d4)


Now, why create a spectrum when we can train the model on the spectrogram? The spectrogram, shaped (44100,5), is quite large. Training the model directly on the spectrogram would lead to an excess of parameters, resulting in unnecessary complexity. Moreover, the spectrogram contains noise that needs filtering. Therefore, the spectrogram is transformed into a spectrum with a shape of (11,220).

To achieve this conversion, the first 220000 samples of the WAV file are considered, forming 220 spectrograms of 1000 samples each. These 1000-sample spectrograms are divided into 11 channels. Each channel processes the first and last k samples within the spectrogram to generate a floating-point number. Consequently, one spectrogram yields 11 distinct channels, hence breaking down the 1000-sample spectrogram into a size (11,1) vector. With 220 such spectrograms, this results in an array of size (11,220). This array represents one WAV file and serves as the model input. Below is a snapshot displaying the spectrum.

![spectrum](https://github.com/Akash8292/Environment_Sound_Detection/assets/97883391/4476cae7-2735-425f-a556-ea990c880e44)



# Prerequisites

`Python>=3.6`

# Getting started

1. Clone the repository or download the zip file.
2. Install necessary packages using `pip install -r requirements.txt`.
3. Run the `custom.ipynb` file to use the pretrained model on custom wav file, change path of model and wav file accordingly.
4. Read the `instructions.txt` for better understanding of repository.

# Future work

In the future, I look forward to using a Recurrent Neural Network model to determine which model provides me with greater accuracy.
