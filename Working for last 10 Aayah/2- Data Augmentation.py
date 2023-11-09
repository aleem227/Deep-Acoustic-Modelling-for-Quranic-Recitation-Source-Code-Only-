import random
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import uuid
from tqdm import tqdm
import os


# Python 3.8
# install matplotlib, librosa
# install python3-tk -> sudo apt install python3-tk


def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal


def time_stretch(signal, time_stretch_rate):
    """Time stretching implemented with librosa:
    https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
    """
    return librosa.effects.time_stretch(signal, time_stretch_rate)


def pitch_scale(signal, sr, num_semitones):
    """Pitch scaling implemented with librosa:
    https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
    """
    return librosa.effects.pitch_shift(signal, sr, num_semitones)


def random_gain(signal, min_factor=0.1, max_factor=0.12):
    gain_rate = random.uniform(min_factor, max_factor)
    augmented_signal = signal * gain_rate
    return augmented_signal


def invert_polarity(signal):
    return signal * -1


if __name__ == "__main__":
    
    audio_dataset_path='C:/Users/aleem/Desktop/Working last 10 aayah/extracted_audio_wav_fold'
    
    
    metadata=pd.read_csv('C:/Users/aleem/Desktop/Working last 10 aayah/metadata_111.csv')
    
    def augmentation(file):
        signal, sr = librosa.load(file_name)
        augmented_signal1 = add_white_noise(signal, 0.1)
        sf.write("C:/Users/aleem/Desktop/Working last 10 aayah/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal1, sr)
        
        augmented_signal2 = time_stretch(signal, 0.5)
        sf.write("C:/Users/aleem/Desktop/Working last 10 aayah/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal2, sr)
        
        augmented_signal3 = time_stretch(signal, 0.8)
        sf.write("C:/Users/aleem/Desktop/Working last 10 aayah/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal3, sr)
        
        augmented_signal4 = pitch_scale(signal, sr, 2)
        sf.write("C:/Users/aleem/Desktop/Working last 10 aayah/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal4, sr)

        augmented_signal5 = pitch_scale(signal, sr, 4)
        sf.write("C:/Users/aleem/Desktop/Working last 10 aayah/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal5, sr)
        
        augmented_signal6 = random_gain(signal, 2, 4)
        sf.write("C:/Users/aleem/Desktop/Working last 10 aayah/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal6, sr)
        
        augmented_signal7 = invert_polarity(signal)
        sf.write("C:/Users/aleem/Desktop/Working last 10 aayah/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal7, sr)
        

    
    for index_num,row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        augmentation(file_name)