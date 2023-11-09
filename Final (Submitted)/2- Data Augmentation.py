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


def random_gain(signal, min_factor, max_factor):
    gain_rate = random.uniform(min_factor, max_factor)
    augmented_signal = signal * gain_rate
    return augmented_signal


def invert_polarity(signal):
    return signal * -1


if __name__ == "__main__":
    
    audio_dataset_path='C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold'
    
    
    metadata=pd.read_csv('C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/metadata_every_surah_for_dataaugmentation/metadata_114.csv')
    
    def augmentation(file):
        signal, sr = librosa.load(file_name, sr= 16000)
        augmented_signal1 = add_white_noise(signal, 0.1)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal1, sr)
        
        augmented_signal2 = add_white_noise(signal, 0.2)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal2, sr)
        
        augmented_signal3 = add_white_noise(signal, 0.3)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal3, sr)
        
        augmented_signal4 = add_white_noise(signal, 0.4)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal4, sr)
        
        augmented_signal5 = add_white_noise(signal, 0.5)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal5, sr)
        
        augmented_signal6 = add_white_noise(signal, 0.6)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal6, sr)
        
        augmented_signal7 = add_white_noise(signal, 0.7)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal7, sr)
        
        augmented_signal8 = add_white_noise(signal, 0.8)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal8, sr)
        
        augmented_signal9 = time_stretch(signal, 0.5)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal9, sr)
        
        augmented_signal10 = time_stretch(signal, 0.4)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal10, sr)
        
        augmented_signal11 = time_stretch(signal, 0.3)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal11, sr)
        # augmented_signal3 = time_stretch(signal, 0.8)
        # sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal3, sr)
        
        augmented_signal12 = time_stretch(signal, 0.2)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal12, sr)
        
        augmented_signal13 = time_stretch(signal, 0.1)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal13, sr)
        
        augmented_signal14 = pitch_scale(signal, sr, 1)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal14, sr)

        augmented_signal15 = pitch_scale(signal, sr, 2)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal15, sr)
        
        augmented_signal16 = pitch_scale(signal, sr, 3)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal16, sr)
        
        augmented_signal17 = pitch_scale(signal, sr, 4)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal17, sr)

        augmented_signal18 = pitch_scale(signal, sr, 5)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal18, sr)
        
        augmented_signal19 = pitch_scale(signal, sr, 6)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal19, sr)
        
        augmented_signal20 = pitch_scale(signal, sr, 7)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal20, sr)
        
        augmented_signal21 = pitch_scale(signal, sr, 8)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal21, sr)
        
        augmented_signal22 = random_gain(signal, 1, 2)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal22, sr)
        
        augmented_signal23 = random_gain(signal, 2, 3)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal23, sr)
        
        augmented_signal24 = random_gain(signal, 2, 4)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal24, sr)
        
        augmented_signal25 = random_gain(signal, 3, 4)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal25, sr)
        
        augmented_signal26 = random_gain(signal, 1, 3)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal26, sr)
        
        augmented_signal27 = invert_polarity(signal)
        sf.write("C:/Users/aleem/Desktop/New_extracted audio wav after cleaning then augmentation/extracted_audio_wav_fold/augmented_files/augmented_audio"+str(uuid.uuid4())+".wav", augmented_signal27, sr)
        

    
    for index_num,row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        augmentation(file_name)