import numpy as np
import librosa
from keras.models import load_model
from keras.models import Model
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Hide the main tkinter window

file_name = filedialog.askopenfilename(
    title="Select a WAV file",
    filetypes=[("WAV files", "*.wav")]
)

if file_name:
    print("Selected WAV file path:", file_name)
else:
    print("No file selected.")

# file_name = 'C:/Users/aleem/Desktop/Currentlyworking_New_extracted audio wav after cleaning then augmentation/sliding_window_1_sec/data/falaq1_testing.wav'


def mfcc_features_extractor_1sec(file):
    # audio, sample_rate = librosa.load(file_name) 
    # mfccs_features = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    # mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    y, sr = librosa.load(file, sr=16000)
    
    y = y[:192000]  #12 Secs for all audios
    zero_padding = np.zeros(192000 - y.shape[0], dtype=np.float32)
    y = np.concatenate([y, zero_padding])   

    # Define window size and hop length
    win_length = int(sr * 1) # 1 second window
    hop_length = int(win_length / 2) # 50% overlap

    # Extract MFCC features for each window
    mfccs = []
    for i in range(0, len(y)-win_length, hop_length):
        mfcc = librosa.feature.mfcc(y[i:i+win_length], sr=sr, n_mfcc=40)
        mfccs.append(mfcc)

    # Combine MFCC features for all windows
    mfccs = np.concatenate(mfccs, axis=1)
    mfccs = mfccs.T

    # Return final MFCC features
    return mfccs

    
    # return mfccs_scaled_features
    


mfcc_extracted_features_single=[]

mfcc_extracted_features_single= mfcc_features_extractor_1sec(file_name)

mfcc_extracted_features_single = np.array(mfcc_extracted_features_single)

mfcc_extracted_features_single = mfcc_extracted_features_single.reshape((1, 704, 40, 1))



# Load the saved model
loaded_model = load_model('Model_CNN2D_startwithnumpylengthsamemfcc_1sec_4032files_90_10_size_result 100_96.29.h5')

# Get the name of the intermediate layer from the loaded model summary
intermediate_layer_name = 'dense'

# Find the index of the intermediate layer in the model's layers list
intermediate_layer_index = [i for i, layer in enumerate(loaded_model.layers) if layer.name == intermediate_layer_name][0]

# Define a new model that includes layers up to the intermediate layer
intermediate_layer_model = Model(inputs=loaded_model.input,
                                  outputs=loaded_model.layers[intermediate_layer_index].output)

# Assuming you already have mfcc_extracted_features of shape (704, 40, 1)
# Replace this with your actual input data

# Predict the intermediate layer output for the input data
intermediate_output = intermediate_layer_model.predict(mfcc_extracted_features_single)

# Now, intermediate_output contains the intermediate layer output for mfcc_extracted_features


import numpy as np

# Load the 10 arrays and store them in a dictionary
arrays = {
    'AL_FALAQ': np.load('al_falaq_mean.npy'),
    'AL_FIL': np.load('al_fil_mean.npy'),
    'AL_IKHLAS': np.load('al_ikhlas_mean.npy'),
    'AL_KAFIRUN': np.load('al_kafirun_mean.npy'),
    'AL_KAUTHAR': np.load('al_kauthar_mean.npy'),
    'AL_MASAD': np.load('al_masad_mean.npy'),
    'AL_MAUN': np.load('al_maun_mean.npy'),
    'AL_QURAISH': np.load('al_quraish_mean.npy'),
    'AN_NAS': np.load('an_nas_mean.npy'),
    'AN_NASR': np.load('an_nasr_mean.npy')
}


# Assuming you already have the 'arrays' dictionary defined as shown in your question
# Also assuming 'intermediate_output' is defined somewhere

closest_distance = float('inf')  # Initialize closest_distance to positive infinity
closest_array = None
closest_array_name = None
closest_index_within_array = None

# Define the names of the arrays you want to compare
array_names_to_compare = ['AL_FALAQ', 'AL_FIL', 'AL_IKHLAS', 'AL_KAFIRUN', 'AL_KAUTHAR',
                          'AL_MASAD', 'AL_MAUN', 'AL_QURAISH', 'AN_NAS', 'AN_NASR']

# Loop through the array names and calculate distances
for array_name in array_names_to_compare:
    array = arrays.get(array_name)
    if array is not None:
        distances = np.linalg.norm(array - intermediate_output, axis=1)
        closest_index = np.argmin(distances)
        closest_distance_for_array = distances[closest_index]

        if closest_distance_for_array < closest_distance:
            closest_distance = closest_distance_for_array
            closest_array = array
            closest_array_name = array_name
            closest_index_within_array = closest_index

# Print the index and name of the closest array and its closest index
if closest_array_name is not None:
    print(f"Closest array name: {closest_array_name}")
    print(f"Closest index within the array: {closest_index_within_array+1}")
    print(f"Closest distance: {closest_distance}")
else:
    print("No valid closest array found.")
