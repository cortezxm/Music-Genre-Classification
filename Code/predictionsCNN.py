import os
import math
import librosa
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

# Preprocessing configuration
SAMPLE_RATE = 22050

# Load the trained model
model = load_model('MGC_mlp.h5')

# Function to preprocess an individual audio file
def preprocess_audio(file_path, segment_duration=3, n_mfcc=13, nfft=2048, hop_length=512):
    # Load the complete audio file
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=signal, sr=sr)

    num_samples_per_segment = int(SAMPLE_RATE * segment_duration)
    expected_num_mfcc_vector_per_segment = math.ceil(num_samples_per_segment / hop_length)
    num_segments = int(duration // segment_duration)

    mfccs = []

    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment

        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=nfft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)

        mfcc = mfcc.T

        if len(mfcc) == expected_num_mfcc_vector_per_segment:
            mfccs.append(mfcc)

    return np.array(mfccs)


# Load the label mapping from the JSON
with open('data.json', 'r') as fp:
    data = json.load(fp)
    mapping = data["mapping"]
    genres = [genre for genre in mapping]

# Path to the .wav file to test
file_path = 'wav Files/Earth Wind & Fire - Lets Groove (Audio).wav'

# Predict the genre
preprocessed_audio = preprocess_audio(file_path, segment_duration=3)
predictions = model.predict(preprocessed_audio)

# Average the predictions of all segments
average_prediction = np.mean(predictions, axis=0)

# Print the percentage of each genre
print("This audio is:")
for pred, genre in zip(average_prediction, genres):
    print("{:.2%} of {}".format(pred, genre))

# Print the prediction
predicted_index = np.argmax(average_prediction)
predicted_genre = genres[predicted_index]
print("The predicted genre is: {}".format(predicted_genre))
