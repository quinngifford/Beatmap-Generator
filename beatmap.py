import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.layers import LSTM, TimeDistributed
import librosa
import numpy as np
import pandas as pd

beat_times = pd.read_csv("beat_times.csv")


def preprocess_audio(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.stft(signal)
    log_spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))
    return log_spectrogram


def create_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(None, 1025)))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    return model


def train_model(model, spectrograms, beat_times):
    spectrograms = np.array(spectrograms)
    beat_times = np.array(beat_times)
    print(spectrograms.shape)
    spectrograms = spectrograms.reshape(
        spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(spectrograms, beat_times, epochs=10, batch_size=32)


def main():
    model = create_model()
    model.summary()
    train_model(model, preprocess_audio(), beat_times)
    tf.k3eras.models.save_model(model, "beatmap_model.h5")


if __name__ == "__main__":
    main()
