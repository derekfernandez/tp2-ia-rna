import librosa
import config
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

def get_labels(path=config.DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

def wav2mfcc(file_path, max_len=config.n_frames):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=14, n_fft=2048, hop_length=128, win_length=512, center=True, fmin=30, fmax=3000, window='hamming')

    # Si el audio es mas corto que lo definido por config, completar con 0s a ambos lados del eje
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Si es mas largo, se recorta
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

def save_data_to_array(path=config.DATA_PATH, max_len=config.n_frames):
    labels, _, _ = get_labels(path)

    for label in labels:

        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


def get_train_test(path=config.DATA_PATH, split_ratio=0.6, random_state=42):

    labels, indices, _ = get_labels(path)

    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)


