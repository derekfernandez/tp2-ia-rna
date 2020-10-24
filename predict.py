from preprocess import *
import config

def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, config.n_mels, config.n_frames)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]
