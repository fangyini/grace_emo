import librosa
import numpy as np
import soundfile as sf
import os

config = {
    'sample_interval': 1/30,
    'window_len': 0.025,
    'n_mfcc': 14,
    'fps': 30
}

def extract_mfcc(file_path, target_num_frames):
    mfccs = mfcc_from_file(file_path, target_num_frames=target_num_frames)
    return mfccs.transpose()
    '''input_list = split_input_target(mfccs)
    input_list = np.stack(input_list)
    return input_list'''

# get mfcc feature from audio file
def mfcc_from_file(audio_path, n_mfcc=config['n_mfcc'], sample_interval=config['sample_interval'], window_len=config['window_len'],
                   target_num_frames=None):
    # Check file extension
    file_ext = os.path.splitext(audio_path)[1].lower()
    
    try:
        if file_ext in ['.m4a', '.mp3', '.wav']:
            # Use librosa for M4A, MP3, and WAV files
            audio, sr = librosa.load(audio_path, sr=None)
        else:
            # Use soundfile for other formats
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {str(e)}")
        raise

    hop_length = int(sample_interval * sr)
    n_fft = int(window_len * sr)

    if target_num_frames is not None:
        # Compute expected length of audio to yield exactly target_num_frames MFCC steps
        expected_len = hop_length * (target_num_frames - 1) + n_fft
        if len(audio) < expected_len:
            pad_len = expected_len - len(audio)
            audio = np.pad(audio, (0, pad_len), mode='constant')
        elif len(audio) > expected_len:
            audio = audio[:expected_len]

    # Calculate MFCC features
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    # Calculate delta features
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs = np.concatenate((mfccs, mfccs_delta), axis=0)
    
    # Normalize MFCC features
    mean, std = np.mean(mfccs, axis=0), np.std(mfccs, axis=0)
    mfccs = (mfccs-mean)/std

    '''padding_front = np.zeros((28, 15))
    padding_back = np.zeros((28, 15))
    front = np.column_stack((padding_front, mfccs))
    mfccs = np.column_stack((front, padding_back))'''
    return mfccs

def split_input_target(mfccs):
    #assert mfccs.shape[1] == parameters.shape[1], 'Squence length in mfcc and parameter is different'
    #assert phonemes.shape[1] == parameters.shape[1], 'Squence length in phoneme and parameter is different'
    seq_len = mfccs.shape[1]
    inputs = mfccs
    # target: parameter at a time, input: silding window (past 80, future 20)
    input_list = []
    for idx in range(15, seq_len-15):
        input_list.append(inputs[:, idx-15:idx+15])
    return input_list


