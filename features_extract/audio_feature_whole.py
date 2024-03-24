# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 22:35:13 2022

@author: lenovo
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import wave
import librosa
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()
import loupe_keras as lpk


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

cluster_size = 16

min_len = 100
max_len = -1
# In[]
def wav2vlad(wave_data, sr):
    global cluster_size
    signal = wave_data
    melspec = librosa.feature.melspectrogram(y=signal, n_mels=80,sr=sr).astype(np.float32).T
    melspec = np.log(np.maximum(1e-6, melspec))
    feature_size = melspec.shape[1]
    max_samples = melspec.shape[0]
    output_dim = cluster_size * 16
    feat = lpk.NetVLAD(feature_size=feature_size, max_samples=max_samples, \
                            cluster_size=cluster_size, output_dim=output_dim) \
                                (tf.convert_to_tensor(melspec))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        r = feat.numpy() 
    return r
        
def extract_features(number, audio_features, targets,path):
    global max_len, min_len
    if not os.path.exists(os.path.dirname(os.getcwd())+"/EATD-Corpus/"+str(path)+str(number)):
        return    
    positive_file = wave.open(os.path.dirname(os.getcwd())+"/EATD-Corpus/"+str(path)+str(number)+"/positive_out.wav")
    sr1 = positive_file.getframerate()
    nframes1 = positive_file.getnframes()
    wave_data1 = np.frombuffer(positive_file.readframes(nframes1), dtype=np.short).astype(float)
    len1 = nframes1 / sr1

    neutral_file = wave.open(os.path.dirname(os.getcwd())+"/EATD-Corpus/"+str(path)+str(number)+"/neutral_out.wav")
    sr2 = neutral_file.getframerate()
    nframes2 = neutral_file.getnframes()
    wave_data2 = np.frombuffer(neutral_file.readframes(nframes2), dtype=np.short).astype(float)
    len2 = nframes2 / sr2

    negative_file = wave.open(os.path.dirname(os.getcwd())+"/EATD-Corpus/"+str(path)+str(number)+"/negative_out.wav")
    sr3 = negative_file.getframerate()
    nframes3 = negative_file.getnframes()
    wave_data3 = np.frombuffer(negative_file.readframes(nframes3), dtype=np.short).astype(float)
    len3 = nframes3/sr3

    for l in [len1, len2, len3]:
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l

    with open(os.path.dirname(os.getcwd())+"/EATD-Corpus/"+str(path)+str(number)+"/new_label.txt") as fli:
        target = float(fli.readline())
    
    if wave_data1.shape[0] < 1:
        wave_data1 = np.array([1e-4]*sr1*5)
    if wave_data2.shape[0] < 1:
        wave_data2 = np.array([1e-4]*sr2*5)
    if wave_data3.shape[0] < 1:
        wave_data3 = np.array([1e-4]*sr3*5)  
    audio_features.append([wav2vlad(wave_data1, sr1), wav2vlad(wave_data2, sr2),wav2vlad(wave_data3, sr3)])
    targets.append(target)

# In[]
from tqdm import tqdm
audio_features = []
audio_targets = []

for index in tqdm(range(112)):
    extract_features(index+1, audio_features, audio_targets,"t_")
# In[]
for index in range(114):
    extract_features(index+1, audio_features, audio_targets, 'v_')
# In[]
print("Saving npz file locally...")
audio_features = np.array(audio_features)
audio_targets = np.array(audio_targets)
print(audio_features.shape)
print(audio_targets.shape)
np.savez(os.path.dirname(os.getcwd())+"/audio_feature.npz", audio_features)
np.savez(os.path.dirname(os.getcwd())+"/audio_target.npz", audio_targets)