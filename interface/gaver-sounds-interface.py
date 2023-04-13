import requests
import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import io 
import matplotlib.pyplot as plt

import struct

import os
import re
from typing import List, Optional

import click

import sys
sys.path.insert(0, '../')
import dnnlib
from networks import stylegan_encoder
from utils import util, training_utils, losses, masking, gaver_sounds, perceptual_guidance
import numpy as np
import torch

import librosa
import librosa.display
import soundfile as sf
import pickle

from tifresi.utils import load_signal
from tifresi.utils import preprocess_signal
from tifresi.stft import GaussTF, GaussTruncTF
from tifresi.transforms import log_spectrogram
from tifresi.transforms import inv_log_spectrogram

from scipy.signal import freqz,butter, lfilter
from PIL import Image

import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# st.title('Analysis-Synthesis')
somehtml = '<h1 style="text-align:center">Analysis-Synthesis In The Latent Space</h1>'
st.markdown(somehtml, unsafe_allow_html=True)

def pghi_stft(x):
    stft_channels = 512
    n_frames = 256
    hop_size = 128
    sample_rate = 16000

    stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
    Y = stft_system.spectrogram(x)
    log_Y= log_spectrogram(Y)
    return np.expand_dims(log_Y, axis=0)

def pghi_istft(x):
    stft_channels = 512
    n_frames = 256
    hop_size = 128
    sample_rate = 16000

    stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
    x = np.squeeze(x,axis=0)
    new_Y = inv_log_spectrogram(x)
    new_y = stft_system.invert_spectrogram(new_Y)
    return np.array(new_y)

def zeropad(signal, audio_length):
    if len(signal) < audio_length:
        return np.append(
            signal, 
            np.zeros(audio_length - len(signal))
        )
    else:
        signal = signal[0:audio_length]
        return signal

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

def get_vector(x):
    return torch.from_numpy(x).float().cuda()

def get_spectrogram(audio):
    stft_channels = 512
    n_frames = 256
    hop_size = 128
    sample_rate = 16000

    audio_pghi = preprocess_signal(audio)
    audio_pghi = zeropad(audio_pghi, n_frames * hop_size )
    audio_pghi = pghi_stft(audio_pghi)
    return audio_pghi

def applyFBFadeFilter(forward_fadetime,backward_fadetime,signal,fs,expo=1):
    forward_num_fad_samp = int(forward_fadetime*fs) 
    backward_num_fad_samp = int(backward_fadetime*fs) 
    signal_length = len(signal) 
    fadefilter = np.ones(signal_length)
    if forward_num_fad_samp>0:
        fadefilter[0:forward_num_fad_samp]=np.linspace(0,1,forward_num_fad_samp)**expo
    if backward_num_fad_samp>0:
        fadefilter[signal_length-backward_num_fad_samp:signal_length]=np.linspace(1,0,backward_num_fad_samp)**expo
    return fadefilter*signal

def butter_bandpass(lowcut, highcut, fs, order=5,btype='bandpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a

def butter_lowhighpass(cut, fs, order=5, btype='lowpass'):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = butter(order, cut, btype=btype)
    return b, a

def butter_bandpass_filter(data, highcut, fs,lowcut=None,  order=5, btype='bandpass'):
    if btype=='bandpass':
        b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    else:
        b, a = butter_lowhighpass(highcut, fs, order=order, btype=btype)
    y = lfilter(b, a, data)
    return y

def get_gaver_sounds(initial_amplitude, impulse_time, filters, total_time=2, locs=None, \
                             sample_rate=16000, hittype='hit', 
                             damping_mult=None, damping_fade_expo=None, 
                             filter_order=None):
    
    y_scratch = np.random.rand(int(impulse_time*sample_rate))
    
    #20%
    start_t = 0.0
    end_t = 0.05*impulse_time
    y1 = initial_amplitude*y_scratch[int(start_t*sample_rate):int(end_t*sample_rate)]
    y1 = 20*butter_bandpass_filter(y1, lowcut=filters[0][0], highcut=filters[0][1], fs=sample_rate, order=2, btype='bandpass')
    y1 = applyFBFadeFilter(forward_fadetime=0,backward_fadetime=0.1*(end_t-start_t),signal=y1,fs=sample_rate, expo=1)
    y1 = np.pad(y1, (int(start_t*sample_rate),len(y_scratch)-int(end_t*sample_rate)), mode='constant')
    
    #Remaining 80%
    start_t = 0.05*impulse_time
    end_t = 1.0*impulse_time
    y2 = initial_amplitude*y_scratch[int(start_t*sample_rate):int(end_t*sample_rate)]
    if not filter_order:
        filter_order = 1
    y2 = 10*butter_bandpass_filter(y2, lowcut=filters[1][0], highcut=filters[1][1], fs=sample_rate, order=filter_order, btype='bandpass')
    if not damping_mult:
        damping_mult = 0.1
        damping_fade_expo = 1
    y2 = applyFBFadeFilter(forward_fadetime=0,backward_fadetime=damping_mult*(end_t-start_t),signal=y2,fs=sample_rate, expo=damping_fade_expo)
    y2 = np.pad(y2, (int(start_t*sample_rate),len(y_scratch)-int(end_t*sample_rate)), mode='constant')
    
    
    y_scratch = y1+y2
    
    signal_mult = 0.00005
    if hittype == 'scratch':
        signal_mult = 0.0005
    signal = signal_mult*np.random.randn(int(total_time*sample_rate))
    
    for loc in locs:
        start_loc = int(loc*sample_rate)
        end_loc = start_loc+len(y_scratch)
        y_scratch_ = y_scratch

        if end_loc > len(signal):
            end_loc = len(signal)
            y_scratch_ = y_scratch_[0:end_loc-start_loc]

        signal[start_loc:end_loc] = y_scratch_

    signal = signal/np.max(signal)
    sf.write('/tmp/temp_signal_loc.wav', signal.astype(float), 16000)
    audio_file = open('/tmp/temp_signal_loc.wav', 'rb')
    audio_bytes = audio_file.read()
    
    fig =plt.figure(figsize=(7, 5))
    a=librosa.display.specshow(get_spectrogram(signal)[0],x_axis='time', y_axis='linear',sr=16000, hop_length=128)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    st.session_state['gaver_audio_loc'] = '/tmp/temp_signal_loc.wav'

    return audio_bytes, img_arr#, '/tmp/temp_signal_loc.wav'

@st.cache_data
def get_model():
    print('getting model')
    stylegan_pkl = "../checkpoints/stylegan2/greatesthits/network-snapshot-002800.pkl"
    encoder_pkl = "../checkpoints/encoder/greatesthits/netE_epoch_best.pth"

    stylegan_pkl_url = "https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/audio-stylegan2/greatesthits/network-snapshot-002800.pkl"
    encoder_pkl_url = "https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/encoder/greatesthits/netE_epoch_best.pth"

    if not os.path.isfile(stylegan_pkl):
        os.makedirs("../checkpoints/stylegan2/greatesthits/", exist_ok=True)
        urllib.request.urlretrieve(stylegan_pkl_url, stylegan_pkl)

    if not os.path.isfile(encoder_pkl):
        os.makedirs("../checkpoints/encoder/greatesthits/", exist_ok=True)
        urllib.request.urlretrieve(encoder_pkl_url, encoder_pkl)

    G = None
    if 'G' not in st.session_state:
        with open(stylegan_pkl, 'rb') as pklfile:
            network = pickle.load(pklfile)
            G = network['G'].eval().cuda()

    netE = None
    if 'netE' not in st.session_state:
        netE = stylegan_encoder.load_stylegan_encoder(domain=None, nz=G.z_dim,
                                                   outdim=128,
                                                   use_RGBM=True,
                                                   use_VAE=False,
                                                   resnet_depth=34,
                                                   ckpt_path=encoder_pkl).eval().cuda()
    return G, netE


def encode_and_reconstruct(audio):
    audio_pghi = preprocess_signal(audio)
    G = st.session_state['G']
    netE = st.session_state['netE']
    
    stft_channels = 512 #Move these constants to a config file.
    n_frames = 256
    hop_size = 128
    sample_rate = 16000
    im_min = -1.0651559
    im_max = 0.9660724
    pghi_min = -50
    pghi_max = 0
    
    audio_pghi = util.zeropad(audio_pghi, n_frames * hop_size )
    audio_pghi = util.pghi_stft(audio_pghi, hop_size=hop_size, stft_channels=stft_channels)
    audio_pghi = util.renormalize(audio_pghi, (np.min(audio_pghi), np.max(audio_pghi)), (im_min, im_max))

    audio_pghi = torch.from_numpy(audio_pghi).float().cuda().unsqueeze(dim=0)
    mask = torch.ones_like(audio_pghi)[:, :1, :, :]
    net_input = torch.cat([audio_pghi, mask], dim=1).cuda()
    
    with torch.no_grad():
        encoded = netE(net_input)

    reconstructed_audio = G.synthesis(torch.stack([encoded] * 14, dim=1))
    filler = torch.full((1, 1, 1, reconstructed_audio[0].shape[1]), torch.min(reconstructed_audio)).cuda()
    reconstructed_audio = torch.cat([reconstructed_audio, filler], dim=2)
    reconstructed_audio = util.renormalize(reconstructed_audio, (torch.min(reconstructed_audio), torch.max(reconstructed_audio)), (pghi_min, pghi_max))
    reconstructed_audio = reconstructed_audio.detach().cpu().numpy()[0]
    reconstructed_audio_wav = util.pghi_istft(reconstructed_audio, hop_size=hop_size, stft_channels=stft_channels)
    
    return encoded, reconstructed_audio_wav


def sample():
    audio_loc = st.session_state['gaver_audio_loc']

    audio, sr = librosa.load(audio_loc, sr=16000)
    G = st.session_state['G']
    netE = st.session_state['netE']

    encoded, reconstructed_audio_wav = encode_and_reconstruct(audio)

    sf.write('/tmp/reconstructed_audio_wav_recon.wav', reconstructed_audio_wav.astype(float), 16000)
    audio_file = open('/tmp/reconstructed_audio_wav_recon.wav', 'rb')
    audio_bytes = audio_file.read()
    
    fig =plt.figure(figsize=(7, 5))
    a=librosa.display.specshow(get_spectrogram(reconstructed_audio_wav)[0],x_axis='time', y_axis='linear',sr=16000, hop_length=128)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    
    st.session_state['audio_bytes'] = audio_bytes
    st.session_state['img_arr'] = img_arr



def main():

    G, netE = get_model()
    if 'G' not in st.session_state:
        st.session_state['G'] = G
    if 'netE' not in st.session_state:
        st.session_state['netE'] = netE


    rate_locs_0_per_sec = [0.05]
    rate_locs_1_per_sec = [0.05, 1.05]
    rate_locs_2_per_sec = [0.05, 0.9, 1.75]
    rate_locs_2irreg_per_sec = [0.05, 1.25, 1.75]
    rate_locs_3_per_sec = [0.05, 0.45, 0.85, 1.25, 1.65]
    rate_locs_4_per_sec = [0.05, 0.35, 0.65, 0.95, 1.25, 1.55, 1.85]
    locs = [rate_locs_0_per_sec, rate_locs_1_per_sec, rate_locs_2_per_sec, rate_locs_3_per_sec, rate_locs_4_per_sec]


    st.sidebar.title('Parameters Options')

    impact_type = 'hit'
    rate =  st.sidebar.selectbox('Rate', (0,1,2,3,4), key='rate_position',)
    
    impulse_time = st.sidebar.slider('Impulse Width', min_value=0.0, max_value=2.0, value=0.05, step=0.01,  format=None, key='impulse_width_position', help=None, args=None, kwargs=None, disabled=False)

    
    attack_lf, attack_hf = st.sidebar.select_slider(
                            'Select a frequency band for attack part',
                            options=np.arange(10,7999,10),
                            value=(10, 700))
    
    trial_lf, trial_hf = st.sidebar.select_slider(
                            'Select a frequency band for trailing part',
                            options=np.arange(10,7999,10),
                            value=(10, 700))
    
    filter_order = st.sidebar.slider('Filter Order', min_value=1.0, max_value=5.0, value=1.0, step=1.0,  format=None, key='filter_order_position', help=None, args=None, kwargs=None, disabled=False)

    damping_mult = st.sidebar.slider('Damping', min_value=0.1, max_value=1.0, value=0.1, step=0.1,  format=None, key='damping_mult_position', help=None, args=None, kwargs=None, disabled=False)
    damping_fade_expo = st.sidebar.slider('Damping Fade Exponent', min_value=1.0, max_value=3.0, value=1.0, step=1.0,  format=None, key='damping_fade_expo_position', help=None, args=None, kwargs=None, disabled=False)
    
    
    col1, col2, col3 = st.columns((5,2,5))

    s, s_pghi = get_gaver_sounds(initial_amplitude=1.0, hittype=impact_type, total_time=2.0, impulse_time=impulse_time, sample_rate=16000,\
                        filters=[(attack_lf, attack_hf), (trial_lf, trial_hf)], locs=locs[rate],\
                        filter_order=filter_order, damping_mult=damping_mult, damping_fade_expo=damping_fade_expo)
    
    if 'audio_bytes' not in st.session_state:
        st.session_state['audio_bytes'] = byte_array = bytes([])
        st.session_state['img_arr'] = np.zeros((500,700,4))
    
    s_recon = st.session_state['audio_bytes']
    s_recon_pghi = st.session_state['img_arr']


    with col1:
        colname = '<div style="padding-left: 30%;"><h3><b><i>Synthetic Reference</i></b></h3></div>'
        st.markdown(colname, unsafe_allow_html=True)
        st.image(s_pghi)
        st.audio(s, format="audio/wav", start_time=0)
    with col2:
        vert_space = '<div style="padding: 40%;"></div>'
        st.markdown(vert_space, unsafe_allow_html=True)
        st.button("**Query Latent Space** =>", on_click=sample, type='primary')
    with col3:
        colname = '<div style="padding-left: 30%;"><h3><b><i>Reconstructed Audio</i></b></h3></div>'
        st.markdown(colname, unsafe_allow_html=True)
        st.image(s_recon_pghi)
        st.audio(s_recon, format="audio/wav", start_time=0)#, sample_rate=16000)


if __name__ == '__main__':
    main()