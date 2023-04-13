import requests
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
import os, sys, io, json
import librosa, librosa.display
import soundfile as sf
import torch
import IPython
from IPython.display import Audio, display
import pickle
import urllib.request

from tifresi.utils import preprocess_signal

import sys
sys.path.insert(0, '../')
import dnnlib
from utils import util, training_utils, losses, masking, gaver_sounds, perceptual_guidance
from networks import stylegan_encoder
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


st.markdown("<h2 style='text-align: center;'>Water Generation <br/>Guided by Semantic Prototypes</h2>", unsafe_allow_html=True)


def get_vector(x):
    return torch.from_numpy(x).float().cuda()

def reconstruct(encoded):
    G = st.session_state['G']
    reconstructed_audio = G.synthesis(encoded)
    filler = torch.full((1, 1, 1, reconstructed_audio[0].shape[1]), torch.min(reconstructed_audio)).cuda()
    reconstructed_audio = torch.cat([reconstructed_audio, filler], dim=2)
    reconstructed_audio = util.renormalize(reconstructed_audio, (torch.min(reconstructed_audio), torch.max(reconstructed_audio)), (-50, 0))
    reconstructed_audio = reconstructed_audio.detach().cpu().numpy()[0]
    reconstructed_audio_wav = util.pghi_istft(reconstructed_audio, hop_size=128, stft_channels=512)
    return reconstructed_audio_wav, reconstructed_audio

@st.cache_data
def get_model():
    print('getting model')
    stylegan_pkl = "../checkpoints/stylegan2/water/network-snapshot-001400.pkl"
    encoder_pkl = "../checkpoints/encoder/water/netE_epoch_best.pth"

    stylegan_pkl_url = "https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/audio-stylegan2/water/network-snapshot-001400.pkl"
    encoder_pkl_url = "https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/encoder/water/netE_epoch_best.pth"

    if not os.path.isfile(stylegan_pkl):
        os.makedirs("../checkpoints/stylegan2/water/", exist_ok=True)
        urllib.request.urlretrieve(stylegan_pkl_url, stylegan_pkl)

    if not os.path.isfile(encoder_pkl):
        os.makedirs("../checkpoints/encoder/water/", exist_ok=True)
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

@st.cache_data
def get_concept_directions():
    filllevel_vector = np.load('direction_vectors/water-filllevel.npy')
    filllevel_vector = get_vector(filllevel_vector/np.linalg.norm(filllevel_vector)) #Can we do something with the magnitude?

    return -1 * filllevel_vector

def sample(pos):
    G = st.session_state['G']
    netE = st.session_state['netE']

    filllevel_vector = get_concept_directions()

    if 'initial_sample' not in st.session_state:
        with np.load('direction_vectors/water_fill_startz.npz') as data:
            z = torch.from_numpy(data['z']).float().cuda()
        w = G.mapping(z, None)
        st.session_state['initial_sample'] = w
        print('Initiating sample', st.session_state['initial_sample'].shape)

    start_time = time.time()
    w = st.session_state['initial_sample']

    w_ = w.clone()
    w_ += filllevel_vector * pos[0]

    img = G.synthesis(w_)    
    audio, img_1 = reconstruct(w_)
    print("--- Time taken to synthesize from G and invert using PGHI = %s seconds ---" % (time.time() - start_time))

    fig =plt.figure(figsize=(7, 5))
    a=librosa.display.specshow(img_1[0],x_axis='time', y_axis='linear',sr=16000)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    sf.write('/tmp/temp_audio_loc.wav', audio.astype(float), 16000)
    print('--------------------------------------------------')


    audio_file = open('/tmp/temp_audio_loc.wav', 'rb')
    audio_bytes = audio_file.read()

    return img_arr, audio_bytes


def main():
    
    G, netE = get_model()
    if 'G' not in st.session_state:
        st.session_state['G'] = G
    if 'netE' not in st.session_state:
        st.session_state['netE'] = netE

    st.sidebar.title('Semantic Directions')


    filllevel_position=st.sidebar.slider('Fill Level', min_value=0.0, max_value=13.0, value=0.0, step=1.0,  format=None, key='filllevel_slider_position', help=None, args=None, kwargs=None, disabled=False)
    spectrogram_placeholder = st.empty()
    audio_placeholder = st.empty()

    s = sample([filllevel_position])
    spectrogram_placeholder.image(s[0])
    audio_placeholder.audio(s[1], format="audio/wav", start_time=0)

if __name__ == '__main__':
    main()