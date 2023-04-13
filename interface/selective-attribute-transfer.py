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

import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


st.title('Perceptual Attribute Transfer')

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

@st.experimental_memo
def get_encoded_dict():
    print('Encoding real data')
    hit_scratches_data_dir = '/home/purnima/appdir/Github/DATA/GreatestHitsDB/seeds/hits_and_scratches_samples'
    realdata_file_dict = {}

    list_of_materials = ['metal', 'tile', 'ceramic', 'cloth', 'carpet', 'paper']

    stft_channels = 512
    n_frames = 256
    hop_size = 128
    sample_rate = 16000
    im_min = -1.0651559
    im_max = 0.9660724

    netE = st.session_state['netE'] 

    start_time = time.time()
    print('Starting getting real data encoded')
    # <material>_<hit_type>_<sample_id>_rate<rate=(0,1,2,3)>.wav --> Use only 0,1,2
    for filename in os.listdir(hit_scratches_data_dir):
        if filename.split('.')[0].split('_')[0] in list_of_materials:
            audio, _ = librosa.load(os.path.join(hit_scratches_data_dir, filename), sr=sample_rate)
            audio_pghi = preprocess_signal(audio)
            audio_pghi = zeropad(audio_pghi, n_frames * hop_size )
            audio_pghi = pghi_stft(audio_pghi)
            audio_pghi = renormalize(audio_pghi, (np.min(audio_pghi), np.max(audio_pghi)), (im_min, im_max))

            audio_pghi = torch.from_numpy(audio_pghi).float().cuda().unsqueeze(dim=0)
            mask = torch.ones_like(audio_pghi)[:, :1, :, :]
            net_input = torch.cat([audio_pghi, mask], dim=1).cuda()
            with torch.no_grad():
                encoded = netE(net_input)
                
            #realdata_file_dict[filename.split('.')[0]]=torch.stack([encoded] * 14, dim=1)
            realdata_file_dict[filename.split('.')[0]]=encoded

    print('Completed getting real data encoded in %s seconds ' % (time.time() - start_time))
    return realdata_file_dict

@st.experimental_memo
def get_curateddata_encoded_dict():
    print('Encoding Curated data')
    hit_scratches_data_dir = '/home/purnima/appdir/Github/DATA/GreatestHitsDB/seeds/hits_and_scratches_samples_for_metrics'
    curateddata_file_dict = {}

    stft_channels = 512
    n_frames = 256
    hop_size = 128
    sample_rate = 16000
    im_min = -1.0651559
    im_max = 0.9660724

    netE = st.session_state['netE'] 

    start_time = time.time()
    print('Starting getting curated data encoded')
    # <material>_<hit_type>_<sample_id>_rate<rate=(0,1,2,3)>.wav --> Use only 0,1,2
    for filename in os.listdir(hit_scratches_data_dir):
        if '.wav' in filename:
            audio, _ = librosa.load(os.path.join(hit_scratches_data_dir, filename), sr=sample_rate)
            audio_pghi = preprocess_signal(audio)
            audio_pghi = zeropad(audio_pghi, n_frames * hop_size )
            audio_pghi = pghi_stft(audio_pghi)
            audio_pghi = renormalize(audio_pghi, (np.min(audio_pghi), np.max(audio_pghi)), (im_min, im_max))

            audio_pghi = torch.from_numpy(audio_pghi).float().cuda().unsqueeze(dim=0)
            mask = torch.ones_like(audio_pghi)[:, :1, :, :]
            net_input = torch.cat([audio_pghi, mask], dim=1).cuda()
            with torch.no_grad():
                encoded = netE(net_input)
                
            #curateddata_file_dict[filename.split('.')[0]]=torch.stack([encoded] * 14, dim=1)
            curateddata_file_dict[filename.split('.')[0]]=encoded

    print('Completed getting curated data encoded in %s seconds ' % (time.time() - start_time))
    return curateddata_file_dict


@st.experimental_memo
def get_model():
    print('getting model')
    #GreatestHits
    checkpoint_num = '2800'
    network_pkl = '/home/purnima/appdir/Github/StyleGANs/audio-stylegan2/training-runs/vis-data-256-split/00000/network-snapshot-00{checkpoint_num}.pkl'.format(checkpoint_num=checkpoint_num)
    encoder_pkl = '/home/purnima/appdir/Github/audio-latent-composition/checkpoints/sgan_encoder_greatesthits_resnet-34_RGBM-corrected-nolpips-redo-corrloss/netE_epoch_best.pth'


    G = None
    if 'model' not in st.session_state:
        with open(network_pkl, 'rb') as pklfile:
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

@st.experimental_memo
def get_concept_directions():
    # Brightness
    brightness_vector = np.load('direction_vectors/brightness.npy')
    brightness_tinnyprototype = np.load('direction_vectors/brightness-tinnyprototype.npy')

    # Rate
    rate_vector = np.load('direction_vectors/rate.npy')
    rate_highrateprototype = np.load('direction_vectors/rate-highrateprototype.npy')

    # Impact type
    impacttype_vector = np.load('direction_vectors/impacttype.npy')
    impacttype_sharphitsprototype = np.load('direction_vectors/impacttype-sharphitsprototype.npy')

    brightness_vector = get_vector(brightness_vector/np.linalg.norm(brightness_vector)) #Can we do something with the magnitude?
    rate_vector = get_vector(rate_vector/np.linalg.norm(rate_vector)) #Can we do something with the magnitude?
    impacttype_vector = get_vector(impacttype_vector/np.linalg.norm(impacttype_vector)) #Can we do something with the magnitude?

    brightness_tinnyprototype = get_vector(brightness_tinnyprototype)
    rate_highrateprototype = get_vector(rate_highrateprototype)
    impacttype_sharphitsprototype = get_vector(impacttype_sharphitsprototype)

    return brightness_vector, brightness_tinnyprototype, rate_vector, rate_highrateprototype, impacttype_vector, impacttype_sharphitsprototype


def transfer_attribute():
    sample = st.session_state['initial_sample']
    ref_sample = st.session_state['initial_ref_sample']

    direction = None
    prototype = None
    slide_position_ref = None
    if st.session_state['transfer_attribute_selection'] == 'Brightness':
        direction = st.session_state['brightness_vector']
        prototype = st.session_state['brightness_tinnyprototype']
        slide_position_ref = 'brightness_slider_position'
    elif st.session_state['transfer_attribute_selection'] == 'Impact Type':
        direction = st.session_state['rate_vector']
        prototype = st.session_state['rate_highrateprototype']
        slide_position_ref = 'impacttype_slider_position'
    elif st.session_state['transfer_attribute_selection'] == 'Rate':
        direction = st.session_state['impacttype_vector']
        prototype = st.session_state['impacttype_sharphitsprototype']
        slide_position_ref = 'rate_slider_position'

    sample_direction = prototype - sample
    sample_proj = (sample_direction @ direction)[0]
    ref_sample_direction = prototype - ref_sample
    ref_sample_proj = (ref_sample_direction @ direction)[0]
    print(prototype.shape, sample.shape, sample_proj.shape, ref_sample_proj.shape, direction.shape)
    sample_modified = sample + (sample_proj-ref_sample_proj) * direction
    st.session_state[slide_position_ref] = torch.clamp(sample_proj-ref_sample_proj, -5.0, 5.0)
    return sample_modified, torch.clamp(sample_proj-ref_sample_proj, -5.0, 5.0)

def sample(pos, which_sample):
    G = st.session_state['G']
    netE = st.session_state['netE']

    brightness_vector, brightness_tinnyprototype, rate_vector, rate_highrateprototype, impacttype_vector, impacttype_sharphitsprototype = get_concept_directions()
    if 'brightness_vector' not in st.session_state:
        st.session_state['brightness_vector'] = brightness_vector
        st.session_state['brightness_tinnyprototype'] = brightness_tinnyprototype

        st.session_state['rate_vector'] = rate_vector
        st.session_state['rate_highrateprototype'] = rate_highrateprototype

        st.session_state['impacttype_vector'] = impacttype_vector
        st.session_state['impacttype_sharphitsprototype'] = impacttype_sharphitsprototype


    if which_sample not in st.session_state:
        z = torch.from_numpy(np.random.rand(1, G.z_dim)).cuda()
        w = G.mapping(z, None)
        st.session_state[which_sample] = w
        print(st.session_state[which_sample].shape)

    start_time = time.time()
    w = st.session_state[which_sample]
    if len(w.shape) < 3: #The w shape is 1X128. Need 1X14X128.
        w = torch.stack([w]*14, dim=1)

    w_ = w.clone()
    w_ += brightness_vector * pos[0]
    w_ += rate_vector * pos[1]
    w_ += impacttype_vector * pos[2]

    img = G.synthesis(w_)
    print("--- Time taken to synthesize from G = %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    img = (img  * 127.5+ 128).clamp(0, 255).to(torch.uint8)
    img = img.detach().cpu().numpy()[0]
    filler = np.full((1, 1, img[0][0].shape[0]), np.min(img))
    img_1 = np.append(img, filler, axis=1) # UNDOING THAT CODE!
    img_1 = img_1/255
    img_1 = -50+img_1*50

    audio = pghi_istft(img_1)

    fig =plt.figure(figsize=(7, 5))
    a=librosa.display.specshow(img_1[0],x_axis='time', y_axis='linear',sr=16000)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    print("--- Time taken to reconstruct audio via PGHI = %s seconds ---" % (time.time() - start_time))

    sf.write('/tmp/temp_audio_loc.wav', audio.astype(float), 16000)
    print('--------------------------------------------------')


    audio_file = open('/tmp/temp_audio_loc.wav', 'rb')
    audio_bytes = audio_file.read()

    print('8888888888888888**************************', img_arr.shape)

    return img_arr, audio_bytes


def change_origsample_z():
    selected_option = st.session_state['selected_curateddata_preset_option']
    print(selected_option)
    if selected_option == 'Random':
        if 'initial_sample' in st.session_state:
            del st.session_state['initial_sample'] 
    else:
        st.session_state['initial_sample'] = st.session_state['curated_data_dict'][selected_option]
    st.session_state['brightness_slider_position'] = 0
    st.session_state['rate_slider_position'] = 0
    st.session_state['impacttype_slider_position'] = 0

def change_refsample_z():
    selected_option = st.session_state['selected_realdata_preset_option']
    print(selected_option)
    if selected_option == 'Random':
        if 'initial_ref_sample' in st.session_state:
            del st.session_state['initial_ref_sample'] 
    else:
        st.session_state['initial_ref_sample'] = st.session_state['real_data_dict'][selected_option]
    st.session_state['brightness_slider_position'] = 0
    st.session_state['rate_slider_position'] = 0
    st.session_state['impacttype_slider_position'] = 0


def main():

    G, netE = get_model()
    if 'G' not in st.session_state:
        st.session_state['G'] = G
    if 'netE' not in st.session_state:
        st.session_state['netE'] = netE

    real_data_dict = get_encoded_dict()
    if 'real_data_dict' not in st.session_state:
        st.session_state['real_data_dict'] = real_data_dict

    curated_data_dict = get_curateddata_encoded_dict()
    if 'curated_data_dict' not in st.session_state:
        #st.session_state['curated_data_dict'] = curated_data_dict
        st.session_state['curated_data_dict'] = real_data_dict
    

    initialCuratedDataOptionsList = ['Random']
    optionlist = st.session_state['curated_data_dict'].keys() or set()
    initialCuratedDataOptionsList.extend(sorted(optionlist))
    curateddata_option = st.sidebar.selectbox('Choose **:red[Target]** Sample',initialCuratedDataOptionsList,key='selected_curateddata_preset_option', on_change=change_origsample_z)
    # st.sidebar.write('Your Selected Target Sample:', curateddata_option)


    initialRealDataOptionsList = ['Random']
    optionlist = st.session_state['real_data_dict'].keys() or set()
    initialRealDataOptionsList.extend(sorted(optionlist))
    realdata_option = st.sidebar.selectbox('Choose **:red[Reference]** Sample',initialRealDataOptionsList,key='selected_realdata_preset_option', on_change=change_refsample_z)
    transfer_attribute_selection = st.sidebar.radio("Choose Reference Attribute to Transfer", ('Brightness', 'Impact Type', 'Rate'), horizontal=True, key='transfer_attribute_selection')
    # st.sidebar.write('You Selected Reference Sample:', realdata_option)

    somehtml = '<div style="border-bottom: dashed grey 4px; ">&nbsp;</div><br/>'
    st.sidebar.markdown(somehtml, unsafe_allow_html=True)
    

    st.sidebar.title('Semantic Directions  For Target Sample')

    birghtness_position=st.sidebar.slider('Brightness', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,  format=None, key='brightness_slider_position', help=None, args=None, kwargs=None, disabled=False)
    rate_position=st.sidebar.slider('Rate', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,  format=None, key='rate_slider_position', help=None, args=None, kwargs=None, disabled=False)
    impacttype_position=st.sidebar.slider('Impact Type', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,  format=None, key='impacttype_slider_position', help=None, args=None, kwargs=None, disabled=False)
    
    col1, col2, col3 = st.columns((5,2,5))

    s = sample([birghtness_position, rate_position, impacttype_position], 'initial_sample')
    s_ref = sample([0,0,0], 'initial_ref_sample')

    with col1:
        colname = '<div style="padding-left: 30%;"><h3><b><i>Reference Sound</i></b></h3></div>'
        st.markdown(colname, unsafe_allow_html=True)
        st.image(s_ref[0])
        st.audio(s_ref[1], format="audio/wav", start_time=0)
    with col2:
        vert_space = '<div style="padding: 40%;"></div>'
        st.markdown(vert_space, unsafe_allow_html=True)
        st.button("**Transfer attribute** =>", on_click=transfer_attribute, type='primary')
    with col3:
        colname = '<div style="padding-left: 30%;"><h3><b><i>Target Sound</i></b></h3></div>'
        st.markdown(colname, unsafe_allow_html=True)
        st.image(s[0])
        st.audio(s[1], format="audio/wav", start_time=0)


if __name__ == '__main__':
    main()