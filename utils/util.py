import torch
import urllib
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython
from IPython.display import Audio, display

from tifresi.utils import load_signal
from tifresi.utils import preprocess_signal
from tifresi.transforms import log_spectrogram
from tifresi.transforms import inv_log_spectrogram
from tifresi.stft import GaussTF, GaussTruncTF

def set_requires_grad(requires_grad, *models):
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, 'unknown type %r' % type(model)

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_config(filepath='config/config.json'):
    config = {}
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

def threshold_spectrogram(im0, threshold=17):
    im0_ = (im0  * 127.5+ 128).clamp(0, 255)/255.0
    im0_ = -50+im0_*50
    
    im0_ = -im0_
    im0_ = (im0_ * ((im0_>threshold).int()*1000)).clamp(0, 50)
    im0_ = -im0_ 

    im0_ = (im0_ + 50.0)/50.0
    im0_ = ((im0_ * 255.0) - 128.0)/127.5
    return im0_

def zeropad(signal, audio_length):
    if len(signal) < audio_length:
        return np.append(
            signal, 
            np.zeros(audio_length - len(signal))
        )
    else:
        signal = signal[0:audio_length]
        return signal

def pghi_stft(x, hop_size=128, stft_channels=512, use_truncated_window=True):
    if use_truncated_window:
        stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
    else:
        stft_system = GaussTF(hop_size=hop_size, stft_channels=stft_channels)
    Y = stft_system.spectrogram(x)
    log_Y= log_spectrogram(Y)
    return np.expand_dims(log_Y,axis=0)

def pghi_istft(x, hop_size=128, stft_channels=512, use_truncated_window=True):
    if use_truncated_window:
        stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
    else:
        stft_system = GaussTF(hop_size=hop_size, stft_channels=stft_channels)

    x = np.squeeze(x,axis=0)
    new_Y = inv_log_spectrogram(x)
    new_y = stft_system.invert_spectrogram(new_Y)
    return new_y

########################################## Notebook Utils############################################


def read_and_plot(f, hop_size=128, stft_channels=512, n_frames=256, title=None, sample_rate=16000):
    y, sr = librosa.load(f, sr=sample_rate)
    plot(y, title)

def plot(y, hop_size=128, stft_channels=512, n_frames=256, title=None, sample_rate=16000):
    IPython.display.display(IPython.display.Audio(y, rate=sample_rate))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    y_pghi = y
    y_pghi = preprocess_signal(y_pghi)
    y_pghi = zeropad(y_pghi, n_frames * hop_size )
    y_pghi = pghi_stft(y_pghi)
    
    a=librosa.display.specshow(y_pghi[0],x_axis='time', y_axis='linear', ax=ax[0], sr=sample_rate, hop_length=hop_size)
    ax[0].set_title(title)
    
    ax[1].plot(y)
    
def plot_multi(y1, y2, hop_size=128, stft_channels=512, n_frames=256, title1=None, title2=None, sample_rate=16000):
    IPython.display.display(IPython.display.Audio(y1, rate=sample_rate))
    IPython.display.display(IPython.display.Audio(y2, rate=sample_rate))
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    y_pghi = y1
    y_pghi = preprocess_signal(y_pghi)
    y_pghi = zeropad(y_pghi, n_frames * hop_size )
    y_pghi = pghi_stft(y_pghi)
    
    a=librosa.display.specshow(y_pghi[0],x_axis='time', y_axis='linear', ax=ax[0], sr=sample_rate, hop_length=hop_size)
    ax[0].set_title(title1)
    
    y_pghi = y2
    y_pghi = preprocess_signal(y_pghi)
    y_pghi = zeropad(y_pghi, n_frames * hop_size )
    y_pghi = pghi_stft(y_pghi)
    
    a=librosa.display.specshow(y_pghi[0],x_axis='time', y_axis='linear', ax=ax[1], sr=sample_rate, hop_length=hop_size)
    ax[1].set_title(title2)