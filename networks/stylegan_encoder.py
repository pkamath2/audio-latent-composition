import torch, os
from utils import customnet, util
from argparse import Namespace
# from utils.pt_stylegan2 import get_generator
from collections import OrderedDict
import torch.nn as nn
from torch.nn.functional import interpolate


def load_stylegan_encoder(domain, nz=512*14, outdim=256, use_RGBM=True, use_VAE=False,
                         resnet_depth=34, ckpt_path=None):
    halfsize = False # hardcoding
    channels_in = 2 if use_RGBM or use_VAE else 3
    print(f"Using halfsize?: {halfsize}")
    print(f"Input channels: {channels_in}")
    encoder = get_stylegan_encoder(ndim_z=nz, resnet_depth=resnet_depth,
                                   halfsize=halfsize, channels_in=channels_in)
    if ckpt_path is None: # does not load weights
        return encoder

    if util.is_url(ckpt_path):
        ckpt = torch.hub.load_state_dict_from_url(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path)
    encoder.load_state_dict(ckpt['state_dict'])
    encoder = encoder.eval()
    return encoder

## code from Jonas
def get_stylegan_encoder(ndim_z=512, add_relu=False, resnet_depth=34, halfsize=True, channels_in=3):
    """
    Return encoder. Change to get a different encoder.
    """
    def make_resnet(halfsize=True, resize=True, ndim_z=512, add_relu=False, resnet_depth=34, channels_in=3):
        # A resnet with the final FC layer removed.
        # Instead, we have a final conv5, leaky relu, and global average pooling.
        native_size = 128 if halfsize else 256
        # Make an encoder model.
        def change_out(layers):
            numch = 512 if resnet_depth < 50 else 2048
            ind = [i for i, (n, l) in enumerate(layers) if n == 'layer4'][0] + 1
            newlayer = ('layer5',
                torch.nn.Sequential(OrderedDict([
                    ('conv5', torch.nn.Conv2d(numch, ndim_z, kernel_size=1)),
                ])))

            layers.insert(ind, newlayer)

            if resize:
                layers[:0] = [('downsample',
                    InterpolationLayer(size=(native_size, native_size)))]

            # Remove FC layer
            layers = layers[:-1]

            if add_relu:
                layers.append( ('postrelu', torch.nn.LeakyReLU(0.2) ))

            # add reshape layer
            layers.append(('to_wplus', customnet.EncoderToWplus(wdim=128)))

            return layers

        encoder = customnet.CustomResNet(
                resnet_depth, modify_sequence=change_out, halfsize=halfsize,
            channels_in=channels_in)

        # Init using He initialization
        def init_weights(m):
            if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        encoder.apply(init_weights)
        return encoder

    encoder = make_resnet(ndim_z=ndim_z, add_relu=add_relu ,resnet_depth=resnet_depth,
                          channels_in=channels_in, halfsize=halfsize)

    return encoder


class InterpolationLayer(nn.Module):
    def __init__(self, size):
        super(InterpolationLayer, self).__init__()

        self.size=size

    def forward(self, x):
        return interpolate(x, size=self.size, mode='area')
