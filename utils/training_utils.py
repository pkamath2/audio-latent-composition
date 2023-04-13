import torch
import argparse
import itertools
import numpy as np

### arguments

# Moved to config.json
# def make_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt_path', type=str, required=True, help="Checkpoint")
#     parser.add_argument('--outf', default='.', help='folder to output model checkpoints')
#     parser.add_argument('--seed', default=0, type=int, help='manual seed')
#     parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
#     parser.add_argument('--netE_type', type=str, required=True, help='type of encoder architecture; e.g. resnet-18, resnet-34')
#     parser.add_argument('--netE', default='', help="path to netE (to continue training)")
#     parser.add_argument('--finetune', type=str, default='', help="finetune from weights at this path")
#     parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
#     parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
#     parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
#     parser.add_argument('--lambda_latent', default=1.0, type=float, help='loss weighting (latent recovery)')
#     parser.add_argument('--lambda_mse', default=1.0, type=float, help='loss weighting (image mse)')
#     parser.add_argument('--lambda_lpips', default=0.0, type=float, help='loss weighting (image perceptual)')
#     parser.add_argument('--thresholding', default=None, type=int, help='Thresholding for the generated spectrogram while training')

#     return parser

### checkpointing

def make_checkpoint(netE, optimizer, epoch, val_loss, save_path):
    sd = {
        'state_dict': netE.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(sd, save_path)

def make_ipol_layer(size):
    return torch.nn.AdaptiveAvgPool2d((size, size))
    # return InterpolationLayer(size)

### dataset utilities

def training_loader(batch_size, global_seed=0):
    '''
    Returns an infinite generator that runs through randomized z
    batches, forever.
    '''
    n=10000
    z_dim=128

    g_epoch = 1
    while True:
        rng = np.random.RandomState(g_epoch+global_seed)
        z_data = torch.from_numpy(rng.standard_normal(n * z_dim)
                .reshape(n, z_dim)).float()
            
        dataloader = torch.utils.data.DataLoader(
                z_data,
                shuffle=False,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True)
        for batch in dataloader:
            yield batch
        g_epoch += 1

def testing_loader(batch_size, global_seed=0):
    '''
    Returns an a short iterator that returns a small set of test data.
    '''
    n=10*batch_size
    z_dim=128
    rng = np.random.RandomState(global_seed)
    z_data = torch.from_numpy(rng.standard_normal(n * z_dim)
            .reshape(n, z_dim)).float()
    
    dataloader = torch.utils.data.DataLoader(
            z_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True)
    return dataloader

def epoch_grouper(loader, epoch_size, num_epochs=None):
    '''
    To use with the infinite training loader: groups the training data
    batches into epochs of the given size.
    '''
    it = iter(loader)
    epoch = 0
    while True:
        chunk_it = itertools.islice(it, epoch_size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)
        epoch += 1
        if num_epochs is not None and epoch >= num_epochs:
            return