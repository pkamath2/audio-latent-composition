import os
import pickle
import oyaml as yaml
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
from tqdm import tqdm

from utils import util, training_utils, losses, masking
from networks import stylegan_encoder


def train(opt):
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    batch_size = int(opt.batchSize)

    cudnn.benchmark = True


    with open(opt.ckpt_path, 'rb') as pklfile:
        network = pickle.load(pklfile)

    G = network['G']
    util.set_requires_grad(False, G)
    print(G)
    G.to(device).eval()
    outdim = 128# AFAIK not used anywhere. What does this do??
    nz = G.z_dim #z-dim 128 for us
    # nz = G.z_dim * 14 #z-dim 128 for us
    label = torch.zeros([1, G.z_dim], device=device)

    depth = int(opt.netE_type.split('-')[-1])
    has_masked_input = True 

    netE = stylegan_encoder.load_stylegan_encoder(domain=None, nz=nz,
                                                   outdim=outdim,
                                                   use_RGBM=True,#opt.masked,
                                                   use_VAE=False,#opt.vae_like,
                                                   resnet_depth=depth,
                                                   ckpt_path=None)
    netE = netE.to(device).train()
    summary(netE, input_size=(2,256,256))# Prints summary/model architecture on command line

    # losses + optimizers
    mse_loss = nn.MSELoss()
    # l1_loss = nn.L1Loss()
    perceptual_loss = losses.LPIPS_Loss(net='vgg', use_gpu=has_cuda)
    util.set_requires_grad(False, perceptual_loss)

    reshape = training_utils.make_ipol_layer(256)
    optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    start_ep = 0
    best_val_loss = float('inf')

    # latent datasets
    train_loader = training_utils.training_loader(batch_size, opt.seed)
    test_loader = training_utils.testing_loader(batch_size, opt.seed)

    # load data from checkpoint
    assert(not (opt.netE and opt.finetune)), "specify 1 of netE or finetune"
    if opt.finetune:
        checkpoint = torch.load(opt.finetune)
        sd = checkpoint['state_dict']
        netE.load_state_dict(sd)
    if opt.netE:
        checkpoint = torch.load(opt.netE)
        netE.load_state_dict(checkpoint['state_dict'])
        optimizerE.load_state_dict(checkpoint['optimizer'])
        start_ep = checkpoint['epoch'] + 1
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']

    
    # uses 1600 samples per epoch, computes number of batches
    # based on batch size
    epoch_batches = 1600 // batch_size
    for epoch, epoch_loader in enumerate(tqdm(
        training_utils.epoch_grouper(train_loader, epoch_batches),
        total=(opt.niter-start_ep)), start_ep):

        # stopping condition
        if epoch > opt.niter:
            break

        print(device)
        # run a train epoch of epoch_batches batches
        for step, z_batch in enumerate(tqdm(
            epoch_loader, total=epoch_batches), 1):
            z_batch = z_batch.to(device)
            netE.zero_grad()


            with torch.no_grad():
                fake_ws = G.mapping(z_batch, None)
                fake_im = G.synthesis(fake_ws) #Output Shape = batch_size (16) X 1 X 256 X 256 

            if has_masked_input:
                hints_fake, mask_fake = masking.mask_upsample(fake_im)
                mask_fake = mask_fake + 0.5 # trained in range [0, 1]
                encoded = netE(torch.cat([hints_fake, mask_fake], dim=1))
                regenerated = G(encoded, label)

            # compute loss
            # fake_wplus = torch.stack([fake_ws] * encoded.shape[1], dim=1)
            encoded_broadcasted = torch.stack([encoded] * fake_ws.shape[1], dim=1)
            loss_latent = mse_loss(encoded_broadcasted, fake_ws)
            loss_mse = mse_loss(regenerated, fake_im)
            loss_perceptual = perceptual_loss.forward(
                reshape(regenerated), reshape(fake_im)).mean()

            loss_id = torch.Tensor((0.,)).to(device)
            loss = (opt.lambda_latent * loss_latent
                    + opt.lambda_mse * loss_mse
                    + opt.lambda_lpips * loss_perceptual
                    + opt.lambda_id * loss_id)

            # optimize
            loss.backward()
            optimizerE.step()

            tqdm.write("Epoch %d step %d Losses z %0.4f mse %0.4f id %0.4f lpips %0.4f total %0.4f"
                           % (epoch, step, loss_latent.item(), loss_mse.item(),
                              loss_id.item(), loss_perceptual.item(), loss.item()))

        # do checkpointing
        if epoch % 50 == 0 or epoch == opt.niter:
            print('Saving Checkpoint')
            training_utils.make_checkpoint(
                netE, optimizerE, epoch,
                None,# test_metrics['loss_total'].avg.item(),
                '%s/netE_epoch_%d.pth' % (opt.outf, epoch))





class InterpolationLayer(torch.nn.Module):
    def __init__(self, size):
        super(InterpolationLayer, self).__init__()

        self.size=size

    def forward(self, x):
        return interpolate(x, size=self.size, mode='area')


if __name__ == '__main__':
    parser = training_utils.make_parser()
    opt = parser.parse_args()
    print(opt)

    opt.outf = opt.outf.format(**vars(opt))

    os.makedirs(opt.outf, exist_ok=True)
    # save options
    with open(os.path.join(opt.outf, 'optE.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)

    train(opt)
