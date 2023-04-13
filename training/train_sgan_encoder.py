import os
import pickle
import oyaml as yaml
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import Namespace

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
    outdim = opt.w_dim
    nz = G.z_dim 
    label = torch.zeros([1, G.z_dim], device=device)

    depth = int(opt.netE_type.split('-')[-1])
    has_masked_input = True 

    netE = stylegan_encoder.load_stylegan_encoder(domain=None, nz=nz,
                                                   outdim=outdim,
                                                   use_RGBM=True, #opt.masked,
                                                   use_VAE=False, #opt.vae_like,
                                                   resnet_depth=depth,
                                                   ckpt_path=None)
    netE = netE.to(device).train()
    summary(netE, input_size=(2, opt.n_frames, opt.n_frames))# Prints summary/model architecture on command line

    # losses + optimizers
    mse_loss = nn.MSELoss()
    perceptual_loss = losses.LPIPS_Loss(net='vgg', use_gpu=has_cuda)
    util.set_requires_grad(False, perceptual_loss)
    
    reshape = training_utils.make_ipol_layer(opt.n_frames)
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
    all_losses = dict(w=[], mse=[], lpips=[], loss_correlation=[], totalloss=[])
    for epoch, epoch_loader in enumerate(tqdm(
        training_utils.epoch_grouper(train_loader, epoch_batches),
        total=(opt.niter-start_ep)), start_ep):

        # stopping condition
        if epoch > opt.niter:
            break

        # run a train epoch of epoch_batches batches
        for step, z_batch in enumerate(tqdm(
            epoch_loader, total=epoch_batches), 1):
            z_batch = z_batch.to(device)
            netE.zero_grad()

            with torch.no_grad():
                fake_ws = G.mapping(z_batch, None)
                fake_im = G.synthesis(fake_ws) #Output Shape = batch_size (16) X 1 X 256 X 256 
            
            if opt.use_thresholding:
                re_fake_im = util.threshold_spectrogram(fake_im, opt.thresholding)
            else:
                re_fake_im = fake_im

            hints_fake, mask_fake = masking.mask_upsample(re_fake_im)
            mask_fake = mask_fake + 0.5 # trained in range [0, 1]
            encoded = netE(torch.cat([hints_fake, mask_fake], dim=1))
            encoded_broadcasted = torch.stack([encoded] * fake_ws.shape[1], dim=1)
            regenerated = G.synthesis(encoded_broadcasted)

            # compute loss
            loss_latent = mse_loss(encoded_broadcasted, fake_ws)
            loss_mse = mse_loss(regenerated, fake_im)
            loss_perceptual = perceptual_loss.forward(
                reshape(regenerated), reshape(fake_im)).mean()

            loss = (opt.lambda_latent * loss_latent
                    + opt.lambda_mse * loss_mse
                    + opt.lambda_lpips * loss_perceptual)

            # optimize
            loss.backward()
            optimizerE.step()

            all_losses['w'].append(loss_latent.item())
            all_losses['mse'].append(loss_mse.item())
            all_losses['lpips'].append(loss_perceptual.item())
            all_losses['totalloss'].append(loss.item())

            tqdm.write("Epoch %d step %d Losses w %0.4f mse %0.4f lpips %0.4f total %0.4f"
                           % (epoch, step, loss_latent.item(), loss_mse.item(),
                              loss_perceptual.item(), 
                              loss.item()))

        # updated to run a small set of test zs 
        # rather than a single fixed batch
        netE.eval()
        test_metrics = {
            'loss_latent': util.AverageMeter('loss_latent'),
            'loss_mse': util.AverageMeter('loss_mse'),
            'loss_perceptual': util.AverageMeter('loss_perceptual'),
            'loss_total': util.AverageMeter('loss_total'),
        }
        for step, test_zs in enumerate(tqdm(test_loader), 1):
            with torch.no_grad():
                test_zs = test_zs.to(device)
                fake_ws = G.mapping(test_zs, None)
                fake_im = G.synthesis(fake_ws) #Output Shape = batch_size (16) X 1 X 256 X 256 
                
                if opt.use_thresholding:
                    re_fake_im = util.threshold_spectrogram(fake_im, opt.thresholding)
                else:
                    re_fake_im = fake_im

                hints_fake, mask_fake = masking.mask_upsample(re_fake_im)
                mask_fake = mask_fake + 0.5 # trained in range [0, 1]
                encoded = netE(torch.cat([hints_fake, mask_fake], dim=1))
                encoded_broadcasted = torch.stack([encoded] * fake_ws.shape[1], dim=1)
                regenerated = G.synthesis(encoded_broadcasted)

                # compute loss
                loss_latent = mse_loss(encoded_broadcasted, fake_ws)
                loss_mse = mse_loss(regenerated, fake_im)
                loss_perceptual = perceptual_loss.forward(
                    reshape(regenerated), reshape(fake_im)).mean()

                loss = (opt.lambda_latent * loss_latent
                        + opt.lambda_mse * loss_mse
                        + opt.lambda_lpips * loss_perceptual)

                tqdm.write("Validation Epoch %d step %d Losses w %0.4f mse %0.4f lpips %0.4f total %0.4f"
                           % (epoch, step, loss_latent.item(), loss_mse.item(),
                              loss_perceptual.item(),  
                              loss.item()))

            # update running avg
            test_metrics['loss_latent'].update(loss_latent)
            test_metrics['loss_mse'].update(loss_mse)
            test_metrics['loss_perceptual'].update(loss_perceptual)
            test_metrics['loss_total'].update(loss)


        # do checkpointing
        if epoch % 50 == 0 or epoch == opt.niter:
            print('Saving Checkpoint')
            training_utils.make_checkpoint(
                netE, optimizerE, epoch,
                test_metrics['loss_total'].avg.item(),
                '%s/netE_epoch_%d.pth' % (opt.outf, epoch))

            # At some point - integrate Tensorboard.
            fig, ax = plt.subplots(1,5, figsize=(20, 3))
            ax[0].plot(all_losses['w'])
            ax[0].set_title('W loss')
            ax[1].plot(all_losses['mse'])
            ax[1].set_title('MSE loss')
            ax[2].plot(all_losses['lpips'])
            ax[2].set_title('LPIPS loss')
            ax[4].plot(all_losses['totalloss'])
            ax[4].set_title('Total loss')
            fig.savefig('%s/netE_epoch_%d.png' % (opt.outf, epoch))

        if test_metrics['loss_total'].avg.item() < best_val_loss:
            # modified to save based on test zs loss rather than
            # final model at the end
            print("Best Validation Loss: %.2f Checkpointing at epoch %d" % (test_metrics['loss_total'].avg.item(), epoch))
            training_utils.make_checkpoint(
                netE, optimizerE, epoch,
                test_metrics['loss_total'].avg.item(),
                '%s/netE_epoch_best.pth' % (opt.outf))
            best_val_loss = test_metrics['loss_total'].avg.item()


if __name__ == '__main__':
    
    config = util.get_config('config/config.json')
    config = Namespace(**dict(**config))
    
    os.makedirs(config.outf, exist_ok=True)
    # save options
    with open(os.path.join(config.outf, 'optE.yml'), 'w') as f:
        yaml.dump(vars(config), f, default_flow_style=False)

    train(config)
