See lucy chai's environment.yml. Need additional following for pDL environment - 
  - pip:
     - oyaml
     - lpips
     - prdc
     - ninja
     - ipywidgets
     - gdown





Local steps:

conda activate pDL-clone

cd ~/appdir/Github/StyleGANs/stylegan2-ada-pytorch

#GreatestHits
python -m training.train_sgan_encoder --ckpt_path /home/purnima/appdir/Github/StyleGANs/stylegan2-ada-pytorch/training-runs/00041-vis-data-256-split-auto1-noaug/network-snapshot-002200.pkl --outf checkpoints/sgan_encoder_greatesthits_resnet-34_RGBM --batchSize 16 --netE_type resnet-34 --lr 0.00001 --niter 1500 --lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 --lambda_id 1.0

python -m training.train_sgan_encoder --ckpt_path /home/purnima/appdir/Github/StyleGANs/audio-stylegan2/training-runs/vis-data-256-split/00000/network-snapshot-002800.pkl --outf checkpoints/sgan_encoder_greatesthits_resnet-34_RGBM-corrected --batchSize 16 --netE_type resnet-34 --lr 0.00001 --niter 1500 --lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 --lambda_id 1.0

python -m training.train_sgan_encoder --ckpt_path /home/purnima/appdir/Github/StyleGANs/audio-stylegan2/training-runs/vis-data-256-split/00000/network-snapshot-002800.pkl --outf checkpoints/sgan_encoder_greatesthits_resnet-34_RGBM-corrected --batchSize 16 --netE_type resnet-34 --lr 0.0001 --niter 1500 --lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 --lambda_id 1.0

python -m training.train_sgan_encoder --ckpt_path /home/purnima/appdir/Github/StyleGANs/audio-stylegan2/training-runs/vis-data-256-split/00000/network-snapshot-002800.pkl --outf checkpoints/sgan_encoder_greatesthits_resnet-34_RGBM-corrected-withthresholding --batchSize 16 --netE_type resnet-34 --lr 0.0001 --niter 1500 --lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 --lambda_id 1.0 --thresholding 25


python -m training.train_sgan_encoder --ckpt_path /home/purnima/appdir/Github/StyleGANs/audio-stylegan2/training-runs/vis-data-256-split/00000/network-snapshot-002800.pkl --outf checkpoints/sgan_encoder_greatesthits_resnet-34_RGBM-corrected-nolpips --batchSize 16 --netE_type resnet-34 --lr 0.0001 --niter 1500 --lambda_mse 1.0 --lambda_lpips 0.0 --lambda_latent 1.0

--restart training
python -m training.train_sgan_encoder --ckpt_path /home/purnima/appdir/Github/StyleGANs/audio-stylegan2/training-runs/vis-data-256-split/00000/network-snapshot-002800.pkl --outf checkpoints/sgan_encoder_greatesthits_resnet-34_RGBM --netE checkpoints/sgan_encoder_greatesthits_resnet-34_RGBM/netE_epoch_700.pth --batchSize 16 --netE_type resnet-34 --lr 0.00001 --niter 1500 --lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 --lambda_id 1.0




#Water Only
python -m training.train_sgan_encoder --ckpt_path /home/purnima/appdir/Github/StyleGANs/stylegan2-ada-pytorch/training-runs/00044-water-data-split-auto1-noaug/network-snapshot-001000.pkl --outf checkpoints/sgan_encoder_water_resnet-34_RGBM --batchSize 8 --netE_type resnet-34 --lr 0.0001 --niter 1500 --lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 --lambda_id 1.0

python -m training.train_sgan_encoder --ckpt_path /home/purnima/appdir/Github/StyleGANs/stylegan2-ada-pytorch/training-runs/00044-water-data-split-auto1-noaug/network-snapshot-001000.pkl --outf checkpoints/sgan_encoder_water_resnet-34_RGBM-withthresholding --batchSize 8 --netE_type resnet-34 --lr 0.0001 --niter 1500 --lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 --lambda_id 1.0 --thresholding 17


#WaterWind
python -m training.train_sgan_encoder --ckpt_path /home/purnima/appdir/Github/StyleGANs/stylegan2-ada-pytorch/training-runs/00043-audio-auto1-noaug/network-snapshot-000200.pkl --outf checkpoints/sgan_encoder_waterwind_resnet-34_RGBM --batchSize 16 --netE_type resnet-34 --lr 0.00001 --niter 1500 --lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 --lambda_id 1.0