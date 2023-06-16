import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading_256Frames import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.lossFunction import VGG16, NLayerDiscriminator, GANLoss, TVLoss
from utils.SSIM import SSIM
from evaluate import evaluate
from unet.unet_model_2D import UNet
from PIL import Image
import numpy as np
import scipy.io as io
import cv2
import matplotlib.pyplot as plt


dir_img = Path('G:/TrainingDataFormat/')
# dir_img = Path('G:/TrainingDataRandNorm/')
dir_mask = Path('G:/TrainingMaskFormat/')
dir_checkpoint = Path('G:/checkPoints/200FramesAccu_AD_intensity_L1/')

dir_img_val = Path('G:/ValDataFormat/')
dir_mask_val = Path('G:/TrainingMaskFormat/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
    

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    val_set = CarvanaDataset(dir_img_val, dir_mask_val)
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=12, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', entity="qiyou")
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9,0.999), eps=1e-7, weight_decay=1e-8, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)


    criterion2 = SSIM(window_size = 3)
    lossLambda = 0.1
    global_step = 0
    
    ## perceptual loss
    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    tvLoss = TVLoss()
    # vgg = VGG16(requires_grad=False).to(device)

    # images = torch.zeros([16,400,291,121])
    # true_masks = torch.zeros([16,1,1451,1201])
    lossContEpoch = np.zeros((epochs))
    lossVGGEpoch = np.zeros((epochs))
    lossTVEpoch = np.zeros((epochs))
    valSSIMEpoch = np.zeros((epochs))
    valPSNREpoch = np.zeros((epochs))
    torch.cuda.empty_cache()
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        epoch_lossCont = 0
        epoch_lossVGG = 0
        epoch_lossTV = 0
        batch_size_tot = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # for i in range(2):
                #     images[i*8:(i+1)*8,:] = torch.squeeze(batch['image'])
                #     true_masks[i*8:(i+1)*8,:] = torch.squeeze(batch['mask'],dim = 0)
                images = batch['image'][:,0:200,:,:]
                true_masks = batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                    
                # images = F.interpolate(images,(256,128),mode='bilinear')
                # for i in range(images.size(dim=0)):
                #     images[i,:] = images[i,:]/torch.max(images[i,:]) 
                images = torch.sum(images,1,keepdim=True)
                images = images/torch.max(images)

                
                for i in range(true_masks.size(dim=0)):
                    upthres = torch.quantile(true_masks[i,:],0.99)
                    true_masks[i,:][true_masks[i,:] > upthres] = upthres
                    true_masks[i,:] = torch.pow(true_masks[i,:], 1.5)
                # true_masks = F.interpolate(true_masks,(1024,1024),mode='bilinear')
                
                # plt.figure()
                # plt.imshow(torch.sum(images[0,:],0).cpu().numpy())    
                # plt.figure()
                # plt.imshow(torch.sum(true_masks[0,:],0).cpu().numpy())   
                              
                # print(images.shape)
                # print(true_masks.shape)
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    lossCont = L1Loss(masks_pred,true_masks)
                    
                    # lossCont = 1 * (
                    #     MSELoss(vgg(masks_pred).relu1_2, vgg(true_masks).relu1_2)
                    #     + MSELoss(vgg(masks_pred).relu2_2, vgg(true_masks).relu2_2)
                    # )                
                    
                    # lossVgg = 0.001 * (
                    #     MSELoss(vgg(masks_pred).relu1_2, vgg(true_masks).relu1_2)
                    #     + MSELoss(vgg(masks_pred).relu2_2, vgg(true_masks).relu2_2)
                    # )

                    # tv_loss = 0.001 * tvLoss(masks_pred)
                    loss = lossCont
                    # loss = lossCont + lossVgg

                    # loss = dice_loss(masks_pred,true_masks)
                    # loss2 = L1Loss(masks_pred[true_masks==0],0*masks_pred[true_masks==0])
                    # loss = loss1+loss2

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                batch_size_tot += images.shape[0]
                epoch_loss += (loss.item()*images.shape[0])
                epoch_lossCont += (lossCont.item()*images.shape[0])
                # epoch_lossVGG += (lossVgg.item()*batch_size)
                # epoch_lossTV += (tv_loss.item()*batch_size)
                
                # pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.set_postfix(**{'Content loss (batch)': epoch_lossCont/batch_size_tot})
                                    # 'Perceptual loss (batch)': epoch_lossVGG/batch_size_tot})
                                    # 'TV loss (batch)': epoch_lossTV/batch_size_tot})
           
        scheduler.step(epoch_loss)
        lossContEpoch[epoch] = epoch_lossCont
        
        # val_score = evaluate(net, val_loader, device) 
        # valSSIMEpoch[epoch] = val_score['ssim']
        # valPSNREpoch[epoch] = val_score['psnr']
        # logging.info(f'Validation score: {val_score}')
        # lossVGGEpoch[epoch] = epoch_lossVGG
        # lossTVEpoch[epoch] = epoch_lossTV
        if save_checkpoint and (epoch+1)%10==0:

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            # logging.info(f'Checkpoint {epoch + 1} saved!      Validation Dice score: {val_score}')
    np.savez( str(dir_checkpoint / 'lossEpoch.npz'), lossContEpoch=lossContEpoch, lossVGGEpoch = lossVGGEpoch, 
                 lossTVEpoch = lossTVEpoch, valSSIMEpoch = valSSIMEpoch, valPSNREpoch = valPSNREpoch)
    plt.figure(1)
    plt.plot(lossContEpoch)  
    plt.title('Content Loss')
    plt.figure(2)
    plt.plot(valSSIMEpoch) 
    plt.title('SSIM Validation')
    plt.figure(3)
    plt.plot(valPSNREpoch) 
    plt.title('PSNR Validation')
    # plt.figure(2)
    # plt.plot(lossVGGEpoch) 
    # plt.figure(3)
    # plt.plot(lossTVEpoch) 
        
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.ConvTranspose2d):
      nn.init.kaiming_uniform_(m.weight.data)
  elif isinstance(m, nn.Conv3d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    net.apply(initialize_weights)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    try:
        train_net(net=net,
                  epochs=50,
                  batch_size=16,
                  learning_rate=0.001,
                  device=device,
                  img_scale=args.scale,
                  val_percent=0,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
