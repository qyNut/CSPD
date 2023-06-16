import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
from utils.lossFunction import VGG16, NLayerDiscriminator, GANLoss, TVLoss
from utils.SSIM import SSIM
from models.csnet_2D import CSNet
import numpy as np

dir_img = Path('data/train/input/')
dir_mask = Path('data/train/output/')
dir_checkpoint = Path('models/weights/')

def train_net(net,
              device,
              epochs:int = 5,
              batch_size:int = 1,
              learning_rate: float = 0.001
              ):

    dataset = BasicDataset(dir_img, dir_mask)
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)


    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9,0.999), eps=1e-7, weight_decay=1e-8, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)  # goal: maximize Dice score

    criterion2 = SSIM(window_size = 3)
    lossLambda = 0.1
    global_step = 0
    
    ## perceptual loss
    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
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
        batch_size_tot = 0

        for batch in train_loader:
            images = batch['image']
            true_masks = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            
            # images = torch.sum(images,1,keepdim=True)
            images = images/torch.max(images)

            
            for i in range(true_masks.size(dim=0)):
                upthres = torch.quantile(true_masks[i,:],0.99)
                true_masks[i,:][true_masks[i,:] > upthres] = upthres
                true_masks[i,:] = torch.pow(true_masks[i,:], 1.5)


            masks_pred = net(images)
            lossCont = L1Loss(masks_pred,true_masks)
            
            loss = lossCont

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            batch_size_tot += images.shape[0]
            epoch_loss += (loss.item()*images.shape[0])
            epoch_lossCont += (lossCont.item()*images.shape[0])
            # epoch_lossVGG += (lossVgg.item()*batch_size)
            # epoch_lossTV += (tv_loss.item()*batch_size)
            
            # pbar.set_postfix(**{'loss (batch)': loss.item()})
            print(**{'Content loss (batch)': epoch_lossCont/batch_size_tot})
           
        scheduler.step(epoch_loss)
        lossContEpoch[epoch] = epoch_lossCont
        
        # val_score = evaluate(net, val_loader, device) 
        # valSSIMEpoch[epoch] = val_score['ssim']
        # valPSNREpoch[epoch] = val_score['psnr']
        # logging.info(f'Validation score: {val_score}')
        # lossVGGEpoch[epoch] = epoch_lossVGG
        # lossTVEpoch[epoch] = epoch_lossTV
        if (epoch+1)%10==0:

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            # logging.info(f'Checkpoint {epoch + 1} saved!      Validation Dice score: {val_score}')
    np.savez( str(dir_checkpoint / 'lossEpoch.npz'), lossContEpoch=lossContEpoch, lossVGGEpoch = lossVGGEpoch, 
                 lossTVEpoch = lossTVEpoch, valSSIMEpoch = valSSIMEpoch, valPSNREpoch = valPSNREpoch)

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    net = CSNet(n_channels=400, n_classes=1, bilinear=False)
            
    folder = 'models/pretrained/'
    model = folder + 'L1.pth'  
    net.load_state_dict(torch.load(model, map_location=device))
    net.to(device=device)
    
    train_net(net=net,epochs=50,batch_size=16,learning_rate=0.001,device=device)
    


