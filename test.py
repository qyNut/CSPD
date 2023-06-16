
import numpy as np
import torch
import torch.nn.functional as F
from models.csnet_2D import CSNet
import scipy.io as io
from PIL import Image
from os import listdir
from os.path import splitext
from pathlib import Path

if __name__ == '__main__':

    net = CSNet(n_channels=400, n_classes=1, bilinear=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    folder = 'models/'
    model = folder + 'L1.pth'  

    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))
    input_dir = Path('data/input/')
    output_dir = 'data/output/'

    depth = 1024
    width = 1024
    mask1 = np.zeros((1,1,depth,width))
    
    ids = [splitext(file)[0] for file in listdir(input_dir) if not file.startswith('.')]
    for name in ids:

        print(name)
        img_file = list(input_dir.glob(name + '.*'))
        img = io.loadmat(img_file[0])['IQData']
        
        img = torch.from_numpy(img).to(device=device, dtype=torch.float32)
        for i in range(img.shape[0]):
            img[i,:,:] = img[i,:,:]/torch.max(img[i,:,:])
 
        img = img[np.newaxis,...]
        img = F.interpolate(img,(256,128),mode='bilinear')

        with torch.no_grad():
            mask1 = mask1 + net(img).cpu().numpy()   

    mask1 = np.squeeze(mask1)  
    mdic = {"mask1": mask1}
    io.savemat(output_dir+'SR.mat', mdic)
    
    
    mask1 = mask1**0.4  
    mask1 = mask1/np.max(mask1)*255   
    mask1 = mask1.astype(np.uint8)
    im = Image.fromarray(mask1)
    im.save(output_dir+'SR.tif')
    

    













 