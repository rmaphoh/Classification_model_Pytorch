import argparse
import logging
import os
import torch
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pycm import *
import matplotlib.pyplot as plt
from scripts.dataset import BasicDataset_outside
from scripts.model import arch_select
import scripts.paired_transforms_tv04 as p_tr
from torch.utils.data import DataLoader


def test_net(model_fl,
              csv_path,
              device,
              epochs=1,
              image_size=(512,512)
              ):

    storage_path ='../Results/M1/'
    n_classes = args.n_class
    if not os.path.isdir(storage_path):
        os.makedirs(storage_path)

    resize = p_tr.Resize(image_size)
    tensorizer = p_tr.ToTensor()
    val_transforms = p_tr.Compose([resize, tensorizer])

    test_dataset = BasicDataset_outside(csv_path, n_classes, transforms=val_transforms)
    n_test = len(test_dataset)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)


    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    filename_list = []

    for epoch in range(epochs):

        model_fl.eval()

        with tqdm(total=n_test, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in val_loader:
                imgs = batch['image']
                filename = batch['img_file']

                imgs = imgs.to(device=device, dtype=torch.float32)

                prediction = model_fl(imgs)
                prediction_softmax = nn.Softmax(dim=1)(prediction)
                _,prediction_decode = torch.max(prediction_softmax, 1)
                
                prediction_decode_list.append(prediction_decode.item())
                filename_list.extend(filename)
                prediction_list.extend(prediction_softmax.cpu().detach().numpy())

                pbar.update(imgs.shape[0])

    Data4stage2 = pd.DataFrame({'Name':filename_list, 'softmax_good':np.array(prediction_list)[:,0], \
        'softmax_usable':np.array(prediction_list)[:,1], 'softmax_bad':np.array(prediction_list)[:,2], 'Prediction': prediction_decode_list})
    Data4stage2.to_csv(storage_path+ '/results_.csv', index = None, encoding='utf8')

    if not os.path.exists('../Results/M1/Good_quality/'):
        os.makedirs('../Results/M1/Good_quality/')
    if not os.path.exists('../Results/M1/Bad_quality/'):
        os.makedirs('../Results/M1/Bad_quality/')

    Eyepacs_pre = Data4stage2['Prediction']
    name_list = Data4stage2['Name']

    Eye_good = 0
    Eye_bad = 0

    for i in range(len(name_list)):
        
        if Eyepacs_pre[i]!=2:
            Eye_good+=1
            shutil.copy('../Results/M0/images/' + name_list[i], '../Results/M1/Good_quality/')
        else:
            Eye_bad+=1        
            shutil.copy('../Results/M0/images/' + name_list[i], '../Results/M1/Bad_quality/')


    print('Good cases by EyePACS_QA is {} '.format(Eye_good))
    print('Bad cases by EyePACS_QA is {} '.format(Eye_bad))





def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=6,
                        help='Batch size', dest='batchsize')
    parser.add_argument( '-dir', '--test_csv_dir', metavar='tcd', type=str,
                        help='path to the csv', dest='test_dir')
    parser.add_argument( '-n', '--n_classes', dest='n_class', type=int, default=False,
                        help='number of class')
    parser.add_argument( '-m', '--model', dest='model_structure', type=str, 
                        help='Backbone of the model')     
    parser.add_argument( '-s', '--image-size', dest='image_size', type=int, 
                        help='size of image')   


    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    csv_path = args.test_dir
    img_size= (args.image_size,args.image_size)

    model_fl = arch_select(model_structure = args.model_structure, pretrained=True, n_classes=args.n_class)

    checkpoint_path = './checkpoint_/{}/best_checkpoint.pth'.format(args.model_structure) 
    
    model_fl.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    logging.info(f'Model loaded from {checkpoint_path}')

    model_fl.to(device=device)


    test_net(model_fl,
                csv_path,
                device=device,
                epochs=args.epochs,
                image_size=img_size)


