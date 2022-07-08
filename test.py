import argparse
import logging
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pycm import *
import matplotlib.pyplot as plt
from scripts.dataset import BasicDataset
from scripts.model import arch_select
import scripts.paired_transforms_tv04 as p_tr
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def test_net(model_fl,
              csv_path,
              device,
              epochs=1,
              image_size=(512,512)
              ):

    storage_path ="./Results/test_on_{}/".format(args.model_structure)
    n_classes = args.n_class
    if not os.path.isdir(storage_path):
        os.makedirs(storage_path)

    resize = p_tr.Resize(image_size)
    tensorizer = p_tr.ToTensor()
    val_transforms = p_tr.Compose([resize, tensorizer])

    test_dataset = BasicDataset(csv_path, n_classes, transforms=val_transforms)
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
                true_label = batch['label']
                filename = batch['img_file']

                imgs = imgs.to(device=device, dtype=torch.float32)
                label_type = torch.float32 if n_classes == 1 else torch.long
                true_label = true_label.to(device=device, dtype=label_type)

                prediction = model_fl(imgs)
                prediction_softmax = nn.Softmax(dim=1)(prediction)
                _,prediction_decode = torch.max(prediction_softmax, 1)
                _,true_label_decode = torch.max(true_label, 1)
                
                prediction_decode_list.append(prediction_decode.item())
                true_label_decode_list.append(true_label_decode.item())
                filename_list.extend(filename)
                prediction_list.extend(prediction_softmax.cpu().detach().numpy())

                pbar.update(imgs.shape[0])

        accuracy = accuracy_score(true_label_decode_list, prediction_decode_list)
        auc_roc = roc_auc_score(true_label_decode_list, np.array(prediction_list),multi_class='ovr')
        F1 = f1_score(true_label_decode_list, prediction_decode_list,average='macro')

        print('Sklearn Testing Metrics - Acc: {:.4f} AUC-roc: {:.4f} F1-score: {:.4f}'.format(accuracy, auc_roc, F1)) 

    cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
    cm.plot(cmap=plt.cm.Blues,number_label=True,plot_lib="matplotlib")
    plt.savefig(storage_path+ '/confusion_matrix_.jpg',dpi=300,bbox_inches ='tight')

    Data4stage2 = pd.DataFrame({'Name':filename_list, 'Label':true_label_decode_list, 'softmax_good':np.array(prediction_list)[:,0], \
        'softmax_usable':np.array(prediction_list)[:,1], 'softmax_bad':np.array(prediction_list)[:,2], 'Prediction': prediction_decode_list})
    Data4stage2.to_csv(storage_path+ '/results_.csv', index = None, encoding='utf8')
    




def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=240,
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


