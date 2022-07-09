import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, csv_path, n_classes, transforms):
        'Initialization'
        self.csv_path = csv_path
        self.n_classes = n_classes
        df = pd.read_csv(self.csv_path)
        self.list_IDs = df.values.tolist()
        self.transform = transforms

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        
        img_name, label, _ = self.list_IDs[index]
        image = Image.open(img_name)
        
        if self.transform is not None:
            image_processed = self.transform(image)
 
        return {
            'img_file': img_name,
            'image': image_processed.type(torch.FloatTensor),
            'label': torch.from_numpy(np.eye(self.n_classes, dtype='uint8')[label]).type(torch.long)
        }



class BasicDataset_outside(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, csv_path, n_classes, transforms):
        'Initialization'
        self.csv_path = csv_path
        self.n_classes = n_classes
        df = pd.read_csv(self.csv_path)
        self.list_IDs = df.values.tolist()
        self.transform = transforms

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        
        img_name, _, _ = self.list_IDs[index]
        image = Image.open('../Results/M0/images/' + img_name)
        
        if self.transform is not None:
            image_processed = self.transform(image)
 
        return {
            'img_file': img_name,
            'image': image_processed.type(torch.FloatTensor),
        }

