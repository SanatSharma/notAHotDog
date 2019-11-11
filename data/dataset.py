import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
from PIL import Image
import os

class NSFWDataset(Dataset):
    def __init__(self, data_path_nsfw, data_path_neutral):
        self.transform = transforms.Compose(
                            [transforms.RandomResizedCrop(299), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
        self.nsfw_dirs = data_path_nsfw
        self.neutral_dir = data_path_neutral   
        self.indexer = Indexer()        
        self.labels = self.create_dataset()

    def __len__(self):
        return len(self.indexer)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = Image.open(self.indexer.get_object(idx))
        #img = self.transform(img)
        label = self.labels[idx]
        
        return (img, label)

    def create_dataset(self):
        labels = []
        # TODO: Slow to append every time. Should I just loop through and get the length in one go?
        for nsfw_dir in self.nsfw_dirs:
            files = [os.path.join(nsfw_dir, p) for p in sorted(os.listdir(nsfw_dir))]
            for file in files:
                self.indexer.get_index(file)
                labels.append(1)
        
        files = [os.path.join(self.neutral_dir, p) for p in sorted(os.listdir(self.neutral_dir))]
        for file in files:
            self.indexer.get_index(file)
            labels.append(0)

        return np.array(labels)

        


        
