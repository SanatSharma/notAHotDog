import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
import numpy as np
from PIL import Image
import os
import io

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
        # If greyscale, convert to RGB
        #if img.getbands()[0] == 'P' or img.getbands()[0] == 'L' or img.getbands()[-1] == "A":
            #print(img.getbands())
        #    img = img.convert('RGB')
        
        if img.getbands() != ('R','G','B'):
            img = img.convert('RGB')

        # Transform to make every image the same size for the neural net
        # Inception v3 needs a [3,299,299] image input
        img = self.transform(img)
        label = self.labels[idx]
        return (img, label)

    def create_dataset(self):
        labels = []
        # TODO: Slow to append every time. Should I just loop through and get the length in one go?
        for nsfw_dir in self.nsfw_dirs:
            files = [os.path.join(nsfw_dir, p) for p in sorted(os.listdir(nsfw_dir))]
            print(nsfw_dir, len(files))
            for file in files:
                self.indexer.get_index(file)
                labels.append(1)
        
        files = [os.path.join(self.neutral_dir, p) for p in sorted(os.listdir(self.neutral_dir))]
        print(self.neutral_dir, len(files))
        for file in files:
            self.indexer.get_index(file)
            labels.append(0)
        print("Dataset size:", len(labels))

        return np.array(labels)

class RuntimeDataset(Dataset):
    def __init__(self, blobs):
        self.transform = transforms.Compose(
                            [transforms.RandomResizedCrop(299), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
        self.blobs = blobs

    def __len__(self):
        return len(self.blobs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        blob = self.blobs[str(idx)].read()
        img = Image.open(io.BytesIO(blob))
        
        if img.getbands() != ('R','G','B'):
            img = img.convert('RGB')
        img = self.transform(img)
        return (img, 0)
        
def create_nsfw_dataset(nsfw_path, neutral_path, args, test_split=.2, shuffle=True):
    
    print("Creating nsfw dataset")
    dataset = NSFWDataset(nsfw_path, neutral_path)
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle :
        #np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    #print(len(train_sampler), len(valid_sampler))

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                            sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size,
                                                sampler=valid_sampler)

    return train_loader, test_loader
