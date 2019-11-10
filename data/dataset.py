import torch
from torchvision import transforms, utils
from torch.utils.data import DataSet, DataLoader

class NSFWDataset(DataSet):
    def __init__(self, data_path_nsfw, data_path_neutral):
        self.transform = transforms.Compose(
                            [transforms.RandomResizedCrop(299), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
        self.nsfw_dirs = data_path_nsfw
        self.neutral_dir = data_path_neutral        

    def __len__(self):
        return 
