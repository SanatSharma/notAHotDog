import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class InceptionV3(nn.Module):
    def __init__(self, output_size=2):
        self.transform = transforms.Compose(
                            [transforms.RandomResizedCrop(299), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
        
        self.inception = models.inception_v3(pretrained=True).to(device)
        self.output_size = output_size
        self.fully_connected = nn.Linear(100, self.output_size)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fully_connected.weight)
    
    def forward(self, input):
        input = self.transform(input)
        intermediate = self.inception(input)
        labels = self.fully_connected(intermediate)
        probs = F.log_softmax(labels, dim=1)
        return probs

def train_network(train_data, dev_data, args):
    pass



