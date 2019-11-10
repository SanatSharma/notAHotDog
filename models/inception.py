import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class InceptionV3(nn.Module):
    def __init__(self, output_size=2):
        
        self.inception = models.inception_v3(pretrained=True).to(device)
        self.output_size = output_size
        self.fully_connected = nn.Linear(100, self.output_size)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fully_connected.weight)
    
    def forward(self, input):     
        intermediate = self.inception(input)
        labels = self.fully_connected(intermediate)
        probs = F.log_softmax(labels, dim=1)
        return probs

def train_network(train_data, dev_data, args):
    model = InceptionV3().to(device)
    optimizer = Adam(model.parameters(), args.lr)
    loss_function = nn.NLLLoss()
    model.train()
    batch_size = args.batch_size

    for i in range(args.epochs):
        print("Epoch:", i)
        epoch_loss = 0

        for idx, data in enumerate(train_data):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            probs = model.forward(inputs)
            loss = loss_function(probs.to(device), labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        print("Epoch loss", epoch_loss)


