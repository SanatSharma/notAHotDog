import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class InceptionV3(nn.Module):
    def __init__(self, output_size=2):
        super(InceptionV3,self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.output_size = output_size
        self.fully_connected = nn.Linear(1000, self.output_size)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fully_connected.weight)
    
    def forward(self, input, train=True): 
        if train:
            intermediate, _ = self.inception(input)
        else:
            intermediate = self.inception(input)
        labels = self.fully_connected(intermediate)
        probs = F.log_softmax(labels, dim=1)
        return probs

def train_network(train_data, args, model=None):
    if model:
        model = model
    else:
        model = InceptionV3().to(device)
    optimizer = Adam(model.parameters(), args.lr)
    loss_function = nn.NLLLoss()
    model.train()
    batch_size = args.batch_size

    for i in range(args.epochs):
        print("Epoch:", i)
        epoch_loss = 0

        for idx, data in tqdm(enumerate(train_data), total=len(train_data)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape, labels.shape)
            optimizer.zero_grad()

            probs = model.forward(inputs)
            #print(probs)
            loss = loss_function(probs.to(device), labels.type(torch.LongTensor).to(device))
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        print("Epoch loss", epoch_loss)

    # Return trained model
    return Trained_Model(model)

# Overall PRecision: .93
# NSFW Precision: .90
# NSFW Recall: .96

class Trained_Model:
    def __init__(self, model):
        self.model = model.to(device)
    
    def evaluate(self, test_data):
        correct, nsfw_correct, false_positive, false_negative, total = 0,0,0,0,0

        self.model.eval()
        for idx, data in tqdm(enumerate(test_data), total=len(test_data)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device)

            probs = self.model.forward(inputs, train=False)
            print(np.exp(probs.detach().numpy()))
            
            for i in range(len(probs)):
                val = torch.argmax(probs[i]).item()
                if val == labels[i].item():
                    if val == 1:
                        nsfw_correct += 1
                    correct +=1
                elif labels[i].item() == 1:
                    false_negative+=1
                else:
                    false_positive+=1
                total+=1

        print("Correctness", str(correct) + "/" + str(total) + ": " + str(round(correct/total, 5)))
        print("Precision nsfw", str(nsfw_correct) + "/" + str(false_positive + nsfw_correct) + ": " + str(round(nsfw_correct/(false_positive + nsfw_correct), 5)))
        print("Recall nsfw", str(nsfw_correct) + "/" + str(nsfw_correct + false_negative) + ": " + str(round(nsfw_correct/nsfw_correct + false_negative, 5)))

    def runtime_api(self, data):
        self.model.eval()
        for idx, d in enumerate(data):
            inputs, labels = d
            inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device)

            probs = self.model.forward(inputs, train=False)
            probs = np.exp(probs.detach().numpy())
            return probs