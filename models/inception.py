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
        super(InceptionV3,self).__init__()
        self.inception = models.inception_v3(pretrained=True)
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

def train_network(train_data, args):
    model = InceptionV3().to(device)
    optimizer = Adam(model.parameters(), args.lr)
    loss_function = nn.NLLLoss()
    model.train()
    batch_size = args.batch_size

    for i in range(args.epochs):
        print("Epoch:", i)
        epoch_loss = 0

        for idx, data in tqdm(enumerate(train_data)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            print(inputs.shape, labels.shape)
            optimizer.zero_grad()

            probs = model.forward(inputs)
            loss = loss_function(probs.to(device), labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        print("Epoch loss", epoch_loss)

    # Return trained model
    return Trained_Model(model)

class Trained_Model:
    def __init__(self, model):
        self.model = model
    
    def evaluate(test_data):
        correct, nsfw_correct, false_positive, false_negative, total = 0,0,0,0, len(test_data)

        self.model.eval()
        for idx, data in tqdm(enumerate(test_data)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            probs = self.model.forward(inputs)
            
            for i in range(len(probs)):
                if probs[i] == labels[i]:
                    if probs[i] == 1:
                        nsfw_correct += 1
                    correct +=1
                elif labels[i] == 1:
                    false_negative+=1
                else:
                    false_positive+=1

        print("Correctness", str(correct) + "/" + str(total) + ": " + str(round(correct/total, 5)))
        print("Precision nsfw", str(nsfw_correct) + "/" + str(false_positive + nsfw_correct) + ": " + str(round(nsfw_correct/(false_positive + nsfw_correct), 5)))
        print("Recall nsfw", str(nsfw_correct) + "/" + str(nsfw_correct + false_negative) + ": " + str(round(nsfw_correct/nsfw_correct + false_negative, 5)))
