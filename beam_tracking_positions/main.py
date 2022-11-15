import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
import requests

from models import GruModelGeneral, SimpleLSTM
from dataloader import GPSSet

sendProgressUpdate = True       # Enables/Disables notifications

paths = {
    'train': 'dataset',
    'test': 'dataset_test'
}

def sendNotification(text, silent=True):
    if sendProgressUpdate:
        token = 'xxxxxxxxxxx:your_token'
        url = f'https://api.telegram.org/bot{token}'
        params = {'chat_id':xxxxxxxxxxx, 'text':text, 'disable_notification':silent}
        r = requests.get(url + '/sendMessage', params=params)

def train_one_epoch(epoch):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(trainLoader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # h = net.initHidden(inputs.shape[0]).to(device)    # For GRU
        # outputs, _ = net(inputs, h)                       # For GRU
        outputs = net(inputs)                               # For SimpleLSTM

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('epoch {} batch {} loss: {}'.format(epoch, i + 1, last_loss))
            sendNotification('epoch {} batch {} loss: {}'.format(epoch, i + 1, last_loss))
            running_loss = 0.


def evaluate():
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    running_loss = 0.
    for i, vdata in enumerate(testLoader):
        inputs, labels = vdata
        inputs = inputs.to(device)
        labels = labels.to(device)
        # h = net.initHidden(inputs.shape[0]).to(device)    # For GRU
        # outputs, _ = net(inputs, h)                       # For GRU
        outputs = net(inputs)                               # For SimpleLSTM

        # Useful when batch size=1
        outputs = outputs.squeeze()
        labels = labels.squeeze()

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        _, top1 = outputs.topk(1)
        _, top2 = outputs.topk(2)
        _, top3 = outputs.topk(3)
        top1_correct += (top1 == labels).sum()
        top2_correct += (top2 == labels).sum()
        top3_correct += (top3 == labels).sum()

    top1_accuracy = top1_correct / i * 100 
    top2_accuracy = top2_correct / i * 100 
    top3_accuracy = top3_correct / i * 100
    running_loss /= i
    
    print(f'Validation: Top1={top1_accuracy}%, Top2={top2_accuracy}%, Top3={top3_accuracy}%, Loss={running_loss}')
    sendNotification(f'Validation: Top1={top1_accuracy}%, Top2={top2_accuracy}%, Top3={top3_accuracy}%, Loss={running_loss}')

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the dataset then split it into train and test
    dataset = GPSSet(path=paths['train'])    
    train_size = round(0.75 * len(dataset))
    test_size = round(0.25 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainLoader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)  
    testLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)   

    # # For GRU
    # N = 2 # size of each single input
    # N_BEAMS = 64
    # net = GruModelGeneral(in_features=N, num_classes=N_BEAMS, 
    #                            num_layers=1, hidden_size=64, embed_size=N_BEAMS, 
    #                            dropout=0.8).to(device)
    net = SimpleLSTM(2, 64, 32, 64).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(net.parameters(), lr=0.0001)

    sendNotification('----------------------------------------------------------------------------')
    sendNotification('Training Start')
    for epoch in range(50):
        net.train()
        train_one_epoch(epoch)

        net.eval()
        evaluate()
        
