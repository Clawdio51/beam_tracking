from tkinter import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import requests

from seq2seq_ConvLSTM import EncoderDecoderConvLSTM
from dataloader import ImageSet, ImageSet_Augmented

sendProgressUpdate = False       # Enables/Disables notifications

paths = {
    'train': 'dataset',
    'test': 'dataset_test'
}

def sendNotification(text, silent=True):
    if sendProgressUpdate:
        token = 'xxxxxxxxxxx:your_token'
        url = f'https://api.telegram.org/bot{token}'
        params = {'chat_id':1388173517, 'text':text, 'disable_notification':silent}
        r = requests.get(url + '/sendMessage', params=params)

def train_one_epoch(epoch):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(trainLoader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # HARDCODED Add AWGN noise to images
        # inputs = inputs + 0.1*torch.randn_like(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # outputs = net(inputs[:, :-1])                 # For simple LSTM
        outputs = net(inputs, future_seq=1)             # For convLSTM

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
            # tb_x = epoch_index * len(trainLoader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
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
        outputs = net(inputs, future_seq=1)

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

    image_resolution = 144          # 144p, 240p, 480p, ... If image_resolution=x, image size is x*(x/1.76)
    
    transform = transforms.Compose([
        transforms.Resize((int(image_resolution/1.76), image_resolution)),
        #transforms.CenterCrop(224),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5], std=[0.2])
    ])
    # For data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((int(image_resolution/1.76), image_resolution)),
        #transforms.CenterCrop(224),
        transforms.Grayscale(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5], std=[0.2])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((int(image_resolution/1.76), image_resolution)),
        #transforms.CenterCrop(224),
        transforms.Grayscale(),
        # transforms.RandAugment(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5], std=[0.2])
    ])

    # Load the dataset then split it into train and test
    dataset = ImageSet(path=paths['train'], transform=transform)     # Without data augmentation
    # dataset = ImageSet_Augmented(train_transform=train_transform, val_transform=val_transform)    # With data augmentation
    train_size = round(0.75 * len(dataset))
    test_size = round(0.25 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # train_dataset = ImageSet(path=paths['train'], transform=transform)
    # test_dataset = ImageSet(path=paths['test'], transform=transform)
    trainLoader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)  # Maybe return to num_workers=2
    testLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)   # Maybe return to num_workers=0 or 2

    net = EncoderDecoderConvLSTM(nf=32, in_chan=1, classes=64, image_resolution=image_resolution).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(net.parameters(), lr=0.0001)

    sendNotification('----------------------------------------------------------------------------')
    sendNotification('Training Start')
    for epoch in range(50):
        # For data augmentation
        # set_is_train works only on ImageSet_Augmented, where it specifies 
        # whether we should apply data augmentation or not.
        # In the case of ImageSet with no data augmentation, this function
        # does nothing
        #dataset.set_is_train(True)
        net.train()
        train_one_epoch(epoch)

        # For data augmentation
        #dataset.set_is_train(False)
        net.eval()
        evaluate()
        
