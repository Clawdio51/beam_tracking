import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import os

from PIL import Image

class ImageSet(Dataset):
    def __init__(self, path='./dataset', transform=None):
        self.df = pd.read_csv(
            os.path.join(path, 'ml_challenge_dev_multi_modal.csv'), 
            usecols=['unit1_rgb_1', 'unit1_rgb_2', 'unit1_rgb_3', 'unit1_rgb_4', 'unit1_rgb_5', 'unit1_beam']
        )
        
        self.transform = transform
        self.path = path    # path argument is needed since data is saved as relative path

    # HARDCODED setter function for is_train. Does nothing in this class
    def set_is_train(self, is_train):
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.loc[idx]
        image_paths = item.drop('unit1_beam')
        label = item.get('unit1_beam')

        images = None
        for key in image_paths.keys():
            path = os.path.join(self.path, image_paths.get(key))
            image = Image.open(path)

            if self.transform:
                image = self.transform(image)
            # image = torch.squeeze(image)    # In case we apply BGR2GRAY transform

            image = torch.unsqueeze(image, dim=0)   # Used for stacking images
            if images == None:
                images = image
            else:
                images = torch.cat((images, image), dim=0)

        label = torch.tensor(label)

        label -= 1      # Beam indices start from 1 instead of 0

        return images, label

class ImageSet_Augmented(Dataset):
    def __init__(self, path, train_transform=None, val_transform=None):
        self.df = pd.read_csv(
            os.path.join(path, 'ml_challenge_dev_multi_modal.csv'), 
            usecols=['unit1_rgb_1', 'unit1_rgb_2', 'unit1_rgb_3', 'unit1_rgb_4', 'unit1_rgb_5', 'unit1_beam']
        )
        
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.path = path
        self.is_train = None

    def set_is_train(self, is_train):
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.loc[idx]
        image_paths = item.drop('unit1_beam')
        label = item.get('unit1_beam')

        # HARDCODED solution for data augmentation on training set only
        transform = self.train_transform if self.is_train else self.val_transform

        images = None
        for key in image_paths.keys():
            path = os.path.join(self.path, image_paths.get(key))
            image = Image.open(path)

            if transform:
                image = transform(image)
            # image = torch.squeeze(image)    # In case we apply BGR2GRAY transform

            image = torch.unsqueeze(image, dim=0)   # Used for stacking images
            if images == None:
                images = image
            else:
                images = torch.cat((images, image), dim=0)

        label = torch.tensor(label)

        label -= 1      # Beam indices start from 1 instead of 0

        return images, label

if __name__ == '__main__':
    dataset = ImageSet(transform=transforms.ToTensor())
    print(dataset[0])
    print(len(dataset))

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # plt.imshow(dataset[1][0][2].permute(1, 2, 0))
    # plt.show()

    image_resolution = 240
    transform = transforms.Compose([
        transforms.Resize((int(image_resolution/1.76), image_resolution)),
        #transforms.CenterCrop(224),
        transforms.Grayscale(),
        # transforms.RandAugment(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=(0.5,), std=(0.2,))
    ])
    dataset = ImageSet(transform=transform)
    image = dataset[0][0][0]
    # image = image + 0.1*torch.randn_like(image)
    print(image)
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.show()

    # plt.hist(image.permute(1, 2, 0).ravel(), bins=20)
    # plt.show()

    print('Stop')