import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class GPSSet(Dataset):
    def __init__(self, path='./dataset', transform=None):
        self.df = pd.read_csv(
            os.path.join(path, 'ml_challenge_dev_multi_modal.csv'), 
            usecols=['unit1_loc', 'unit2_loc_1', 'unit2_loc_2', 'unit1_beam']
        )
        
        self.transform = transform
        self.path = path    # path argument is needed since data is saved as relative path

        self.delete_later = torch.randn(len(self.df), 1, 10)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.loc[idx]
        position_paths = item.drop(['unit1_beam', 'unit1_loc'])     # Path to position of user
        position_bs_path = item.get('unit1_loc')                    # Path to position of base station
        label = item.get('unit1_beam')                              # Beam index

        position_bs = np.loadtxt(os.path.join(self.path, position_bs_path))
        positions = None
        for key in position_paths.keys():
            path = os.path.join(self.path, position_paths.get(key))
            position = np.loadtxt(path)

            # Find position relative to base station and normalize
            # The reason we are doing this is because the changes in the absolute position are negligible
            # For "normalization", the value 1000 is chosen arbitrarily. Must find better technique (divide by maximum positions?)
            position = (position - position_bs) * 1000
            
            position = torch.tensor(position).float()
            if self.transform:
                position = self.transform(position)
            # image = torch.squeeze(image)    # In case we apply BGR2GRAY transform

            position = torch.unsqueeze(position, dim=0)   # Used for stacking images
            if positions == None:
                positions = position
            else:
                positions = torch.cat((positions, position), dim=0)

        label = torch.tensor(label)

        label -= 1      # Beam indices start from 1 instead of 0

        return positions, label

if __name__ == '__main__':
    path = 'N:\\Claudio\\Competition - Multi Modal\\dataset'
    dataset = GPSSet(path=path)
    print(dataset[0])
    print(len(dataset))

    print('Stop')