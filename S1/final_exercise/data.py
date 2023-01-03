import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

#def mnist():
    # exchange with the corrupted mnist dataset
#    train_paths = ['S1/final_exercise/corruptmnist/train_0.npz', 'S1/final_exercise/corruptmnist/train_1.npz', 'S1/final_exercise/corruptmnist/train_2.npz', 'S1/final_exercise/corruptmnist/train_3.npz', 'S1/final_exercise/corruptmnist/train_4.npz']
    #train = [np.load(f) for f in train_paths]
#    train = np.load('S1/final_exercise/corruptmnist/train_0.npz')
#    print('hej')
#    print(train['labels'][0])
#    train_dl = zip(train['images'], train['labels'])
#    test = np.load('S1/final_exercise/corruptmnist/test.npz')
#    return train_dl, test

class MyDataset(Dataset):
    def __init__(self, *filepaths):
        content = [np.load(f) for f in filepaths]
        self.imgs, self.labels = content[0] | content[1] | content[2] | content[3] | content[4] 
  
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return (self.imgs[idx], self.labels[idx])



def mnist():
    # exchange with the corrupted mnist dataset
    train_dl = DataLoader(MyDataset('S1/final_exercise/corruptmnist/train_0.npz', 'S1/final_exercise/corruptmnist/train_1.npz', 'S1/final_exercise/corruptmnist/train_2.npz', 'S1/final_exercise/corruptmnist/train_3.npz', 'S1/final_exercise/corruptmnist/train_4.npz'), batch_size = 16)
    test_dl = DataLoader(MyDataset('S1/final_exercise/corruptmnist/test.npz'), batch_size = 16)
    return train_dl, test_dl