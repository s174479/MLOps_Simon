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
    def __init__(self, filepaths):
        mnist_data = {}
        for f in filepaths:
            data = np.load(f)
            images = data['images']
            labels = data['labels']
            if (len(mnist_data)==0):
                mnist_data['images'] = images
                mnist_data['labels'] = labels
            else:
                mnist_data['images'] = np.concatenate((mnist_data['images'], images), axis=0)
                mnist_data['labels'] = np.concatenate((mnist_data['labels'], labels), axis=0)
        self.imgs = mnist_data['images']
        self.labels = mnist_data['labels']
  
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return (self.imgs[idx], self.labels[idx])



def mnist():
    # exchange with the corrupted mnist dataset
    train_dl = MyDataset(['corruptmnist/train_0.npz', 'corruptmnist/train_1.npz', 'corruptmnist/train_2.npz', 'corruptmnist/train_3.npz', 'corruptmnist/train_4.npz'])
    test_dl = MyDataset(['corruptmnist/test.npz'])
    return train_dl, test_dl


