import sys

sys.path.append(r'D:\anaconda3\envs\pytorch_py36\Library\bin\libtiff.dll')
sys.path.append(r'D:\anaconda3\envs\pytorch_py36\conda-meta\libtiff-4.1.0-h56a325e_1.json')
from PIL import Image
# from libtiff import TIFF
from imgaug.augmenters import HorizontalFlip
import torch
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from torchvision.transforms import Normalize, ToTensor, Compose
import matplotlib.pyplot as plt

DEVICE = 'cpu'


class ValleyDataset(data.Dataset):
    def __init__(self, data_txt, data_folder, mean, std, phase):
        self.df = pd.read_csv(data_txt, header=None)
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(mean, std, phase)
        self.tif_stream = []
        self.label_stream = []

        _len = len(self.df)
        for idx in range(0, _len):
            name = self.df.loc[idx][0]
            dem_path = self.root + '/images/' + str(name) + '.tif'
            label_path = self.root + '/labels/' + str(name) + '.png'
            dem = Image.open(dem_path)
            label = Image.open(label_path)
            self.tif_stream.append(dem)
            self.label_stream.append(label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print(self.tif_stream[idx])
        # print(np.array(self.label_stream[idx]).shape)
        dem = torch.from_numpy(np.array(self.tif_stream[idx])).reshape(1, 256, 256).to(torch.float32)
        label = torch.from_numpy(np.array(self.label_stream[idx])).permute(2, 0, 1)[0].view(256, 256).long()
        label[label == 255] = 1
        return dem, label


def visualize(dataset, idxs):
    n = len(idxs)
    print('total:', n)
    plt.figure(figsize=(8, 15))
    for i, image_idx in enumerate(idxs):
        dem, label = dataset.__getitem__(image_idx)
        plt.subplot(2 * n, 2, 2 * i + 1)
        print(2 * i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title('DEM ' + str(image_idx))
        plt.imshow(dem)
        plt.subplot(2 * n, 2, 2 * i + 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('LABEL ' + str(image_idx))
        plt.imshow(label)
        print(2 * i + 2)
    plt.show()


def get_transforms(mean, std, phase):
    list_transforms = []
    if phase == 'train':
        # 可训练时需要的预处理方法
        list_transforms.extend([
            HorizontalFlip(p=0.5),  # only horizontal flip as of now
        ])
    elif phase == 'val':
        # 可检验时需要的预处理方法
        pass
    else:
        pass
    # 都需要的预处理方法
    list_transforms.extend(
        [
            ToTensor(),
            # Normalize(mean=mean, std=std),
        ]
    )
    transforms = Compose(list_transforms)
    return transforms


if __name__ == '__main__':
    dataset = ValleyDataset(
        data_txt=r'F:\ValleySeg\datasets\train.txt',
        data_folder=r'G:\ValleyDataset',
        mean=0, std=1, phase='train')
    visualize(dataset, [0, 1, 2])
