from torch.utils.data.dataloader import DataLoader
import sys
import os
project_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(r'..\data')
from valley_dataset import ValleyDataset
import valley_dataset


def create_valley_data_loader(args=None):
    if args is None:
        args = []
    train_dataset = ValleyDataset(
        data_txt=project_path + r'/../datasets/train.txt',
        data_folder=project_path + r'/../..',
        mean=0, std=1, phase='train')
    valid_dataset = ValleyDataset(
        data_txt=project_path + r'/../datasets/train.txt',
        data_folder=project_path + r'/../..',
        mean=0, std=1, phase='valid')

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, valid_loader


if __name__ == '__main__':
    train_loader, test_loader = create_valley_data_loader()
    for i, (dem, label) in enumerate(train_loader):
        print(i)
