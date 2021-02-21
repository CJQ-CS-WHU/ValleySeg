import segmentation_models_pytorch as smp
import torch
import model
import torchvision
from data import create_valley_data_loader

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
# DEVICE = 'cuda'
DEVICE = 'cpu'
train_loader, valid_loader = create_valley_data_loader()

# fpn = model.create_model()
model = smp.FPN('resnet18', in_channels=1, classes=2)
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])
