import segmentation_models_pytorch as smp
import torch
import torchvision
from data import create_valley_data_loader

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

# 训练参数设定
DEVICE = 'cpu'
NUM_EPOCH = 40
# 数据集
train_loader, valid_loader = create_valley_data_loader()

# 模型
# fpn = model.create_model()
net = smp.FPN('resnet18', in_channels=1, classes=2)

# 损失函数
loss = smp.utils.losses.DiceLoss()

# 验证指标
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# 优化器
optimizer = torch.optim.Adam([
    dict(params=net.parameters(), lr=0.0001),
])


def train():
    net.train()
    # net.cuda()
    seg_loss = 0.0

    for epoch in range(NUM_EPOCH):
        for dem, label in train_loader:
            print(dem, label)


if __name__ == '__main__':
    train()
