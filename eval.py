import sys
import os
from tensorboardX import SummaryWriter
import segmentation_models_pytorch as smp
import torch
import torchvision
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

project_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_path + r'/data')
localtime = time.asctime(time.localtime(time.time()))
from data import create_valley_data_loader

# 模型全集：https://smp.readthedocs.io/en/latest/models.html#linknet
# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
# 启动tensorboardX命令：


cuda_id = 2
DEVICE = 'cuda'
NUM_EPOCH = 20
SAVE_PRE = 1
EVAL_PRE = 1
PRINT_PRE = 1
NUM_STEPS_STOP = 100000
PTH_DIR = project_path + r'/../checkpoints/' \
          + 'time.struct_time(tm_year=2021, tm_mon=2, tm_mday=26, tm_hour=15, tm_min=34, tm_sec=33, tm_wday=4, ' \
            'tm_yday=57, tm_isdst=0)' + '/iter_31.pth'
FIG_SAVE_PATH = project_path + r'/../'
# 数据集
train_loader, valid_loader = create_valley_data_loader()

# 模型
net = smp.Linknet('resnet50', in_channels=1, classes=2).cuda(cuda_id)

# 验证损失
eval_loss = 0.0
CUDA_ID = 2


def eval():
    net.load_state_dict(torch.load(PTH_DIR))
    net.cuda(CUDA_ID)
    net.eval()
    i = 0
    for dem, label in iter(valid_loader):
        print(dem.shape)
        print(label.shape)
        i += 1
        pred = net(dem.cuda(CUDA_ID))
        pred = F.softmax(pred, dim=1)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        pred = pred[:, 1, :, :]
        print(pred.shape)
        save(dem, pred, label, i)
        exit()


def save(dem, pred, label, i):
    """
    pred <class 'torch.Tensor'>
    dem <class 'torch.Tensor'>
    label <class 'torch.Tensor'>
    """
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(dem.view(256, 256))
    plt.title('DEM')
    plt.subplot(3, 1, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pred.view(256, 256).int().cpu())
    plt.title('PREDICT')
    plt.subplot(3, 1, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(label.view(256, 256))
    plt.title("LABEL")
    plt.savefig('linknet_iter_31_' + str(i) + '.png')
    plt.show()
    print('linknet_iter_31_' + str(i) + '.png')


if __name__ == '__main__':
    eval()
