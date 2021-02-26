import sys
import os
from tensorboardX import SummaryWriter
import segmentation_models_pytorch as smp
import torch
import torchvision
import torch.nn.functional as F
import time

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
SAVE_DIR = project_path + r'/../checkpoints/' + str(localtime)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 数据集
train_loader, valid_loader = create_valley_data_loader()

# 模型
net = smp.Linknet('resnet50', in_channels=1, classes=1).cuda(cuda_id)

# 损失函数
loss = torch.nn.CrossEntropyLoss().cuda(cuda_id)

# 验证指标
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# 优化器
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001)

# tensorboardX
writer = SummaryWriter(project_path + r'/../runs/' + str(localtime))  # 数据存放在这个文件夹


def train():
    net.train()
    net.cuda(cuda_id)
    print('-' * 30)
    print('Train')
    print('-' * 30)
    i = 0
    train_loss = 0.0
    val_loss = 0.0
    train_iou = 0.0
    val_iou = 0.0
    for epoch in range(NUM_EPOCH):
        for dem, label in train_loader:
            i += 1
            optimizer.zero_grad()
            pred = net(dem.cuda(cuda_id))
            seg_loss = loss(pred.cuda(cuda_id), label.cuda(cuda_id))
            seg_loss.backward()
            optimizer.step()
            train_loss += seg_loss
            train_iou += metrics[0](pred, label.cuda(cuda_id))
            # 保存权重
            if i % SAVE_PRE == 0:
                torch.save(net.state_dict(), SAVE_DIR + '/iter_' + str(i) + '.pth')

            # 模型检验
            # val_len = len(valid_loader)

            if i % EVAL_PRE == 0:
                j = 0
                val_len = 500
                val_iou = 0.0
                val_loss = 0.0
                net.eval()
                with torch.no_grad():
                    for eval_dem, eval_label in valid_loader:
                        j += 1
                        eval_pred = net(eval_dem.cuda(cuda_id))
                        val_loss += loss(eval_pred, eval_label.cuda(cuda_id))
                        val_iou += metrics[0](eval_pred, eval_label.cuda(cuda_id))
                        if j >= val_len:
                            break
                val_loss /= val_len
                val_iou /= val_len

            # 记录参数
            writer.add_scalar('loss', seg_loss, i)
            writer.add_scalar('iou', train_iou, i)
            writer.add_scalar('loss', val_loss, i)
            writer.add_scalar('iou', val_loss, i)

            if i % PRINT_PRE == 0:
                print('[iter %d][loss train %.4f][loss val %.4f][mIOU train %.4f][mIOU val %.4f]'
                      % (i + 1, train_loss / PRINT_PRE, val_loss, train_iou / PRINT_PRE, val_iou / PRINT_PRE))
                train_loss = 0.0
        writer.export_scalars_to_json(project_path + r"/../result/log/epoch_" + str(epoch) + ".json")
    writer.close()


def save_pre(dem, pred, label):
    """
    保存一个图，左边是DEM，中间是预测，右边是label
    dem tensor :(batch ,2 ,256, 256)
    :return:
    """


if __name__ == '__main__':
    train()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    print(" ")
