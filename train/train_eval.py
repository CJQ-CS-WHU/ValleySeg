import sys
import os
from tensorboardX import SummaryWriter
import segmentation_models_pytorch as smp
import torch
import torchvision
import torch.nn.functional as F

project_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_path + r'/../data')

from data import create_valley_data_loader

# 模型全集：https://smp.readthedocs.io/en/latest/models.html#linknet
# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
# 启动tensorboardX命令：


DEVICE = 'cuda'
NUM_EPOCH = 40
SAVE_PRE = 1
EVAL_PRE = 1
PRINT_PRE = 1
SAVE_DIR = project_path + r'/../result/pth'

writer = SummaryWriter(project_path + r'/../runs')  # 数据存放在这个文件夹
NUM_STEPS_STOP = 100000
# 数据集
train_loader, valid_loader = create_valley_data_loader()
cuda_id = 2
# 模型
# fpn = model.create_model()
net = smp.Linknet('resnet50', in_channels=1, classes=2).cuda(cuda_id)
# net = torch.nn.DataParallel(net, device_ids=[1, 2, 3])
# 损失函数
loss = torch.nn.CrossEntropyLoss().cuda(cuda_id)

# 验证指标
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# 优化器
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001)


def train():
    net.train()
    net.cuda(cuda_id)
    print('-' * 20)
    print('Train')
    print('-' * 20)
    i = 0
    train_loss = 0.0
    val_loss = 0.0
    for epoch in range(NUM_EPOCH):
        for dem, label in train_loader:
            i += 1
            optimizer.zero_grad()
            # print(dem.size())
            pred = net(dem.cuda(cuda_id))
            # print(pred.size())
            # print(label.size())
            seg_loss = loss(pred.cuda(cuda_id), label.cuda(cuda_id))
            # mIOU = metrics[0](pred, label)
            seg_loss.backward()
            optimizer.step()
            train_loss += seg_loss

            # 保存权重
            if i % SAVE_PRE == 0:
                print('save:', i)
                torch.save(net.state_dict(), SAVE_DIR + '/iter_' + str(i) + '.pth')

            # 模型检验
            # val_len = len(valid_loader)

            if i % EVAL_PRE == 0:
                val_len = 500
                j = 0
                val_loss = 0.0
                net.eval()
                with torch.no_grad():
                    for eval_dem, eval_label in valid_loader:
                        j += 1
                        eval_pred = net(eval_dem.cuda(cuda_id))
                        val_loss += loss(eval_pred.cuda(cuda_id), eval_label.cuda(cuda_id))
                        if j >= val_len:
                            break
                    val_loss /= val_len

            # 记录参数
            writer.add_scalar('seg loss', seg_loss, i)
            if i % PRINT_PRE == 0:
                print('[iter %d][loss train %.4f][loss val %.4f]' % (i + 1, train_loss / PRINT_PRE, val_loss))
                train_loss = 0.0
        writer.export_scalars_to_json(project_path + r"/../result/log/epoch_" + str(epoch) + ".json")


writer.close()
# 1. 计算多个指标 cross entropy +mIOU
# 2. 保存每个阶段指标解最优的模型 暂时不需要
# 3. 模型检验部分代码 已完成
# 4.
if __name__ == '__main__':
    train()
    print(" ")
