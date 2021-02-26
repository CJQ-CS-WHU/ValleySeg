import torch
import torch
from torch import nn
import math
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# class Mnist_net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
#         self.bias = nn.Parameter(torch.zeros(10))
#
#     def forward(self, xb):
#         xb = xb @ self.weights + self.bias
#         return xb
#
#
# model = Mnist_net()
# loss_func = F.cross_entropy
# xb = torch.randn(2, 784)
# yb = torch.tensor([1, 1])
# print("model(xb):", model(xb).size())
# print("yb:", yb.size())
# print(yb)
# loss = loss_func(model(xb), yb)
# print("loss:", loss)
if __name__ == '__main__':
    # print("")
    # dem = torch.randn(30, 11, 2, 10)
    # label = torch.ones(30, 11, 2, 10).long()
    # loss = torch.nn.CrossEntropyLoss()
    # l = loss(dem, label)
    # print(l)
    # --------------------------------------------- #
    # import torch
    # import numpy as np
    #
    # gt = np.random.randint(0, 2, size=[5, 5])  # 先生成一个15*15的label，值在5以内，意思是5类分割任务
    # gt = torch.LongTensor(gt)
    #
    #
    # def get_one_hot(label, N):
    #     size = list(label.size())
    #     label = label.view(-1)  # reshape 为向量
    #     ones = torch.sparse.torch.eye(N)
    #     ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    #     size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    #     return ones.view(*size)
    #
    #
    # gt_one_hot = get_one_hot(gt, 3)
    # # print(gt_one_hot)
    # print(gt_one_hot.shape)

    # print(gt_one_hot.argmax(-1) == gt)  # 判断one hot 转换方式是否正确，全是1就是正确的
    # ---------------------------------------------- #
    # import torch
    # from tensorboardX import SummaryWriter
    #
    # writer = SummaryWriter()
    # x = torch.FloatTensor([100])
    # y = torch.FloatTensor([500])
    #
    # for epoch in range(100):
    #     x /= 1.5
    #     y /= 1.5
    #     loss = y - x
    #     print(loss)
    #     writer.add_histogram('zz/x', x, epoch)
    #     writer.add_histogram('zz/y', y, epoch)
    #     writer.add_scalar('data/x', x, epoch)
    #     writer.add_scalar('data/y', y, epoch)
    #     writer.add_scalar('data/loss', loss, epoch)
    #     writer.add_scalars('data/scalar_group', {'x': x,
    #                                              'y': y,
    #                                              'loss': loss}, epoch)
    #     writer.add_text('zz/text', 'zz: this is epoch ' + str(epoch), epoch)
    #
    # # export scalar data to JSON for external processing
    # writer.export_scalars_to_json("./test.json")
    # writer.close()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    y_pre_1 = torch.zeros(1, 1, 256, 256).to(torch.float32)
    y_pre_2 = torch.ones(1, 1, 256, 256).to(torch.float32)

    y_pre = torch.cat([y_pre_2, y_pre_1], 1)
    print(y_pre.shape)
    y_label = torch.ones(1, 256, 256).long()
    print(y_label.shape)
    # iou = metrics[0](y_pre, y_label)
    loss = torch.nn.CrossEntropyLoss()
    # loss = smp.losses.FocalLoss(mode='binary', alpha=0.5)
    l = loss(y_pre, y_label)
    print(l)
    # print(iou)

def show_dem_label(dem, pred, label):
    plt.subplot(2, 2, 1)
    plt.title('DEM ' + str(image_idx))
    plt.imshow(dem)
    plt.subplot(2 * n, 2, 2 * i + 2)
    plt.title('LABEL ' + str(image_idx))
    plt.imshow(label)
    print(2 * i + 2)