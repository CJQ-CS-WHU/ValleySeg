import torch
import segmentation_models_pytorch as smp

# 建立模型：https://cuijiahua.com/blog/2019/12/dl-15.html
# 训练框架：https://github.com/Jack-Cherish/Deep-Learning/blob/master/Pytorch-Seg/lesson-2/train.py
# 模型选型：https://github.com/qubvel/segmentation_models

encoder = 'resnet18'
encoder_weight = 'imagenet'
classes = 1
in_channel = 1
activation = 'sigmoid'
device = 'cpu'


def create_model(args=None):
    global model
    if args == 'FPN':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weight,
            classes=classes,
            in_channels=in_channel,
            activation=activation,
        )
    elif args == 'UNet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weight,
            classes=classes,
            in_channels=in_channel,
            activation=activation)
    return model


if __name__ == '__main__':
    model = create_model('UNet')
    x = torch.randn((64, 1, 256, 256))
    print(x.shape)
    # print(model)
    y = model(x)  # torch.Size([64, 1, 256, 256])
    print(y.shape)
