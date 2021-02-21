import segmentation_models_pytorch as smp
import torch
import torchvision


def create_model(args=None):
    if args is None:
        args = []
    fpn, preprocess = FPN_Resnet()
    return fpn


import torch
import numpy as np
import segmentation_models_pytorch as smp


def FPN_Resnet(
        encoder='resnet18',
        encoder_weight='imagenet',
        classes=None,
        in_channel=1,
        activation='sigmoid',
        device='cuda'):
    # could be None for logits or 'softmax2d' for multicalss segmentation
    # create segmentation model with pretrained encoder
    if classes is None:
        classes = ['valley']
    model = smp.FPN(
        encoder_name=encoder,
        encoder_weights=encoder_weight,
        in_channels=1,
        classes=len(classes),
        activation=activation,
    )
    # model.cuda(device)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weight)
    return model, preprocessing_fn


if __name__ == '__main__':
    model = create_model()
    x = torch.randn((64, 1, 256, 256))
    print(x.shape)
    print(model)
    y = model(x)
    print(y.shape)
