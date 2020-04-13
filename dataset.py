#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-13 21:50
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : dataset.py
"""

from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, default_loader
from PIL import Image


def grey_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class Faces(Dataset):

    def __init__(self, root, loader=default_loader, transform=None, target_transform=None):
        super(Faces, self).__init__()
        self.folder = ImageFolder(root, transform=transform, target_transform=target_transform,
                                  loader=loader)

    def __getitem__(self, index):
        return self.folder.__getitem__(index)

    def __len__(self):
        return self.folder.__len__()