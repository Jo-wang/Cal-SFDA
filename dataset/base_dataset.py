import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
import imageio
from PIL import ImageFilter
import pandas as pd
import albumentations as A
import cv2

class BaseDataSet(data.Dataset):
    def __init__(self, root, list_path, dataset, num_class, joint_transform=None, transform=None,
                 label_transform=None, max_iters=None, ignore_label=255, set='val',
                 plabel_path=None, max_prop=None, selected=None, centroid=None, wei_path=None):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.set = set
        self.num_class = num_class
        self.dataset = dataset
        self.transform = transform
        self.joint_transform = joint_transform
        self.label_transform = label_transform
        self.plabel_path = plabel_path
        self.centroid = centroid
        
        self.dict = {}

        if self.set != 'train':
            self.list_path = (self.list_path).replace('train', self.set)

        self.img_ids = []
        if selected is not None:
            self.img_ids = selected
        else:
            with open(self.list_path) as f:
                for item in f.readlines():
                    fields = item.strip().split('\t')[0]
                    if ' ' in fields:
                        fields = fields.split(' ')[0]
                    self.img_ids.append(fields)

        if max_iters is not None:
            self.img_ids = self.img_ids * \
                int(np.ceil(float(max_iters) / len(self.img_ids)))
        elif max_prop is not None:
            total = len(self.img_ids)
            to_sel = int(np.floor(total * max_prop))
            index = list(np.random.choice(total, to_sel, replace=False))
            self.img_ids = [self.img_ids[i] for i in index]

        self.files = []
        if self.dataset == "synthia":
            if self.num_class == 16:
                self.id2train = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                                 15: 6, 9: 7, 6: 8, 1: 9, 10: 10, 17: 11,
                                 8: 12, 19: 13, 12: 14, 11: 15}
        else:
            self.id2train = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                             19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                             26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        if self.dataset == 'cityscapes' and self.num_class == 16:

            self.id2train = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
                             21: 8, 23: 9, 24: 10, 25: 11, 26: 12, 28: 13, 32: 14, 33: 15}


        if self.dataset == 'gta5':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'labels')
            else:
                label_root = self.plabel_path

            for name in self.img_ids:
                img_file = osp.join(self.root, "images/%s" % name)
                label_file = osp.join(label_root, "%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
            print("finished")

        elif self.dataset == 'cityscapes':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'gtFine', self.set)
            else:
                label_root = self.plabel_path
                label_root1 = osp.join(self.root, 'gtFine', self.set)
            for name in self.img_ids:
                img_file = osp.join(
                    self.root, "leftImg8bit/%s/%s" % (self.set, name))
                label_name = name.replace('leftImg8bit', 'gtFine_labelIds')
                label_file = osp.join(label_root, '%s' % (label_name))
                if self.plabel_path is not None:
                    label_file1 = osp.join(label_root1, '%s' % (label_name))
                    self.files.append({
                        "img": img_file,
                        "label": label_file,
                        "gt": label_file1,
                        "name": name
                    })
                else:
                    self.files.append({
                        "img": img_file,
                        "label": label_file,
                        "name": name
                    })

        if self.dataset == 'synthia':
            imageio.plugins.freeimage.download()
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'GT/LABELS')
            else:
                label_root = self.plabel_path

            for name in self.img_ids:
                img_file = osp.join(self.root, "RGB/%s" % name)
                label_file = osp.join(label_root, "%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
            print("......")
            print("finised!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        try:
            image = Image.open(datafiles["img"]).convert('RGB')
           
            if self.dataset == 'synthia':
                name = datafiles["name"]
                label = np.asarray(imageio.imread(
                    datafiles["label"], format='PNG-FI'))[:, :, 0]
            else:
                name = datafiles["name"]
                label = Image.open(datafiles["label"])
                label = np.asarray(label, dtype=np.uint8)
            if self.plabel_path is not None:
                gt = Image.open(datafiles["gt"])
                gt = np.asarray(gt, dtype=np.uint8)
                gt_copy = 255 * np.ones(gt.shape, dtype=np.uint8)
                for k, v in self.id2train.items():
                    gt_copy[gt == k] = v
                gt = Image.fromarray(gt_copy.astype(np.uint8))
         

            if self.plabel_path is None:
                label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
                for k, v in self.id2train.items():
                    label_copy[label == k] = v

                label = Image.fromarray(label_copy.astype(np.uint8))
            else:
                label = Image.fromarray(label.astype(np.uint8))

            if self.joint_transform is not None:
               
                image, label, crop_position = self.joint_transform(image, label, None)
               
            if self.transform is not None:
                image = self.transform(image)  
                
                    
                     
            if self.label_transform is not None:
                label = self.label_transform(label)
                if self.plabel_path is not None:
                    gt = self.label_transform(gt)
           
            if self.plabel_path is None:
                gt = label

        except Exception as e:
            print(index)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        return image, label, gt, 1, name
