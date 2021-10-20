'''
@Project : RepresentationLearningLand 
@File    : classication_ambiguity_dataset.py
@Author  : Wenyuan Li
@Date    : 2020/10/28 16:15 
@Desc    :  
'''

import torch
import torch.nn as nn
import torchvision
import os
import torch.utils.data as data_utils
import numpy as np
from PIL import Image
import torchvision.transforms
import glob
from GeoKR.utils import path_utils
from GeoKR.datasets.transforms.classification.builder import build_transforms
from GeoKR.datasets.transforms.classification.classification_transforms import Compose
import json
import random
from ..base_dataset import BaseDataset
import cv2

class RepresentationDataset(BaseDataset):

    def __init__(self,data_path,transforms_cfg,class_list,label_path,
                 with_label=True,sample_step=10,with_name=False,
                 data_format='*.jpg',batch_size=1,num_epoch=1,start_epoch=0,drop_last=False,**kwargs):
        super(RepresentationDataset, self).__init__(num_epoch,start_epoch)
        if os.path.isdir(data_path):
            self.data_folder = data_path
            self.data_files = glob.glob(os.path.join(self.data_folder, data_format))
        else:
            self.data_folder = data_path
            self.data_files = np.loadtxt(self.data_folder, dtype=np.str,encoding='utf-8').tolist()
        if sample_step > 0:
            self.data_files = self.data_files[0::sample_step]
        self.with_label = with_label
        self.with_name = with_name
        self.class_list = class_list
        self.label_path=label_path
        self.num_epoch=num_epoch
        self.start_epoch=start_epoch
        self.label_dict={}


        transforms = []
        for param in transforms_cfg.values():
            transforms.append(build_transforms(**param))
        self.transforms = Compose(transforms)

        self.data_len=len(self.data_files)
        if drop_last:
            self.data_len=int(self.data_len//batch_size)*batch_size

    def __getitem__(self, item):
        item=self.get_item(item)
        item_result = []
        img_file = self.data_files[item]
        try:
            img=cv2.imread(img_file)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transforms(img)
            item_result.append(img)
        except:
            count = 0
            while count < 10:
                try:
                    print('读取出错，开始第%d次随机读取' % count)
                    id = random.randint(0, self.__len__())
                    img_file = self.data_files[(item + id) % self.data_len]
                    # file_name = path_utils.get_filename(img_file, is_suffix=False)
                    img = Image.open(img_file)
                    img = self.transforms(img)
                    item_result.append(img)
                    count += 1
                    break
                except:
                    count += 1
                    continue

        if self.with_label:
            if self.label_dict.get(img_file) is not None:
                item_result.append(self.label_dict[img_file])
            else:
                file_name = path_utils.get_filename(img_file, is_suffix=False)
                parent_folder=path_utils.get_parent_folder(img_file,with_root=False)
                label_path = os.path.join(self.label_path,parent_folder, '%s.json' % file_name)
                label_json = json.load(open(label_path, 'r'))
                labels=[]
                for cls in self.class_list:
                    if isinstance(cls,tuple) or isinstance(cls,list):
                        tmp=0
                        for c in cls:
                            tmp+=label_json[c]
                        labels.append(tmp)
                    elif isinstance(cls,str):
                        labels.append(label_json[cls])
                    else:
                        raise NotImplementedError(cls)
                labels = np.array(labels)
                labels = labels / (np.sum(labels) + 1e-6)
                labels = torch.from_numpy(labels)
                self.label_dict[img_file] = labels
                item_result.append(labels)
        if self.with_name:
            item_result.append(img_file)

        return item_result
