'''
@Project : RepresentationLearningLand 
@File    : classification_ambiguity_net.py
@Author  : Wenyuan Li
@Date    : 2020/10/28 16:30 
@Desc    :  
'''

import torch
import torch.nn as nn
import torchvision
import os
import torch.utils.data as data_utils
from .mean_teacher_net import MeanTeacherNet


class RepresentationNet(nn.Module):
    def __init__(self, backbone_cfg: dict,
                 head_cfg: dict, **kwargs):
        super(RepresentationNet, self).__init__()
        self.student_model=MeanTeacherNet(backbone_cfg,head_cfg)
        self.teacher_model=MeanTeacherNet(backbone_cfg,head_cfg)
        self.backbone_cfg=backbone_cfg



    def forward(self,input:nn.Module)\
            ->(nn.Module,nn.Module,nn.Module,nn.Module):
        logits_s,prob_s=self.student_model.forward(input)
        with torch.no_grad():
            logits_t, prob_t = self.teacher_model.forward(input)
        return logits_s,prob_s,logits_t, prob_t



    def print_key_args(self,**kwargs):
        for key,value in kwargs.items():
            print('{0}:{1}'.format(key,value))



