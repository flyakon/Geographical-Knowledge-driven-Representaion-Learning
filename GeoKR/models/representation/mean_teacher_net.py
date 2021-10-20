'''
@Project : RepresentationLearningLand 
@File    : mean_teacher_net.py
@Author  : Wenyuan Li
@Date    : 2020/10/28 16:45 
@Desc    :  
'''

import torch
import torch.nn as nn
from GeoKR.models.backbone.builder import build_backbone
from GeoKR.utils import utils

class MeanTeacherNet(nn.Module):
    def __init__(self,backbone_cfg:dict,
                 head_cfg:dict):
        super(MeanTeacherNet, self).__init__()
        self.backbone=build_backbone(**backbone_cfg)
        in_channels = head_cfg['in_channels']
        num_classes = head_cfg['num_classes']
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        hidden_channels=head_cfg['hidden_channels']
        if hidden_channels is not None:
            self.class_fc = nn.Sequential(*[
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels,num_classes)
            ])
        else:
            self.class_fc = nn.Sequential(*[
                nn.Linear(in_channels, num_classes)
            ])

    def forward(self,input):
        x,_=self.backbone(input)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.class_fc(x)
        prob = self.softmax(logits)
        return logits, prob

    def update_weights(self,model:nn.Module,alpha=0.95):
        for w_t,w_s in zip(self.parameters(),model.parameters()):
            w_t.data=alpha*w_t.data+(1-alpha)*w_s.data

    def save_model(self,checkpoint_path,
                                 epoch, global_step,prefix='student'):
        utils.save_model(self,checkpoint_path,epoch,global_step,prefix)

    def load_model(self,checkpoint_path,prefix='cub_model'):
        state_dict, current_epoch, global_step = utils.load_model(checkpoint_path,prefix=prefix)
        if state_dict is not None:
            print('resume from epoch %d global_step %d' % (current_epoch, global_step))
            self.load_state_dict(state_dict, strict=True)
        return current_epoch,global_step



    def print_weights(self):
        for p,v in self.named_parameters():
            print(p,v.mean().item())
            break






if __name__=='__main__':
    backbone_cfg = dict(
        name='vgg16_bn',
        num_classes=None,
        in_channels=3,
        pretrained=True,
        out_keys=None
    )
    head_cfg = dict(
        name='ClassificationHead',
        in_key=None,
        feature_finetune=True,  # whether ot not to finetune  conv layers
        in_channels=512,
        num_classes=6,
        img_size=512,
    )
    student_model=MeanTeacherNet(backbone_cfg,head_cfg)
    teacher_model = MeanTeacherNet(backbone_cfg, head_cfg)
    student_model.print_weights()
    teacher_model.print_weights()
    student_model.update_weights(teacher_model)
    student_model.print_weights()
    teacher_model.print_weights()