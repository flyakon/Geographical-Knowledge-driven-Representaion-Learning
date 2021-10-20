'''
@Project : RepresentationLearningLand 
@File    : mean_teacher_losses.py
@Author  : Wenyuan Li
@Date    : 2020/10/28 17:56 
@Desc    :  
'''
import torch
import torch.nn as nn


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()


    def forward(self,pred_s,pred_t,labels)->(torch.Tensor,torch.Tensor):
        prob_s=torch.softmax(pred_s,dim=1)
        classification_loss=-labels*torch.log_softmax(pred_s,dim=1)
        classification_loss=torch.mean(classification_loss)
        consistence_loss=-prob_s*torch.log_softmax(pred_t,dim=1)
        consistence_loss=torch.mean(consistence_loss)
        return classification_loss,consistence_loss

