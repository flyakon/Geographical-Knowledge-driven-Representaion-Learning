'''
@Project : UpSpeed 
@File    : base_dataset.py
@Author  : Wenyuan Li
@Date    : 2020/12/26 19:05 
@Desc    :  
'''
import torch
import torchvision
import torch.utils.data as data_utils
import numpy as np
class BaseDataset(data_utils.Dataset):
    def __init__(self,num_epoch=1,start_epoch=0,**kwargs):
        super(BaseDataset, self).__init__()
        self.num_epoch=num_epoch
        self.start_epoch=start_epoch

        self.data_len=num_epoch

    def __len__(self):
        return self.data_len*(self.num_epoch-self.start_epoch)

    def get_epoch(self,global_step,batch_size):
        return global_step//(np.ceil(self.data_len/batch_size))

    def get_iter(self,global_step,batch_size):
        return global_step%(np.ceil(self.data_len/batch_size))

    def get_item(self,item):
        return item%self.data_len

    def get_loader_len(self,batch_size):
        return self.data_len//batch_size
