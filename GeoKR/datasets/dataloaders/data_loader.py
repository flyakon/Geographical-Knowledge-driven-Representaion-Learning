'''
@Project : RepresentationLearningLand 
@File    : data_loader.py
@Author  : Wenyuan Li
@Date    : 2020/10/16 18:06 
@Desc    :  
'''

import torch
import torch.utils.data as data_utils
import os
import numpy as np
import queue
import time
import datetime
import multiprocessing

class AyscDataloader(data_utils.DataLoader):
    def __init__(self, dataset,epoch,batch_size=1,drop_last=True,shuffle=True
                 ,num_workers=4,data_queue_maxsize=50,*args,**kwargs):

        super(AyscDataloader,self).__init__(dataset,batch_size=batch_size,drop_last=drop_last
                                            ,shuffle=shuffle,num_workers=num_workers,**kwargs)
        self.max_epoch=epoch
        self.epoch=0
        self.data_idx_list=list(range(0,len(dataset)))
        self.data_len=len(dataset)
        self.data_queue_maxsize=data_queue_maxsize

        self.shuffle=shuffle
        self.data_queue = multiprocessing.Manager().Queue(50)


    def get_train_data(self, data_queue):

        while self.epoch < self.max_epoch:
            if self.shuffle:
                np.random.shuffle(self.data_idx_list)
                idx_list=self.data_idx_list
            else:
                idx_list=self.data_idx_list
            if self.drop_last:
                end_idx=self.data_len-self.batch_size
            else:
                end_idx=self.data_len

            idx=0
            while idx<end_idx:
                if not data_queue.full():
                    # start_time = datetime.datetime.now()
                    data_batch=[self.dataset[idx_list[i]] for i in range(idx,idx+self.batch_size)]

                    data_batch=self.collate_fn(data_batch)
                    data_queue.put(data_batch)
                    # end_time = datetime.datetime.now()
                    # delta = (end_time - start_time)
                    # delta = delta.microseconds / 1000.+delta.seconds*1000.
                    # print(delta)
                    idx=idx+self.batch_size
                else:
                    time.sleep(0.001)

            self.epoch=self.epoch+1


if __name__=='__main__':
    multiprocessing.freeze_support()
    from RepresentationLearningLand.datasets.datasets.classification.classfication_dataset import ClassificationDataset
    import torchvision

    transforms_cfg = dict(
        RandomHorizontalFlip=dict(name='RandomHorizontalFlip'),
        RandomVerticalFlip=dict(name='RandomVerticalFlip'),
        Rotate=dict(name='Rotate'),
        ColorJitter=dict(name='ColorJitter', brightness=0.3, contrast=(0.5, 1.5),
                         saturation=(0.5, 1.5), hue=(-0.3, 0.3)),
        Resize=dict(name='Resize', size=(512, 512)),
        ToTensor=dict(name='ToTensor')
    )
    class_list=['Artifical_Surfaces','Cultivated_Land','Waterbodies','Foreast','Grassland','Wetland']
    dataset=ClassificationDataset(r'G:\expriment\representation_learning_land\dataset\val.txt',transforms_cfg,class_list)
    dataloader=AyscDataloader(dataset,50,4,num_workers=4)


    data_process = multiprocessing.Pool(1)
    for _ in range(1):
        data_process.apply_async(dataloader.get_train_data, (dataloader.data_queue,))


    while True:
        time.sleep(1)
        size = dataloader.data_queue.qsize()
        if size>0:
            dataloader.data_queue.get()