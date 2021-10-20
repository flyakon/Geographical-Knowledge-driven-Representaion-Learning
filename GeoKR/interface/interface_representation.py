'''
@Project : RepresentationLearningLand 
@File    : interface_mean_tacher.py
@Author  : Wenyuan Li
@Date    : 2020/11/13 20:18 
@Desc    :  
'''

import torch
import torch.nn as nn
import os
import torch.utils.data as data_utils
from GeoKR.models.representation.representation_net import RepresentationNet
from GeoKR.datasets.datasets.representation import representation_dataset
from GeoKR.losses.builder import builder_loss
from GeoKR.utils.optims.builder import build_lr_schedule,build_optim
from torch.utils.tensorboard import SummaryWriter
from GeoKR.metric.time_metric import TimeMetric
from torch.cuda.amp import autocast,GradScaler

class InterfaceRepresentation():
    def __init__(self, backbone_cfg: dict,
                 head_cfg: dict,
                 train_cfg: dict,
                 test_cfg: dict, **kwargs):
        super(InterfaceRepresentation, self).__init__()
        self.model=RepresentationNet(backbone_cfg=backbone_cfg,head_cfg=head_cfg
                                          ,train_cfg=train_cfg,test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.backbone_cfg = backbone_cfg

    def run_train_interface(self,**kwargs):
        batch_size = self.train_cfg['batch_size']
        device_id = self.train_cfg['device']
        device_id=[int(id) for id in device_id.split(',')]
        device='cuda' if device_id[0]>=0 else 'cpu'
        num_epoch = self.train_cfg['num_epoch']
        num_workers = self.train_cfg['num_workers']
        if 'checkpoint_path' in kwargs.keys() and kwargs['checkpoint_path'] is not None:
            checkpoint_path=kwargs['checkpoint_path']
        else:
            checkpoint_path = self.train_cfg['checkpoints']['checkpoints_path']
        if 'with_pretrained' in kwargs.keys() and kwargs['with_pretrained'] is not None:
            with_pretrained=kwargs['with_pretrained']
        else:
            with_pretrained=self.train_cfg['checkpoints']['with_pretrained']
        if 'sample_step' in kwargs.keys() and kwargs['sample_step'] is not None:
            self.train_cfg['train_data']['sample_step']=kwargs['sample_step']
        if 'log_path' in kwargs.keys() and kwargs['log_path'] is not None:
            log_path = kwargs['log_path']
        else:
            log_path = self.train_cfg['log']['log_path']
        if 'precheckpoints_path' in kwargs.keys() and kwargs['precheckpoints_path'] is not None:
            pretrained_checkpoints_path=kwargs['precheckpoints_path']
        else:
            pretrained_checkpoints_path=self.train_cfg['checkpoints']['pretrained_checkpoints_path']
        save_step = self.train_cfg['checkpoints']['save_step']
        save_last = self.train_cfg['checkpoints']['save_last']

        log_step = self.train_cfg['log']['log_step']
        with_vis = self.train_cfg['log']['with_vis']
        vis_path = self.train_cfg['log']['vis_path']

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file=os.path.join(log_path,'log.txt')
        log_fp=open(log_file,'w')

        self.print_key_args(checkpoint_path=checkpoint_path,
                            with_pretrained=with_pretrained,
                            pretrained_checkpoints_path=pretrained_checkpoints_path,
                            with_imagenet=self.backbone_cfg['pretrained'],
                            sample_step=self.train_cfg['train_data']['sample_step'],
                            log_path=log_path,
                            train_data_path=self.train_cfg['train_data']['data_path'],
                            valid_data_path=self.train_cfg['valid_data']['data_path'],
                            device=device)
        update_steps=self.train_cfg['update_step']
        if with_vis:
            if not os.path.exists(os.path.join(vis_path,'train_result')):
                os.makedirs(os.path.join(vis_path,'train_result'))
            if not os.path.exists(os.path.join(vis_path,'valid_result')):
                os.makedirs(os.path.join(vis_path,'valid_result'))


        self.model.to(device)
        current_epoch, global_step = self.model.student_model.load_model(checkpoint_path, prefix='student')
        current_epoch, global_step = self.model.teacher_model.load_model(checkpoint_path, prefix='teacher')

        if len(device_id) > 1:
            parallel_model = nn.DataParallel(self.model)
        else:
            parallel_model=self.model

        train_dataset = representation_dataset.RepresentationDataset(**self.train_cfg['train_data']
                                                                                      ,num_epoch=num_epoch,
                                                                                      start_epoch=current_epoch)
        train_dataloader = data_utils.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False,
                                                 num_workers=num_workers)
        data_len = train_dataset.get_loader_len(batch_size)


        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        criterion=builder_loss(**self.train_cfg['losses']['representationLoss'])
        optimizer = build_optim(params=self.model.student_model.parameters(), **self.train_cfg['optimizer'])
        if 'lr_schedule' in self.train_cfg.keys():
            lr_schedule=build_lr_schedule(optimizer=optimizer,**self.train_cfg['lr_schedule'])

        summary = SummaryWriter(log_path)

        time_metric=TimeMetric()
        sorted(update_steps,reverse=True)
        if global_step>update_steps[-1]:
            update_step=update_steps[-1]
            steps_len=len(update_steps)
            i=0
            while i<steps_len:
                update_steps.pop(0)
                i+=1
        else:
            update_step=update_steps[0]
        update_step=int(update_step/batch_size*16)
        print(update_step,len(update_steps))
        last_epoch = current_epoch
        epoch=current_epoch
        scaler = GradScaler()
        for data in train_dataloader:
            global_step += 1
            epoch = train_dataset.get_epoch(global_step,batch_size)
            parallel_model.train()
            train_img, train_label = data
            train_img = train_img.to(device)
            train_label = train_label.to(device)
            with autocast():
                train_logits_s, train_prob_s,train_logits_t,train_prob_t = parallel_model.forward(train_img)
                train_class_loss,train_consistence_loss = criterion(train_logits_s,train_logits_t, train_label)
                train_loss=train_class_loss+train_consistence_loss
            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            # train_loss.backward()
            # optimizer.step()

            if global_step % log_step == 1:

                i=train_dataset.get_iter(global_step,batch_size)
                fps=time_metric.get_fps(log_step*batch_size)
                time_metric.reset()
                print('epoch:%d,batch:%d/%d,iter:%d/%d,class loss:%f consis loss:%f train loss:%f,fps:%f' %
                      (epoch, i,data_len,global_step,len(train_dataloader), train_class_loss.item(),train_consistence_loss.item(),
                       train_loss.item(), fps))
                log_fp.writelines('epoch:%d,batch:%d/%d,iter:%d/%d,class loss:%f consis loss:%f train loss:%f,fps:%f\n' %
                      (epoch, i,data_len,global_step,len(train_dataloader), train_class_loss.item(),train_consistence_loss.item(),
                       train_loss.item(), fps))
                log_fp.flush()
                summary.add_scalar('train/total_loss', train_loss, global_step)
                summary.add_scalar('train/class', train_class_loss, global_step)
                summary.add_scalar('train/consistence', train_consistence_loss, global_step)


            if global_step % update_step==1:
                print('update weights')
                self.model.teacher_model.update_weights(self.model.student_model)
                if len(update_steps)>0:
                    update_step=update_steps.pop(0)
                    update_step=int(update_step/batch_size*16)
                self.model.teacher_model.save_model(checkpoint_path, epoch, global_step, 'teacher')
                self.model.student_model.save_model(checkpoint_path, epoch, global_step, 'student')
            if epoch != last_epoch:
                last_epoch=epoch
                if 'lr_schedule' in self.train_cfg.keys():
                    lr_schedule.step(epoch=epoch)
                    summary.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'],
                                       global_step)
                lr = optimizer.param_groups[0]['lr']
                summary.add_scalar('learning_rate', lr, global_step)

                if save_step>0 and epoch % save_step == 0:
                    print('save model')
                    self.model.student_model.save_model(checkpoint_path,epoch,global_step,'student')
                    self.model.teacher_model.save_model(checkpoint_path,epoch,global_step,'teacher')


        if save_last:
            print('save model')
            self.model.student_model.save_model(checkpoint_path, epoch, global_step, 'student')
            self.model.teacher_model.save_model(checkpoint_path, epoch, global_step, 'teacher')
        log_fp.close()


    def print_key_args(self,**kwargs):
        for key,value in kwargs.items():
            print('{0}:{1}'.format(key,value))