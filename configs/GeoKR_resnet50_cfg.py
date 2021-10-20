'''
@Project : RepresentationLearningLand 
@File    : calss_resnet50_meanteacher_cfg.py
@Author  : Wenyuan Li
@Date    : 2020/11/18 21:25 
@Desc    :  
'''

mode='train'
img_size=256
config=dict(
    name='InterfaceRepresentation',
    backbone_cfg=dict(
        name='resnet50',
        num_classes=None,
        in_channels=3,
        pretrained=True,
        out_keys=None
    ),
    head_cfg=dict(
        name='ClassificationHead',
        in_key=None,
        feature_finetune=True, # whether ot not to finetune  conv layers
        in_channels=2048,
        hidden_channels=None,
        num_classes=8,
        img_size=img_size,
    ),
    train_cfg=dict(
        batch_size=8,
        device='0',
        num_epoch=10,
        num_workers=0,
        update_step=[x for x in range(500,30801,10000)],
        train_data=dict(
            data_path=r'I:\experiments\representation_learning_land\dataset\GeoKR\train.txt',
            label_path=r'J:\experiments\GeoCon\dataset\RSIData_WithLabels_10landcovers',
            data_format='*.tif',
            class_list=['Artifical_Surfaces', 'Bareland', 'Cultivated_Land', 'Foreast', 'Grassland', 'Permanent_Snow',
                        'Waterbodies', 'Wetland'],
            img_size=img_size,
            with_label=True,
            with_name=False,
            sample_step=-1,
            transforms_cfg=dict(
                RandomHorizontalFlip=dict(name='RandomHorizontalFlip'),
                RandomVerticalFlip=dict(name='RandomVerticalFlip'),
                Rotate=dict(name='Rotate'),
                ColorJitter=dict(name='ColorJitter', brightness=0.3, contrast=(0.5, 1.5),
                                 saturation=(0.5, 1.5),hue=(-0.3, 0.3)),
                Resize=dict(name='Resize',size=(img_size,img_size)),
                ToTensor=dict(name='ToTensor')
            ),
        ),
        valid_data=dict(
            data_path=r'I:\experiments\representation_learning_land\dataset\GeoKR\train.txt',
            label_path=r'J:\experiments\GeoCon\dataset\RSIData_WithLabels_10landcovers',
            data_format='*.tif',
            class_list=['Artifical_Surfaces', 'Bareland', 'Cultivated_Land', 'Foreast', 'Grassland', 'Permanent_Snow',
                        'Waterbodies', 'Wetland'],
            img_size=img_size,
            with_label=True,
            with_name=False,
            sample_step=-1,
            transforms_cfg=dict(
                Resize=dict(name='Resize',size=(img_size,img_size)),
                ToTensor=dict(name='ToTensor')
            ),
        ),

        losses=dict(
            representationLoss=dict(name='MeanTeacherLoss')
        ),
        metric=dict(
            as_binary=False,num_classes=10,as_mean=True
        ),
        optimizer=dict(
            name='Adam',
            lr=0.001
        ),
        checkpoints=dict(
            checkpoints_path=r'checkpoints/pretrain/GeoKP_resnet50',
            save_step=1,
            with_pretrained=False,
            pretrained_checkpoints_path=
            r'checkpoints/pretrain/checkpoints_RSIData_resnet50',
            save_last=True
        ),
        lr_schedule=dict(
            name='stepLR',
            step_size=1,
            gamma=0.9
        ),
        log=dict(
            log_path=r'log/pretrain/GeoKP_resnet50',
            log_step=50,
            with_vis=False,
            vis_path=r''
        ),
    ),
)
