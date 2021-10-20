'''
@Project : RepresentationLearningLand 
@File    : trian.py.py
@Author  : Wenyuan Li
@Date    : 2020/10/9 10:55 
'''

import argparse
import mmcv
from GeoKR.interface.builder import builder_models
from GeoKR.utils import utils


parse=argparse.ArgumentParser()
parse.add_argument('--config_file',default=r'configs/GeoKR_resnet50_cfg.py',type=str)
#
parse.add_argument('--checkpoints_path',default=None,type=str)
parse.add_argument('--precheckpoints_path',default=None,type=str)
parse.add_argument('--with_pretrain',default=None,type=utils.str2bool)
parse.add_argument('--with_imagenet',default=None,type=utils.str2bool)
parse.add_argument('--sample_step',default=None,type=int)
parse.add_argument('--log_path',default=None,type=str)


if __name__=='__main__':
    args = parse.parse_args()
    print(args)
    cfg = mmcv.Config.fromfile(args.config_file)
    if args.with_imagenet is not None:
        cfg['config']['backbone_cfg']['pretrained'] = args.with_imagenet

    models=builder_models(**cfg['config'])
    models.run_train_interface(checkpoint_path=args.checkpoints_path,
                              with_pretrained=args.with_pretrain,
                              precheckpoints_path=args.precheckpoints_path,
                              sample_step=args.sample_step,
                              log_path=args.log_path)
