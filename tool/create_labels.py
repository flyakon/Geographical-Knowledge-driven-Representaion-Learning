'''
@Project : RepresentationLearningLand 
@File    : create_labels.py
@Author  : Wenyuan Li
@Date    : 2020/10/6 17:22 
'''

from GeoKR.utils import gdal_utils,path_utils,global_land_utils
import os
import glob
from skimage import io
import tqdm
import shutil
import numpy as np
import sys
import argparse
import json
import cv2

import multiprocessing


parse=argparse.ArgumentParser()

parse.add_argument('--land_path',type=str,default=r'../../dataset/GlobalLand30/2020')
parse.add_argument('--data_path',type=str,
                   default=r'../..\dataset\Levir-KR\crop_data')
parse.add_argument('--result_path',type=str,
                   default=r'../../dataset\Levir-KR\RSIData_WithLabels')

def process(img_files,result_path,globalLandUtils:global_land_utils.GlobalLand,with_vis=False,
            vis_step=100,thread_id=0):
    if thread_id==0:
        img_files=tqdm.tqdm(img_files)
    i=0
    nomatching_list=[]
    full_sea_list=[]
    for img_file in img_files:
        i+=1
        file_name=path_utils.get_filename(img_file,is_suffix=False)
        parent_folder=path_utils.get_parent_folder(img_file,with_root=False)
        result_folder=os.path.join(result_path,parent_folder)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        result_file = os.path.join(result_folder, '%s.json' % file_name)
        if os.path.exists(result_file):
            continue
        # print(gdal_utils.get_geoTransform(img_file))
        label_map = globalLandUtils.find_create_labels_map(img_file)
        if label_map is not None:
            class_list,class_count=np.unique(label_map,return_counts=True)
            if len(class_list)==1 and class_list[0]==255:
                full_sea_list.append(img_file)
                continue
            # print(class_list)
            if thread_id==0 and with_vis and i % vis_step==0:
                result_file = os.path.join(result_folder, '%s.tiff' % file_name)
                if os.path.exists(result_file):
                    continue
                label_map=globalLandUtils.vis_labels(label_map)
                label_map = label_map.astype(np.uint8)
                img=gdal_utils.read_full_image(img_file,as_rgb=True,data_format='NUMPY_FORMAT',normalize=False)
                # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img=img.astype(np.uint8)

                io.imsave(result_file,img,check_contrast=False)
                result_file = os.path.join(result_folder, '%s_label.jpg' % file_name)
                height,width=img.shape[0:2]
                label_map=cv2.resize(label_map,(height,width))
                io.imsave(result_file, label_map,check_contrast=False)

            labels_dict={}
            for cls in globalLandUtils.class_names:
                labels_dict[cls]=0

            for cls,count in zip(class_list,class_count):
                if cls>=70 and cls<80:
                    cls=70
                if cls not in globalLandUtils.class_idx:
                    continue
                class_idx=globalLandUtils.class_idx.index(cls)
                class_name=globalLandUtils.class_names[class_idx]
                if class_name in labels_dict.keys():
                    labels_dict[class_name]=int(count)

            result_file = os.path.join(result_folder, '%s.json' % file_name)
            fp=open(result_file,'w')
            json.dump(labels_dict,fp,ensure_ascii=False,indent=4)
            fp.close()
        else:
            nomatching_list.append(img_file)

    fp=open(os.path.join(result_path,'%s_%d.txt'%('no_matching',thread_id)),'w')
    fp.writelines('%d/%d\n'%(len(nomatching_list),len(img_files)))
    for line in nomatching_list:
        fp.writelines('%s\n'%line)
    fp.close()

    fp = open(os.path.join(result_path, '%s_%d.txt' % ('full_sea', thread_id)), 'w')
    fp.writelines('%d/%d\n' % (len(full_sea_list), len(img_files)))
    for line in full_sea_list:
        fp.writelines('%s\n' % line)
    fp.close()

if __name__=='__main__':
    args = parse.parse_args()

    land_path=args.land_path
    data_path=args.data_path
    data_format=r'*/*.tiff'
    with_vis=True
    vis_step=100
    skip_exists=True
    result_path=args.result_path
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    globalLandUtils=global_land_utils.GlobalLand(land_path,'*/*.tif')

    img_files=glob.glob(os.path.join(data_path,data_format))
    # np.random.shuffle(img_files)
    if skip_exists:
        tmp_list = []
        for data_file in img_files:
            file_name=path_utils.get_filename(data_file,is_suffix=False)
            parent_folder = path_utils.get_parent_folder(data_file, with_root=False)
            result_folder = os.path.join(result_path, parent_folder)
            result_file = os.path.join(result_folder, '%s.json' % file_name)
            if os.path.exists(result_file):
                continue
            tmp_list.append(data_file)
        data_files = tmp_list
    else:
        data_files=img_files
    print('image count:%d' % len(data_files))
    num_thread = 4
    # process(data_files, result_path,globalLandUtils,with_vis,vis_step, 0)
    train_data_process = multiprocessing.Pool(num_thread)
    for i in range(num_thread):
        start_idx = int(i * len(data_files) / num_thread)
        end_idx = int((i + 1) * len(data_files) / num_thread)
        train_data_process.apply_async(process, (data_files[start_idx:end_idx], result_path,
                                                 globalLandUtils,with_vis,vis_step, i))

    train_data_process.close()
    train_data_process.join()




