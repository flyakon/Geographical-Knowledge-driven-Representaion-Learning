'''
@Project : RepresentationLearningLand
@File    : crop_image.py
@Author  : Wenyuan Li
@Date    : 2020/10/6 13:59
'''
import glob
import GeoKR.utils.gdal_utils as gdal_utils
from GeoKR.utils import path_utils
import os
import numpy as np
import cv2
import tqdm
# from skimage import exposure
import threading
import multiprocessing
from skimage import exposure

def read_block_img(img_file,x,y,crop_size,im_width,im_height,scale,as_rgb=True,normalize=True,normalize_factor=4):
    if x+crop_size>=im_width:
        x=im_width-crop_size
    if y+crop_size>=im_height:
        y=im_height-crop_size

    img=gdal_utils.read_image(img_file,x,y,crop_size,crop_size,scale_factor=scale,as_rgb=as_rgb,
                              normalize=normalize,normalize_factor=normalize_factor)
    return img

def check_img(img):

    img=np.mean(img,axis=0)
    max_value=np.max(img)
    ratios=np.count_nonzero(img<20)/img.size
    if max_value < 20 or ratios > 0.1:
        return False
    if exposure.is_low_contrast(img):
        return False
    cloud_img=img
    cloud_ratio=np.count_nonzero(cloud_img>230)/cloud_img.size
    if cloud_ratio>0.5:
        return False
    return True

def process(data_files,result_folder,crop_size = 512,scale = 1,
            normalize_factor=4,options=None,thred_id=0,crop_step=400):
    crop_size=int(crop_size*scale)
    crop_step = crop_step
    total_count=0
    for data_file in data_files:
        img_height, img_width, _ = gdal_utils.get_image_shape(data_file)
        total_count+=img_width * img_height / crop_step / crop_step
    if thred_id==0:
        pbar = tqdm.tqdm(range(int(total_count)))
        print('image count:%d' % len(data_files))
    if normalize_factor<1:
        percent_normalize=True
        factor = int(normalize_factor * 100)
        # print(percent_normalize,factor)
    else:
        percent_normalize=False

    for data_file in data_files:
        if percent_normalize:

            img=gdal_utils.read_full_image(data_file,scale_factor=20,as_rgb=False,normalize=False,
                                           data_format='NUMPY_FORMAT')
            # print(gdal_utils.get_image_shape(data_file))
            # print(gdal_utils.get_geoTransform(data_file))
            # print(img.max(),img.min())
            mask = np.mean(img, axis=-1)
            min_value = np.percentile(img[mask > 0], factor)
            max_value = np.percentile(img[mask > 0], 100-factor)
            normalize_factor=[min_value,max_value]
        file_name = path_utils.get_filename(data_file, is_suffix=False)
        file_name=file_name.replace('_GZ','')
        result_path = os.path.join(result_folder, path_utils.get_parent_folder(data_file,with_root=False))
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        im_height, im_width, _ = gdal_utils.get_image_shape(data_file)
        geo_transforms = gdal_utils.get_geoTransform(data_file)
        # if file_name.startswith('GF1_'):
        #     normalize_factor=4
        # elif file_name.startswith('GF6_'):
        #     normalize_factor=16
        # else:
        #     normalize_factor=4
        for x in range(0, im_width, crop_step):
            for y in range(0, im_height, crop_step):
                if thred_id == 0:
                    pbar.update(1)
                result_file = os.path.join(result_path, '%s_%d_%d.tiff' % (file_name, x, y))
                if os.path.exists(result_file):
                    continue
                img = read_block_img(data_file, x, y, crop_size, im_width, im_height, scale=scale,
                                     as_rgb=False,normalize=True,normalize_factor=normalize_factor)
                # img=img[[3,2,1],:,:]

                if not check_img(img):
                    continue
                _, crop_height, crop_width = img.shape
                assert crop_height == crop_size / scale and crop_width == crop_size / scale
                crop_geotransforms = []
                crop_geotransforms.extend(
                    [geo_transforms[0] + geo_transforms[1] * x+geo_transforms[2]*y,
                     geo_transforms[1] * scale,
                     geo_transforms[2]* scale,
                     geo_transforms[3] + geo_transforms[4]*x+ geo_transforms[5] * y,
                     geo_transforms[4]* scale,
                     geo_transforms[5] * scale
                     ]
                )
                gdal_utils.save_full_image(result_file, img, geoTranfsorm=crop_geotransforms,
                proj=gdal_utils.get_projection(data_file),options=options)


if __name__=='__main__':

    data_path=r'../../dataset/Levir-KP/source_data'
    data_format='*/.tiff'
    result_path=r'../../dataset/Levir-KP/crop_data'

    options = ["TILED=TRUE", "COMPRESS=DEFLATE","NUM_THREADS=2","ZLEVEL=9"]
    crop_size = 256
    crop_step=200
    scale = 1
    normalize_factor = 0.01
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if os.path.isdir(data_path):
        data_files=glob.glob(os.path.join(data_path,data_format))
    else:
        data_files=np.loadtxt(data_path,dtype=np.str).tolist()

    tmp_list=[]
    for data_file in data_files:
        file_name=path_utils.get_filename(data_file,is_suffix=False)
        if os.path.exists(os.path.join(result_path,path_utils.get_parent_folder(data_file,with_root=False))):
            continue
        tmp_list.append(data_file)
    data_files=tmp_list
    print('image count:%d' % len(data_files))
    num_thread = 4
    # process(data_files, result_path,crop_size,scale,normalize_factor,options=options)
    train_data_process = multiprocessing.Pool(num_thread)
    for i in range(num_thread):
        start_idx = int(i * len(data_files) / num_thread)
        end_idx = int((i + 1) * len(data_files) / num_thread)
        train_data_process.apply_async(process, (data_files[start_idx:end_idx],result_path,
                                                 crop_size,scale,normalize_factor,options,i,crop_step))

    train_data_process.close()
    train_data_process.join()





