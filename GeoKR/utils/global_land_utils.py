import glob
import os
from RepresentationLearningLand.utils import path_utils,gdal_utils
import tqdm
import pickle
import numpy as np
from typing import List
import tqdm
from skimage import io

class SingleGlobalLand(object):
    def __init__(self,img_file):
        self.parse_args(img_file)
        self.img_file = img_file

    def parse_args(self,data_file):
        '''
        根据文件命名规则确定该区域的zone等信息
        :param data_file:
        :return:
        '''
        file_name=path_utils.get_filename(data_file,is_suffix=False)
        self.data_file=data_file
        self.south=False if file_name[0]=='n' else True
        self.zone_id=int(file_name[1:3])
        self.begin_lat=int(file_name[4:6])
        if abs(self.begin_lat)<85:
            self.proj='utm'
        else:
            self.proj='aeqd'

        self.year=file_name[7:11]

    def __repr__(self):
        return  '%s'%self.data_file

    def __eq__(self, other):
        if isinstance(other,SingleGlobalLand):
            return super(self,SingleGlobalLand).__eq__(other)
        lon,lat=other
        if abs(lat)<=60:
            zone_id = lon // 6 + 31
        elif abs(lat)<85:
            zone_id = lon // 6 + 31
            if zone_id %2==0:
                zone_id=zone_id-1
        else:
            zone_id=0
        south = False if lat> 0 else True
        is_eq=True
        if zone_id!=self.zone_id:
            is_eq=False
            return is_eq
        if south!=self.south:
            is_eq=False
            return is_eq
        if not (abs(lat)>=self.begin_lat and abs(lat)<=self.begin_lat+5):
            is_eq=False
        return is_eq

    def __repr__(self):
        return self.img_file

    def __crop_topleft(self,xmin,ymin,xmax,ymax):
        if xmax>self.im_width:
            xmax=self.im_width
        if ymax>self.im_height:
            ymax=self.im_height
        if ymax>ymin and xmax>xmin:

            img=gdal_utils.read_image(self.img_file,xmin,ymin,xmax-xmin,ymax-ymin,as_rgb=False,data_format='NUMPY_FORMAT',
                                      normalize=False)
            return img
        else:
            return None

    def __crop_topright(self,x,y):
        img=gdal_utils.read_image(self.img_file,0,y,x,self.im_height-y,as_rgb=False)
        return img

    def __crop_bottomright(self,x,y):
        img=gdal_utils.read_image(self.img_file,0,0,x,y,as_rgb=False)
        return img

    def __crop_bottomleft(self,x,y):
        img=gdal_utils.read_image(self.img_file,x,0,self.im_width-x,y,as_rgb=False)
        return img

    def create_label(self,img_file):
        '''
        根据输入的图像生成该区域的地表标签
        :param img_file:
        :return:
        '''
        self.geo_tranforms = gdal_utils.get_geoTransform(self.img_file)
        self.im_height, self.im_width, _ = gdal_utils.get_image_shape(self.img_file)
        target_geotransform=gdal_utils.get_geoTransform(img_file)
        if self == (target_geotransform[0],target_geotransform[3]):

            target_height,target_width,_=gdal_utils.get_image_shape(img_file)
            x,y=gdal_utils.convt_geo(target_geotransform[0],target_geotransform[3],proj=self.proj,
                                     zone_id=self.zone_id,south=self.south,inverse=False,lat_0=-90,lon_0=0)
            #首先计算目标图像在标签中的起始位置
            xmin=(x-self.geo_tranforms[0])/self.geo_tranforms[1]
            ymin=(y-self.geo_tranforms[3])/self.geo_tranforms[5]

            x, y = gdal_utils.convt_geo(target_geotransform[0]+target_geotransform[1]*target_width,
                                        target_geotransform[3]+target_geotransform[5]*target_height,
                                        proj=self.proj,
                                        zone_id=self.zone_id, south=self.south, inverse=False,lat_0=-90,lon_0=0)
            # 首先计算目标图像在标签中的起始位置
            xmax = (x - self.geo_tranforms[0]) / self.geo_tranforms[1]
            ymax = (y - self.geo_tranforms[3]) / self.geo_tranforms[5]
            return self.__crop_topleft(xmin,ymin,xmax,ymax)
        else:
            raise NotImplementedError

    def get_info(self):
        img=gdal_utils.read_full_image(self.img_file,as_rgb=False,data_format='NUMPY_FORMAT',
                                       normalize_factor=False,scale_factor=20)
        im_height,im_width,_=gdal_utils.get_image_shape(self.img_file)
        transforms=gdal_utils.get_geoTransform(self.img_file)
        xmin,ymin=gdal_utils.convt_geo(transforms[0],transforms[3],
                                       zone_id=self.zone_id,south=self.south,
                                       proj=self.proj,lat_0=-90,lon_0=0)
        xmax=transforms[0]+transforms[1]*im_width
        ymax=transforms[3]+transforms[5]*im_height
        xmax, ymax = gdal_utils.convt_geo(xmax,ymax, proj=self.proj,
                                          zone_id=self.zone_id, south=self.south,lat_0=-90,lon_0=0)
        return xmin,ymin,xmax,ymax,img

class GlobalLand(object):

    def __init__(self,file_path,file_format):
        self.data_list=[]
        self.load_data(file_path,file_format)
        self.class_names=['Cultivated_Land', 'Foreast','Grassland','Shrubland','Wetland','Waterbodies','Tundra',
                          'Artifical_Surfaces','Bareland','Permanent_Snow','sea']
        self.class_idx=[10,20,30,40,50,60,70,80,90,100,255]
        self.color_map=[(249,160,255),(0,99,0),(99,255,0),(0,255,199),(0,99,255),(0,0,255),(99,99,51),
                        (255,0,0),(191,191,191),(198,239,255),(0,198,255)]






    def load_data(self,file_path,file_format,tmp_file='globalland30.pickle'):
        img_files=glob.glob(os.path.join(file_path,file_format))
        for img_file in tqdm.tqdm(img_files):
            self.data_list.append(SingleGlobalLand(img_file))

    def find_matching_files(self,data_file)->List[SingleGlobalLand]:
        '''
        find files of GlobalLand30 that are intersection with data_file
        :param data_file:
        :return:
        '''

        geo_transforms = gdal_utils.get_geoTransform(data_file)
        im_height,im_width,_=gdal_utils.get_image_shape(data_file)
        lon,lat=geo_transforms[0],geo_transforms[3]
        matching_files=[x for x in self.data_list if x==(lon,lat)]
        return matching_files


    def vis_labels(self,label_map):
        height,width=label_map.shape[0:2]
        result_map=np.zeros([height,width,3],dtype=np.uint8)
        for i,idx in enumerate(self.class_idx):
            result_map = np.where(label_map[:, :, np.newaxis] == idx, self.color_map[i], result_map)
        return result_map

    def find_create_labels_map(self,img_file,result_file=None):
        matching_files=self.find_matching_files(img_file)
        if len(matching_files)>0:
            matching_file=matching_files[0]
            img=matching_file.create_label(img_file)
            if result_file is not None:
                gdal_utils.save_full_image(result_file,img,geoTranfsorm=gdal_utils.get_geoTransform(img_file),
                                           proj=gdal_utils.get_projection(img_file),data_format='NUMPY_FORMAT')
            return img
        else:
            #print('No matching file found')
            return None


    def vis_labels(self,label_map):
        vis_img=np.zeros_like(label_map)
        total_cls=np.unique(label_map)
        for cls_idx in total_cls:
            if cls_idx==0:
                continue
            elif cls_idx>=70 and cls_idx<80:
                cls_idx=70
            idx=self.class_idx.index(cls_idx)
            color=self.color_map[idx]
            vis_img=np.where(label_map==cls_idx,color,vis_img)
        return vis_img



if __name__=='__main__':
    img_path=r'I:\experiments\representation_learning_rsi\dataset\GlobalLand30'
    img_format='*/*.tif'
    search_file=r'F:\资源中心目标检测数据_20190510\GF1\GF1_PMS1_E106.6_N29.7_20180807_L1A0003376167_FUSION_GEO.tiff'
    model=GlobalLand(img_path,img_format)
    label_map=model.find_create_labels_map(search_file,'../../dataset/label.tif')
    vis_img=model.vis_labels(label_map)
    io.imsave('../../dataset/vis.tif',vis_img)
