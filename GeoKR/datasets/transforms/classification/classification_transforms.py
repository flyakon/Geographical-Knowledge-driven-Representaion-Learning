import torch
import torch.utils.data as data_utils
import os
import numpy as np
from PIL import Image
import torchvision.transforms
import random
import torchvision.transforms.functional as F
import cv2
import sys
import collections
import numbers
import datetime

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

class VerticalFlip():
    """Vertically flip the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
            bnd_boxes (ndarray): [Nx4] bndboxes (xmin,ymihn,xmax,ymax)
        Returns:
            PIL Image: Randomly flipped image.
            ndarrsy: [Nx4] Randomly flipped bnndboxes
        """

        if random.random() < self.p:
            img=F.vflip(img)
        return img


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class HorizontalFlip():
    """Horizontally flip the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
            bnd_boxes (ndarray): [Nx4] bndboxes (xmin,ymihn,xmax,ymax)
        Returns:
            PIL Image: Randomly flipped image.
            ndarrsy: [Nx4] Randomly flipped bnndboxes
        """

        if random.random() < self.p:
            img=F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Rotate():
    """Rotate the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
            angle (int): Rotated angle [0,90,180,270]
        """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
            bnd_boxes (ndarray): [Nx4] bndboxes (xmin,ymihn,xmax,ymax)
        Returns:
            PIL Image: Randomly flipped image.
            ndarrsy: [Nx4] Randomly flipped bnndboxes
        """
        angles=[0,90,180,270]
        idx=random.randint(0,3)
        self.angle=angles[idx]

        if random.random() < self.p:
            img=F.rotate(img,self.angle,expand=True)
        return img




    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop():
    """Horizontally flip the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """
    def __init__(self,crop_ratio_min=0.8,crop_ratio_max=0.95):
        self.crop_ratio_min=crop_ratio_min
        self.crop_ratio_max=crop_ratio_max

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self,img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        img_height=img.height
        img_width=img.width
        self.crop_ratio=random.random()
        self.crop_ratio=self.crop_ratio*(self.crop_ratio_max-self.crop_ratio_min)+self.crop_ratio_min
        height=int(img_height*self.crop_ratio)
        width=int(img_width*self.crop_ratio)
        i, j, h, w = self.get_params(img,(height,width))
        img=F.crop(img, i, j, h, w)
        return img

class RandomApply(object):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        self.transforms=transforms
        self.p = p

    def __call__(self, img):

        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

class RandomOrder(object):
    """Apply a list of transformations in a random order
    """
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)

        for i in order:
            img = self.transforms[i](img)
        return img

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):

        for t in self.transforms:
            img = t(img)
        return img

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, img):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        img=F.to_tensor(img)

        return img


    def __repr__(self):
        return self.__class__.__name__ + '()'

class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        img=F.resize(img, self.size, self.interpolation)

        return img


    def __repr__(self):

        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        img=transform(img)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

if __name__=='__main__':
    compose_transforms = []
    compose_transforms.append(VerticalFlip())
    compose_transforms.append(Rotate())
    compose_transforms.append(RandomCrop(0.8,1))
    compose_transforms.append(ColorJitter(0.15,(0.75,1.25),(0.5,1.5),(-0.15,0.15)))
    transform=RandomOrder(compose_transforms)
    compose_transforms=[]
    compose_transforms.append(transform)
    compose_transforms.append(Resize((224, 224)))
    compose_transforms.append(ToTensor())

    trans=Compose(compose_transforms)
    sourceimg=cv2.imread(r'../../dataset/cloud/train/GF1_PMS1_E104.0_N41.1_20160314_L1A0001467562-MSS1.jpg')
    labelimg=cv2.imread(r'../../dataset/cloud/train/GF1_PMS1_E104.0_N41.1_20160314_L1A0001467562-MSS1.png',cv2.IMREAD_GRAYSCALE)
    img=np.copy(sourceimg)

    # cv2.imshow('sourceimg',sourceimg)
    # cv2.imshow('sourcelabel', labelimg)

    sourceimage=Image.fromarray(sourceimg)
    sourcelabel=Image.fromarray(labelimg)
    start_time=datetime.datetime.now()
    image,label=trans(sourceimage,sourcelabel)
    end_time=datetime.datetime.now()
    total_time = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
    print(total_time)

    img=image.numpy()*255
    img = np.transpose(img, [1, 2, 0])
    img=img.astype(np.uint8)

    cv2.imshow('convertimg', img)

    img = label.numpy() * 255
    print(np.unique(img))
    img = np.squeeze(img, 0)
    img = img.astype(np.uint8)

    cv2.imshow('convertlabel', img)
    cv2.waitKey()

