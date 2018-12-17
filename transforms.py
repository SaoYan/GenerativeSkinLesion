import random
from PIL import Image
import torchvision.transforms.functional as trF

#----------------------------------------------------------------------------
# Re-write transforms for our own use
# reference: official code https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
#----------------------------------------------------------------------------

class RatioCenterCrop(object):
    def __init__(self, ratio=1.):
        assert ratio <= 1. and ratio > 0
        self.ratio = ratio

    def __call__(self, image):
        width, height = image.size
        new_size = self.ratio * min(width, height)
        img = trF.center_crop(image, new_size)
        return img

class RandomRotate(object):
    def __init__(self, resample=False, expand=False, center=None):
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        idx = random.randint(0,3)
        angle = idx * 90
        return angle

    def __call__(self, image):
        angle = self.get_params()
        img = trF.rotate(image, angle, self.resample, self.expand, self.center)
        return img
