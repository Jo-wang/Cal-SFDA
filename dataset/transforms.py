import random
import numpy as np
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as torch_tr
from scipy.ndimage.interpolation import shift

from skimage.segmentation import find_boundaries

try:
    import accimage
except ImportError:
    accimage = None
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if isinstance(img, tuple):
            img = img[0]
        w, h = img.size
        if (w == self.size[0] and h == self.size[1]):
            return img
        return img.resize(self.size, Image.BICUBIC)     

class ResizeLabel(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if isinstance(img, tuple):
            img = img[0]
        w, h = img.size
        if (w == self.size[0] and h == self.size[1]):
            return img
        return img.resize(self.size, Image.NEAREST)

class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        if isinstance(img, tuple):
            img, wei = img
            return (torch.from_numpy(np.array(img, dtype=np.int32)).long(), torch.from_numpy(np.array(wei, dtype=np.float32)).float())
        else:
            return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class ResizeHeight(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.target_h = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        target_w = int(w / h * self.target_h)
        return img.resize((target_w, self.target_h), self.interpolation)


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))



class ToTensor(object):
    def __call__(self, img):
        img = torch.from_numpy(img)
        return img



class SubMean(object):
    def __init__(self, mean=np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)):
        self.mean = mean
        
    def __call__(self, img):
        img = np.asarray(img, np.float32)
        img -= self.mean
        img = img.transpose((2, 0, 1)) 
        return img



class CRST_trans(object):
    def __init__(self, mean=np.array((0.406, 0.456, 0.485), dtype=np.float32) , std=np.array((0.225, 0.224, 0.229), dtype=np.float32)):
        self.mean = mean
        self.std  = std
    def __call__(self, img):
        img = np.asarray(img, np.float32)
        img = img/255.0
        img -= self.mean
        img = img/self.std
        img = img[:, :, ::-1]
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img.copy())



class RandomGaussianBlur(object):
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))



class RandomBilateralBlur(object):
    def __call__(self, img):
        sigma = random.uniform(0.05,0.75)
        blurred_img = denoise_bilateral(np.array(img), sigma_spatial=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def adjust_brightness(img, brightness_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = torch_tr.Compose(transforms)

        return transform

    def __call__(self, img):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)
