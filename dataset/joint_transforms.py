import math
import numbers
from PIL import Image, ImageOps
import numpy as np
import random
from skimage import measure
import math

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, centroid):
        crop_position = None
        for t in self.transforms:
            if t.__class__.__name__=='RandomSizeAndCrop':
                img, mask, crop_position = t(img, mask, centroid)
            else:
                img, mask = t(img, mask)
        return img, mask, crop_position



class RandomCropRec(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        W, H = self.size
        w, h = img.size
        if (W, H) == (w, h):
            return img, mask
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            img = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        img = img.crop(crop)
        mask = mask.crop(crop)
        return img, mask



class RandomCrop(object):
    def __init__(self, size, ignore_index=255, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, mask, centroid=None):

        if isinstance(mask, tuple):
            mask, weight = mask
        else:
            weight = None

        w, h = img.size

        th, tw = self.size
        if w == tw and h == th:
            if weight is None:
                return img, mask
            else:
                return img, (mask, weight)

        if self.nopad:
            if th > h or tw > w:
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                w, h = img.size

        if centroid is not None:
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        if weight is not None:
            img, mask, weight = img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), weight.crop((x1, y1, x1 + tw, y1 + th))
            return img, (mask, weight)
        else:
            self.params = (x1, x1+tw, y1, y1+th)
            return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), (x1, x1+tw, y1, y1+th)

    def get_params(self):
        return self.params

class ResizeHeight(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.target_h = size
        self.interpolation = interpolation

    def __call__(self, img, mask):
        w, h = img.size
        if isinstance(mask, tuple):
            mask, weight = mask
            target_w = int(w / h * self.target_h)
            return (img.resize((target_w, self.target_h), self.interpolation),
                    (mask.resize((target_w, self.target_h), Image.NEAREST), weight.resize((target_w, self.target_h), Image.NEAREST)))
        else:
            target_w = int(w / h * self.target_h)
            return (img.resize((target_w, self.target_h), self.interpolation),
                    mask.resize((target_w, self.target_h), Image.NEAREST))



class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))



class CenterCropPad(object):
    def __init__(self, size, ignore_index=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index

    def __call__(self, img, mask):
        
        assert img.size == mask.size
        w, h = img.size
        if isinstance(self.size, tuple):
                tw, th = self.size[0], self.size[1]
        else:
                th, tw = self.size, self.size
	

        if w < tw:
            pad_x = tw - w
        else:
            pad_x = 0
        if h < th:
            pad_y = th - h
        else:
            pad_y = 0

        if pad_x or pad_y:
            img = ImageOps.expand(img, border=(pad_x, pad_y, pad_x, pad_y), fill=0)
            mask = ImageOps.expand(mask, border=(pad_x, pad_y, pad_x, pad_y),
                                   fill=self.ignore_index)

        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))



class PadImage(object):
    def __init__(self, size, ignore_index):
        self.size = size
        self.ignore_index = ignore_index
        
    def __call__(self, img, mask):
        assert img.size == mask.size
        th, tw = self.size, self.size

        w, h = img.size
        
        if w > tw or h > th :
            wpercent = (tw/float(w))    
            target_h = int((float(img.size[1])*float(wpercent)))
            img, mask = img.resize((tw, target_h), Image.BICUBIC), mask.resize((tw, target_h), Image.NEAREST)

        w, h = img.size
        img = ImageOps.expand(img, border=(0,0,tw-w, th-h), fill=0)
        mask = ImageOps.expand(mask, border=(0,0,tw-w, th-h), fill=self.ignore_index)
        
        return img, mask



class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if isinstance(mask, tuple):
            mask, weight = mask
        else:
            weight = None

        if weight is None:
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(
                    Image.FLIP_LEFT_RIGHT)
            return img, mask
        else:
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), (mask.transpose(Image.FLIP_LEFT_RIGHT), weight.transpose(Image.FLIP_LEFT_RIGHT))
            return img, (mask, weight)



class FreeScale(object):

    def __init__(self, size):
        self.size = tuple(reversed(size))  

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BICUBIC), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BICUBIC), mask.resize(
                (ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BICUBIC), mask.resize(
                (ow, oh), Image.NEAREST)



class ScaleMin(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BICUBIC), mask.resize(
                (ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BICUBIC), mask.resize(
                (ow, oh), Image.NEAREST)



class Resize(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, list):
            self.size = tuple(size)
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        weight = None
        if isinstance(mask, tuple):
            mask, weight = mask

        if (w == h and w == self.size):
            if weight is None:
                 return img, mask
            else:
                return img, (mask, weight)
        if weight is None:
            return (img.resize(self.size, Image.BICUBIC),
                    mask.resize(self.size, Image.NEAREST))
        else:
            return img.resize(self.size, Image.BICUBIC),(mask.resize(self.size, Image.NEAREST), weight.resize(self.size, Image.NEAREST))



class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BICUBIC),\
                    mask.resize((self.size, self.size), Image.NEAREST)

        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BICUBIC), mask.rotate(
            rotate_degree, Image.NEAREST)



class RandomSizeAndCrop(object):

    def __init__(self, size, crop_nopad,
                 scale_min=0.5, scale_max=2.0, ignore_index=0, pre_size=None, rec=False, centroid=None):
        self.rec= rec
        self.size = size
        if rec:
            self.crop = RandomCropRec((self.size*2, self.size))
        else:
            self.crop = RandomCrop(self.size, ignore_index=ignore_index, nopad=crop_nopad)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.pre_size = pre_size

    def __call__(self, img, mask, centroid_tuple=None):

        if isinstance(mask, tuple):
            mask, weight = mask
        else:
            weight = None
        if self.pre_size is None:
            scale_amt = 1.
        elif img.size[1] < img.size[0]:
            scale_amt = self.pre_size / img.size[1]
        else:
            scale_amt = self.pre_size / img.size[0]

        scale_amt *= random.choice([0.5, 0.75, 1.0, 1.25, 1.5])
        w, h = [int(i * scale_amt) for i in img.size]

        if centroid_tuple is not None:
            centroid_num, grid = centroid_tuple
            num_grid = int(math.sqrt(grid))
            row = math.floor(centroid_num/float(num_grid))+1
            col = centroid_num%(num_grid)+1
            grid_w, grid_h = w/float(num_grid), h/float(num_grid)
            centroid = grid_w*(col-0.5), grid_h*(row-0.5)
            centroid = [int(c * scale_amt) for c in centroid]
        else:
            centroid=None

        if weight is None:
            img, mask = img.resize((w, h), Image.BICUBIC), mask.resize((w, h), Image.NEAREST)
        else:
            img  = img.resize((w, h), Image.BICUBIC)
            mask = (mask.resize((w, h), Image.NEAREST), weight.resize((w, h), Image.NEAREST))
        
        if self.rec:
            return self.crop(img, mask)
        else:
            return self.crop(img, mask, centroid)



class SlidingCropOld(object):

    def __init__(self, crop_size, stride_rate, ignore_label):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = ignore_label

    def _pad(self, img, mask):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant',
                      constant_values=self.ignore_label)
        return img, mask

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        mask = np.array(mask)

        if long_size > self.crop_size:
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            img_sublist, mask_sublist = [], []
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    img_sub = img[sy: ey, sx: ex, :]
                    mask_sub = mask[sy: ey, sx: ex]
                    img_sub, mask_sub = self._pad(img_sub, mask_sub)
                    img_sublist.append(
                        Image.fromarray(
                            img_sub.astype(
                                np.uint8)).convert('RGB'))
                    mask_sublist.append(
                        Image.fromarray(
                            mask_sub.astype(
                                np.uint8)).convert('P'))
            return img_sublist, mask_sublist
        else:
            img, mask = self._pad(img, mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            return img, mask



class SlidingCrop(object):

    def __init__(self, crop_size, stride_rate, ignore_label):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = ignore_label

    def _pad(self, img, mask):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant',
                      constant_values=self.ignore_label)
        return img, mask, h, w

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        mask = np.array(mask)

        if long_size > self.crop_size:
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            img_slices, mask_slices, slices_info = [], [], []
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    img_sub = img[sy: ey, sx: ex, :]
                    mask_sub = mask[sy: ey, sx: ex]
                    img_sub, mask_sub, sub_h, sub_w = self._pad(img_sub, mask_sub)
                    img_slices.append(
                        Image.fromarray(
                            img_sub.astype(
                                np.uint8)).convert('RGB'))
                    mask_slices.append(
                        Image.fromarray(
                            mask_sub.astype(
                                np.uint8)).convert('P'))
                    slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
            return img_slices, mask_slices, slices_info
        else:
            img, mask, sub_h, sub_w = self._pad(img, mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            return [img], [mask], [[0, sub_h, 0, sub_w, sub_h, sub_w]]



class ClassUniform(object):
    def __init__(self, size, crop_nopad, scale_min=0.5, scale_max=2.0, ignore_index=0,
                 class_list=[16]):
        self.size = size
        self.crop = RandomCrop(self.size, ignore_index=ignore_index, nopad=crop_nopad)

        self.class_list = class_list

        self.scale_min = scale_min
        self.scale_max = scale_max

    def detect_peaks(self, image):
        from scipy import ndimage
        neighborhood = ndimage.generate_binary_structure(2, 2)
        local_max = ndimage.maximum_filter(image, footprint=neighborhood) == image
        background = (image == 0)
        eroded_background = ndimage.binary_erosion(background, structure=neighborhood,
                                           border_value=1)
        detected_peaks = local_max ^ eroded_background
        return detected_peaks

    def __call__(self, img, mask):
        if isinstance(mask, tuple):
            mask_size = mask[0].size
        else:
            mask_size = mask.size
        assert img.size == mask_size
        scale_amt = random.choice([0.5, 0.75, 1.0, 1.25, 1.5])
        w = int(scale_amt * img.size[0])
        h = int(scale_amt * img.size[1])

        if scale_amt < 1.0:
            if isinstance(mask, tuple):
                img, mask = img.resize((w, h), Image.BICUBIC),(mask[0].resize((w, h),Image.NEAREST), mask[1].resize((w, h),Image.NEAREST))
                return self.crop(img, mask)
            else:
                img, mask = img.resize((w, h), Image.BICUBIC),mask.resize((w, h),Image.NEAREST)
                return self.crop(img, mask)
        else:

            if not isinstance(mask, tuple):
                origw, origh = mask.size
                img_new, mask_new = img.resize((w, h), Image.BICUBIC), mask.resize((w, h), Image.NEAREST)
            else:
                origw, origh = mask[0].size
                img_new, mask_new = img.resize((w, h), Image.BICUBIC), (mask[0].resize((w, h), Image.NEAREST), mask[1].resize((w, h), Image.NEAREST))
            
            interested_class = self.class_list 

            if not isinstance(mask, tuple):
                data = np.array(mask)
            else: 
                data = np.array(mask[0])
                
            map = np.zeros((origh, origw))
            ones = np.ones((origh, origw))
            for class_of_interest in interested_class:

                map += np.where(data == class_of_interest, ones, 0)
            if map.sum()==0:

                return self.crop(img_new, mask_new)
            else:

                window_size = 250
                measure.block_reduce(map, (window_size, window_size), np.sum)

                locs = []
                sums = []
                for x in range(0, map.shape[0] - window_size, window_size):
                    for y in range(0, map.shape[1] - window_size, window_size):
                        current_sum = map[x:x + window_size, y:y + window_size].sum()
                        if current_sum>0:
                            sums.append(current_sum)
                            locs.append((x, y))

                if len(sums)==0:
                    return self.crop(img_new, mask_new)
                ratio = (float(origw) / w, float(origh) / h)
                to_select = min(len(sums), 10) 
                indices = np.argsort(sums)[-to_select:]

                randompick = np.random.randint(to_select)

                y, x = locs[indices[randompick]]
                y, x = int(y * ratio[0]), int(x * ratio[1])
                window_size = window_size * ratio[0]
                cropx = random.uniform(
                    max(0, (x - window_size / 2) - (self.size - window_size)),
                    max((x - window_size / 2), (x - window_size / 2) - (
                        (w - window_size) - x + window_size / 2)))

                cropy = random.uniform(
                    max(0, (y - window_size / 2) - (self.size - window_size)),
                    max((y - window_size / 2), (y - window_size / 2) - (
                        (h - window_size) - y + window_size / 2)))

                return_img = img_new.crop(
                    (cropx, cropy, cropx + self.size, cropy + self.size))
                if not isinstance(mask, tuple):
                    return_mask = mask_new.crop(
                        (cropx, cropy, cropx + self.size, cropy + self.size))
                else:
                    return_mask = mask_new[0].crop((cropx, cropy, cropx + self.size, cropy + self.size)), mask_new[1].crop((cropx, cropy, cropx + self.size, cropy + self.size))
                return (return_img, return_mask)
                    
