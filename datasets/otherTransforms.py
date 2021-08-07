import random
import math
from PIL import Image
import numpy as np


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: The probability that the Random Erasing operation will be performed.
         scale: (Minimum, Maximum) proportion of erased area against input image.
         asp_ratio: Minimum aspect ratio of erased area. [asp_ratio, 1/asp_ratio]
    """

    def __init__(self, p=0.5, scale=(0.01, 0.1), asp_ratio=0.3, value=0):
        self.p = p
        self.value = value
        self.scale = scale
        self.asp_ratio = asp_ratio

    def __call__(self, img):
        """
        :param img: PIL.Image
        :return: PIL.Image
        """
        if random.uniform(0, 1) > self.p:
            return img
        w, h = img.size
        for attempt in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * w * h
            aspect_ratio = random.uniform(self.asp_ratio, 1. / self.asp_ratio)

            h_occlusion = int(round(math.sqrt(target_area * aspect_ratio)))
            w_occlusion = int(round(math.sqrt(target_area / aspect_ratio)))
            if w_occlusion < w and h_occlusion < h:
                if len(img.getbands()) == 1:
                    rectangle = Image.fromarray(np.uint8(np.ones((w_occlusion, h_occlusion)) * self.value * 255))
                else:
                    raise ValueError
                    rectangle = Image.fromarray(
                        np.uint8(np.ones(w_occlusion, h_occlusion, len(img.getbands())) * self.value * 255))

                random_position_x = random.randint(0, w - w_occlusion)
                random_position_y = random.randint(0, h - h_occlusion)
                img.paste(rectangle, (random_position_x, random_position_y))
                return img
        return img

