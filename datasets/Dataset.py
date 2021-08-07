import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import PIL.ImageOps
from functools import partial

from .otherTransforms import RandomErasing


def flip_channel(ts):
    idx = torch.arange(ts.size(0)-1,-1,step=-1,device=ts.device)
    return ts.index_select(index=idx.long(), dim=0)

def expand_to_three_channel(ts, size):
    return ts.expand(3, size, size)


class SbirDataset(Dataset):
    def __init__(self, dataset, data_path="datasets/data", image_size=224):

        self.dataset = dataset
        self.image_size = image_size
        self.samples = self._make_samples(dataset, data_path)
        self._make_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_fn, label = self.samples[index]
        img = Image.open(img_fn)
        img = self._transform(img)

        return img, label

    def _make_samples(self, dataset, data_path):

        file_dir = os.path.abspath(__file__).rsplit(os.sep, 1)[0]
        file_path = os.path.join(file_dir, "anno", dataset + '_' + self.file_path)

        samples, self.cate_num = [], {}
        with open(file_path, 'r') as f:
            for img_fn, label in json.load(f)[self.phase]:
                img_fn = os.path.join(*img_fn.split('/')) # compatible for windows
                samples.append((os.path.join(data_path, dataset, img_fn), label))
                self.cate_num[label] = self.cate_num.get(label, 0) + 1
        return samples

    def _make_transform(self):
        raise NotImplementedError

    def get_loader(self, *args, **kwargs):
        return DataLoader(self, pin_memory=True, **kwargs)


class SketchDataset(SbirDataset):
    file_path = 'sketch.json'
    def __init__(self, dataset, data_path="datasets/data", image_size=224, phase="test"):
        self.phase = phase
        super(SketchDataset, self).__init__(dataset, data_path, image_size)

    def _make_transform(self):
        size = self.image_size
        if self.phase == "test":
            trans = [transforms.Resize((int(size*1.1), int(size*1.1))),
                     transforms.CenterCrop(size)]
        elif self.phase == "train":
            trans = [transforms.Grayscale(),
                     transforms.Lambda(PIL.ImageOps.invert),
                     transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomRotation(90, Image.BICUBIC),
                     RandomErasing(0.6, (0.02, 0.2), 1),
                     transforms.Lambda(PIL.ImageOps.invert)]
        trans.append(transforms.ToTensor())
        trans.append(partial(expand_to_three_channel, size=size))
        self._transform = transforms.Compose(trans)


class PhotoDataset(SbirDataset):
    file_path = 'photo.json'
    def __init__(self, dataset, data_path="datasets/data", image_size=224, phase="test"):
        self.phase = phase
        super(PhotoDataset, self).__init__(dataset, data_path, image_size)


    def _make_transform(self):
        size = self.image_size
        trans = [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Lambda(partial(expand_to_three_channel, size=size))
        ]
        if self.dataset == "TUBerlin":
            trans.append(transforms.Lambda(flip_channel))
        if self.phase == "train":
            trans.insert(1, transforms.RandomHorizontalFlip(),)
        self._transform = transforms.Compose(trans)


def get_dataset(domain="sketch", phase="train", **kwargs):
    if domain == 'sketch':
        return SketchDataset(phase=phase, **kwargs)
    elif domain == 'photo':
        return PhotoDataset(phase=phase, **kwargs)
    else:
        raise ValueError('domain: %s, phase: %s is invalid.' % (domain, phase))
