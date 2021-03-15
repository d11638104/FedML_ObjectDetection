import glob
import random
import os
import sys
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from utils.augmentations import *
from utils.transforms import *

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'), 
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None, arg=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.device = torch.device("cuda:" + str(arg.gpu) if torch.cuda.is_available() else "cpu")
        self.args = arg

    def __getitem__(self, index):
        
        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes)) # from xywh to yxyx
                bb = []
                for box in bb_targets:
                    x1 = box[2]
                    y1 = box[3]
                    x2 = box[2] + box[4]
                    y2 = box[3] + box[5]
                    bb.append([box[0], box[1], y1, x1, y2, x2])
                bb_targets = torch.FloatTensor(bb)
            except:
                print(f"Could not apply transform.")
                return

        return img_path, img, bb_targets #[:, 0], bb_targets[:, 1:]

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))
        
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        
        # Resize images to input shape
        scale = []
        for img in imgs:
          s = float(img.shape[0]) / self.img_size
          scale.append(s)
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        bb_targets = torch.cat(bb_targets, 0)
        
        return imgs, bb_targets[:, 2:], bb_targets[:, 1], scale

    def __len__(self):
        return len(self.img_files)
    
    def split(self, args, client_num, mode = "train"):
        count = 0
        dataset_dict = {}
        data_num_dict = {}
        local_data_num = int(len(self.img_files) / client_num)
        residual = len(self.img_files) - local_data_num * client_num
        for i in range(client_num):
            if i < residual:
                data_num_dict[i] = local_data_num + 1
            else:
                data_num_dict[i] = local_data_num
            local_count = 0
            local_data = []
            while(1):
                if local_count >= data_num_dict[i]:
                    if mode == "train":
                        dataset_dict[i] = SplitDataset(local_data, multiscale=args.multiscale_training, img_size=args.img_size, transform=AUGMENTATION_TRANSFORMS, args=self.args)
                    else:
                        dataset_dict[i] = SplitDataset(local_data, img_size=args.img_size, multiscale=False, transform=DEFAULT_TRANSFORMS, args=self.args)
                    break
                local_data.append(self.img_files[count])
                count += 1
                local_count += 1
                
        return data_num_dict, dataset_dict

class SplitDataset(Dataset):
    def __init__(self, img_files, img_size=416, multiscale=True, transform=None, args=None):
        self.img_files = img_files

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        

    def __getitem__(self, index):
        
        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes)) # from xywh to yxyx
                bb = []
                for box in bb_targets:
                    x1 = box[2]
                    y1 = box[3]
                    x2 = box[2] + box[4]
                    y2 = box[3] + box[5]
                    bb.append([box[0], box[1], y1, x1, y2, x2])
                bb_targets = torch.FloatTensor(bb)
            except:
                print(f"Could not apply transform.")
                return
                
        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))
        
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        
        # Resize images to input shape
        scale = []
        for img in imgs:
          s = float(img.shape[0]) / self.img_size
          scale.append(s)
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        bb_targets = torch.cat(bb_targets, 0)
        if bb_targets.shape[0] == 0:
            return None, None, None, None
        
        return imgs, bb_targets[:, 2:], bb_targets[:, 1], scale

    def __len__(self):
        return len(self.img_files)