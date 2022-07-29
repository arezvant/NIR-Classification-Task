# python libraties
import cv2, itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
from statistics import mean
from configuration import config
from tifffile import TiffFile
from matplotlib.pyplot import figure
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

class RetiSpecDataset():

    def __init__(self, dir, flag=None, transform=None):
        self.dir = dir
        self.flag = flag
        self.transform = transform
        file_list = glob(self.dir + "/*")
        self.data = []

        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob(class_path + "/*.tif"):
                self.data.append([img_path, class_name])

        self.class_map = {"Forest" : 0, "River": 1}
        self.img_dim = (config.INPUT_HEIGHT, config.INPUT_WIDTH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = TiffFile(img_path).asarray().astype(np.float32)
        img_n = img/(np.max(img))
        img = cv2.resize(img_n, self.img_dim)

        if self.flag == 'rgb':
          class_id = self.class_map[class_name]
          img_RGB = img[:, :, :3]
          if self.transform:
            img_RGB = self.transform(img_RGB)
          img_RGB_tensor = torch.from_numpy(img_RGB)
          img_RGB_tensor = img_RGB_tensor.permute(2, 0, 1)
          class_id = torch.tensor([class_id])
          return img_RGB_tensor.float(), class_id

        elif self.flag == 'nir':
          class_id = self.class_map[class_name]
          img_N = img[:, :, 3]
          if self.transform:
            img_N = self.transform(img_N)
          img_N = np.repeat(img_N[..., np.newaxis], 3, -1)
          img_N_tensor = torch.from_numpy(img_N)
          img_N_tensor = img_N_tensor.permute(2, 0, 1)
          return img_N_tensor.float(), class_id

        elif self.flag == 'mix':
          class_id = self.class_map[class_name]
          c = np.random.randint(3, size=1)[0]
          img[:,:,c] = img[:,:,c] * 0
          img_mix = img
          if self.transform:
            img_mix = self.transform(img_mix)
          img_mix_tensor = torch.from_numpy(img_mix)
          img_mix_tensor = img_mix_tensor.permute(2, 0, 1)
          return img_mix_tensor.float(), class_id

        else:
          class_id = self.class_map[class_name]
          if self.transform:
            img_n = self.transform(img_n)
          img_tensor = torch.from_numpy(img_n)
          img_tensor = img_tensor.permute(2, 0, 1)
          class_id = torch.tensor([class_id])
          return img_tensor.float(), class_id
