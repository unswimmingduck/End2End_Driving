from torch.utils.data import Dataset
import json
import cv2
import numpy as np
import os
from torchvision import transforms
import torch

class Dockey_Dataset(Dataset):
    def __init__(self, info_path) -> None:
        super(Dockey_Dataset, self).__init__()
        self.info_path = info_path
        self.data_info = self.get_info()
    
    def get_info(self):
        f = open(os.path.join(self.info_path, "data_info.json"))
        result = json.loads(f.read())
        return result

    def data_augment(self, img, angle):
        # flipping the img and angle
        flip_flag = np.random.choice([0, 1])
        if flip_flag:
            img, angle = cv2.flip(img, 1), -angle

        # color space conversion and random brightness adjustment
        color_flag = np.random.choice([0, 1])
        if color_flag:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            alpha = np.random.uniform(low = 0.1, high = 0.5, size = None)
            v = hsv[:, :, 2]
            v = v * alpha
            hsv[:, :, 2] = v.astype('uint8')
            img = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
        return img, angle

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        # reading the data
        img = cv2.imread(self.data_info[str(index)]['img'])
        angle = float(self.data_info[str(index)]['dir'])
        # data augment
        img, angle = self.data_augment(img, angle)

        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(img).cuda()
        angle = torch.from_numpy(np.array([angle])).float().cuda()

        return tensor_img, angle
        