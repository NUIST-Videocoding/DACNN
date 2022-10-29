import torch
import torchvision
import os
import torch.utils.data as data
from PIL import Image
import random

class LSIQAfolder(data.Dataset):
    def __init__(self, root, index_1, index_2, transform):
        # print(index_1)
        # print(index_2)

        dst_txtpath = os.path.join(root, 'data_info_IQABD/dst_name.txt')
        dst_path_temp = open(dst_txtpath, 'r')
        dst_path = []
        for line in dst_path_temp:
            line = line.split('\n')
            dst_path.append(line[0])
        # print(len(dst_path))#314895

        ref_txtpath = os.path.join(root, 'data_info_IQABD/ref_name.txt')
        ref_path_temp = open(ref_txtpath, 'r')
        ref_path = []
        for line in ref_path_temp:
            line = line.split('\n')
            ref_path.append(line[0])
        # print(len(ref_path))#314895

        mos_txtpath = os.path.join(root, 'data_info_IQABD/pmos.txt')
        mos_temp = open(mos_txtpath, 'r')
        mos = []
        for line in mos_temp:
            line = line.split('\n')
            mos.append(line[0])
        # print(len(mos))#314895

        sample = []
        for i, item in enumerate(index_1):
            # sample.append((os.path.join(root, dst_path[item]), os.path.join(root, ref_path[item]), mos[item]))
            sample.append((os.path.join(root, dst_path[item]), os.path.join(root, dst_path[index_2[i]]),mos[item], mos[index_2[i]]))
            # print(sample)
            # i=0

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        dst_path, ref_path, target_1, target_2 = self.samples[index]
        # print(dst_path)
        # print(ref_path)
        # print(target)

        dst_sample = pil_loader(dst_path)
        dst_sample = self.transform(dst_sample)

        ref_sample = pil_loader(ref_path)
        ref_sample = self.transform(ref_sample)

        return dst_sample, ref_sample, target_1, target_2

    def __len__(self):
        length = len(self.samples)
        return length

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DataLoader(object):
    def __init__(self, path, img_indx_1, img_indx_2, patch_size, batch_size=1, istrain=True):
        self.batch_size = batch_size
        self.istrain = istrain

        # Train transforms
        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        # Test transforms
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])

        self.data = LSIQAfolder(
            root=path, index_1=img_indx_1, index_2= img_indx_2, transform=transforms)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, num_workers = 4)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False, num_workers = 4)
        return dataloader
