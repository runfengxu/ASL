from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import scipy.io as scio
import torch


def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    return img


class EyeBrowDataset(Dataset):
    def __init__(self, mat_file, img_folder, window_size, transform=None):
        self.data = scio.loadmat(mat_file)
        print(self.data.keys())
        self.img_folder = img_folder
        self.transform = transform
        self.label = self.data['gt'][0]
        self.all_path = self.data['img_file']
        self.transform_img = transforms.Compose([self.transform])
        self.window_size = window_size

    def __getitem__(self, idx):

        #gt_height = (self.label[idx] + 2) / 4

        gt_height = self.label[idx]
        window_path_list = self.all_path[idx]

        if gt_height == 0.9:
            print(window_path_list)
            print(gt_height)

        imgs = []
        for i in range(self.window_size):
            img_path = self.img_folder + window_path_list[i].split()[0]
            pic = load_img(img_path)
            pic = self.transform_img(pic)  # 3 * 224 * 224
            # print(pic.shape)
            imgs.append(pic.unsqueeze(0))
        imgseq = torch.cat((imgs[0:self.window_size]), 0).squeeze()
        # gseq.shape)

        # sample = {'image':imgseq,'gt_height':gt_height}

        return imgseq, gt_height,self.all_path[idx]



    # return 0
    def __len__(self):
        return len(self.label)

# dataset = EyeBrowDataset('/dresden/users/rx43/ASL/eyebrow_detecting/dataset/train_all.mat','/dresden/users/rx43/ASL/eyebrow_detecting/',7,
# 	transform = transforms.Compose([transforms.ToTensor(),
# 	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# 	)
#
# print(dataset[15][0].shape)
