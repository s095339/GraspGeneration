from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths



import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#
from lib.dataset.Objectron_dataset import Objectron_dataset
from lib.models.model import *
from lib.opts import opts



if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir_pth = "./lib/data/Objectron/outf_all/camera_train"
    opt = 0
    dataset = Objectron_dataset(data_dir_pth,opt,"train" )
    opt = opts()
    print(opt)
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = create_model(arch = 'dlav0_34')
    opt.heads = {'ct_hm': opt.num_classes, 'bbox_wh': 2, 'kps_sbpxl_dis': 1}


    print("end")