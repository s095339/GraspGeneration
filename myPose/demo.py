from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import numpy as np
import os
import cv2

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
from lib.dataset.grasp_test_dataset import grasp_test_dataset

from lib.PoseGrasp.scene import ImageScene, LabelScene

from lib.utils.transform import create_homog_matrix, create_rot_mat_axisAlign
import torch
import torch.utils.data
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  #local param
  PATH = "./image"
  if not os.path.exists(PATH):
    os.mkdir(PATH)
  opt.depth = True
  #
  opt.debug = 0
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  #opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  #mycode====================
  #opt.num_classes = 2
  #==========================

  data_dir_pth = "./lib/data/foup_grasp_dataset"
  Dataset = grasp_test_dataset(data_dir_pth, opt)
  
  dist_th_list = [0.02,0.03,0.04]
  angle_th_list = [np.pi/12, np.pi/8, np.pi/4]

  id = 0
  for dist_th in dist_th_list:
    angle_th = angle_th_list[id]
    for iter in range(len(Dataset)):
      batch = Dataset[iter]
      meta, labelscene = batch
      #labelscene.scene_imshow(rotate=True)
      image = meta['image']
    
      
      imagescene = detector.run(image,meta)

      imagescene.scene_info_show()
      imagescene.scene_imshow()
      

    id+=1

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
