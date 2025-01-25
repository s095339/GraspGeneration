
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
from torchvision.io import read_image

import cv2
import numpy as np

import json
import math


from pose_solver.pose_recover_foup import PoseSolverFoup
from pose_solver.cubic_bbox_foup import Axis
from pose_solver.foup_pnp_shell import pnp_shell


class foup_test_dataset():
    
    def __init__(self, data_dir_pth ,opt):
        self.opt = opt

        self.depth = self.opt.depth
        self.test_data_list = []

        self.data_dir_pth = data_dir_pth
        self._data_rng = np.random.RandomState(123)
        
        

        
        self.max_objs = 10
        
        
        self.num_classes = self.opt.num_classes
        self.demo_class = self.opt.train_class

        
        self.get_test_data()
        print("total:",len(self.test_data_list))

    def get_test_data(self):
        print("collecting data from: ",self.data_dir_pth)
        data_dir_list = os.listdir(self.data_dir_pth)
        print(data_dir_list)
        i=0

        data_idx = 0

        for dir_pth in data_dir_list:
            
            print("get testing data from", dir_pth)
            anns_path = os.path.join(dir_pth, "annotation")



            #每一個data_dir
            pth = os.path.join(self.data_dir_pth, anns_path)
            anno_list = [img for img in os.listdir(pth) if img.endswith(".json")]
            #print(imglist)

            
            for anno in anno_list:
                with open(os.path.join(pth,anno)) as f:
                    anns = json.load(f)
                
                color_path = os.path.join(self.data_dir_pth,dir_pth, anns["color_data_path"])
                if self.depth:
                    depth_path = os.path.join(self.data_dir_pth,dir_pth, anns["depth_data_path"])
                else:
                    depth_path = None
                
                test_data_dict = {
                    "annotation": os.path.join(pth,anno),
                    "color_img": color_path,
                    "depth_img": depth_path
                }

                data_idx+=1
                if(data_idx % (self.opt.spilt_ratio+1) == 0):
                    self.test_data_list.append(test_data_dict)
                
                #抓出裡面所有的.png檔案



    def __len__(self):
        return len(self.test_data_list)
    
    #from objectron
    def check_object_visibility(self, meta, keypoints: np.ndarray) -> float:
        """Check if object is visible in the image."""

        #modified based on https://github.com/google-research-datasets/Objectron/issues/37

        w = meta['width']
        h = meta['height']
        
        visibility = 1.0
        # Check if object center is inside image.
        cx, cy = keypoints[0]
        if not (0 < cx/w < 1 and 0 < cy/h < 1):
            visibility = 0.0
        # Check if all keypoints are not too far away from image border.
        if any(not (-0.5 < x/w < 1.5 and -0.5 < y/h < 1.5) for x, y in keypoints[1:]):
            visibility = 0.0

        return visibility
    def __getitem__(self, idx):
        
        #初始參數設定========================
        max_objs = 10
    
        #===================================
        #=========================================================================#
        #               1. 讀出一筆訓練資料 包含其annotation                        #
        #=========================================================================#
        test_data  = self.test_data_list[idx]

        color_img_path = test_data["color_img"]
        depth_img_path = test_data["depth_img"]
        #print(img_path)
        anns_path = test_data["annotation"]
        #print(anns)
        #print("file:", anns_path)

        #把訓練資料讀近來===========================
        with open(anns_path) as f:
            anns = json.load(f)
        
        color_img = None
        depth_img = None
        try:
            print("read data:", anns_path)
            color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
            if self.depth: 
                depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
                #print("depth.type", depth_img.dtype) #np.uint16

        except:
            return None
        

        #cv2.imshow("yoyom",color_img)
        #cv2.waitKey(0)
        
            
        test_inp = None
        if self.depth:
            test_inp =  np.dstack((color_img.astype(np.float32), depth_img.astype(np.float32)))
        else:
            test_inp = color_img

            #print("equal?",
            #        np.array_equal(
            #        test_inp[:,:,:3],
            #        color_img
            #    )
            #)
            

        intr = anns["camera_data"]["intrinsics"]
        width = anns['camera_data']['width']
        height = anns['camera_data']['height']
        num_objs = min(len(anns['objects']), max_objs)
            
        camera_intr = np.array(
            [
                [intr['fx'],0         ,intr['cx']],
                [0,         intr['fy'],intr['cy']],
                [0,         0,         1]
            ]
        )

        #pose_solver = PoseSolverFoup(camera_intr)
        meta = {  #一張照片的基本資訊
                        'image':test_inp,
                        'intrinsic':camera_intr,
                        'width': width,
                        'height':height
                    }

        cls_list = []
        instances_2d = []
        instances_3d = []
        visibility = []
        scale_list = []
        scale_list = []
        bboxs = []
        for obj_idx in range(num_objs):
            #print("obj_idx: ",obj_idx )
            #所有的box
            bbox = {}
            ann = anns['objects'][obj_idx]
            
            #一個物件的9個keypoint 和size=====================================================
            nine_keypoint_anno = ann['projected_cuboid']# 包含中心點在內的9個keypoints
            abs_scale = ann['scale'] #絕對size
            obj_ct_anno = nine_keypoint_anno[0]
            kps_anno = nine_keypoint_anno[1:]
            cls_id = ann['class']
            if cls_id not in self.demo_class:  continue
            #================================================================================
            
            bbox['kps'] = nine_keypoint_anno
            bbox['abs_scale'] = abs_scale
            #3d_instance
            #ret, tvec, rvec, projected_points, reprojectionError = pose_solver.solve(nine_keypoint_anno[1:],abs_scale)
            
            box_scale = None
            
            if self.opt.depth:
                box_scale = abs_scale
            else:
                box_scale = np.array(abs_scale)/np.array(abs_scale)[1]

            try:
                #TODO: ground_dataset_2/annotation/92.jason的最後一個box有問題
                projected_points, point_3d_cam, _, _, bbox = \
                    pnp_shell(meta, bbox, nine_keypoint_anno, box_scale)
            except:
                continue
            #TODO: 這邊是先用相對scale做預測得到3d keypoints，以後可能改成絕對scale預測得到3d keypoints

            vis = self.check_object_visibility(meta, nine_keypoint_anno)
            

            cls_list.append(cls_id)
            instances_2d.append(projected_points)
            instances_3d.append(point_3d_cam)
            scale_list.append(abs_scale)
            scale_list.append(box_scale)
            visibility.append(vis)
            bboxs.append(bbox)


    
        ret = {
            'cls':cls_list,
            '2d_instance':instances_2d,
            '3d_instance':instances_3d,
            'visibility':visibility,
            'scale': scale_list,
            'scale':scale_list,
            'bboxs': bboxs
        }

        return meta, ret

    
