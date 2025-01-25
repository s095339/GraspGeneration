
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




from PoseGrasp.scene import LabelScene,ImageScene
from PoseGrasp.Obj_Grasp import SceneObj,SingleHandGrasp,TwoHandsGrasp

class grasp_test_dataset():
    
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
            #print("read data:", anns_path)
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
        meta = {  #一張照片的基本資訊\
                        'color_img':color_img,
                        'image':test_inp,
                        'intrinsic':camera_intr,
                        'width': width,
                        'height':height
                    }

        labelscene = LabelScene(meta = meta)

        for obj_idx in range(num_objs):
            #print("obj_idx: ",obj_idx )
            #所有的box
            bbox = {}
            ann = anns['objects'][obj_idx]
            
            #一個物件的9個keypoint 和size=====================================================
            nine_keypoint_anno = np.array(ann['projected_cuboid'])# 包含中心點在內的9個keypoints
            abs_scale = ann['scale'] #絕對size
            obj_ct_anno = nine_keypoint_anno[0]
            kps_anno = nine_keypoint_anno[1:]
            #print("kps_anno:",kps_anno.shape)
            cls_id = ann['class']
            
            #================================================================================
            
            bbox['kps'] = nine_keypoint_anno
            bbox['abs_scale'] = abs_scale
            #3d_instance
            #ret, tvec, rvec, projected_points, reprojectionError = pose_solver.solve(nine_keypoint_anno[1:],abs_scale)
            
            box_scale = None
            
            if self.opt.depth:
                box_scale = np.array(abs_scale)
            else:
                box_scale = np.array(abs_scale)/np.array(abs_scale)[1]
            
            scence_obj = SceneObj(
                center_point_2d=obj_ct_anno,
                keypoint8_2d=kps_anno,
                size = box_scale,
                cls = cls_id,
                meta = meta,
                bbox = None
            )
            #grasp=============================================================================
            paired_grasp_list = ann['paired_grasp_list']
            single_grasp_list = ann['single_grasp_list']
            #================================================================================
            
            for paired_grasp0, paired_grasp1 in paired_grasp_list:
                    
                
                # grasp0 kpt ===================================================
                # 訪設變換
                grasp0_kpt = np.array(paired_grasp0['projected_keypoints'])[:5]
                #print("grasp0_kpt.shape:",grasp0_kpt.shape)
                grasp1_kpt = np.array(paired_grasp1['projected_keypoints'])[:5]
                
                paired_grasp_center = (grasp0_kpt[0] + grasp1_kpt[0])/2
                keypoint8_2d = np.concatenate(
                    (grasp0_kpt[1:], grasp1_kpt[1:]),
                    axis=0
                )
                #=================================================================
                
                #grasp width=====================================================
                width0 = paired_grasp0["width"]
                width1 = paired_grasp1["width"]
                #=================================================================



                #paired grasp center heatmap=========================
                
                #paired grasp type=================================================
                cls_id == int(paired_grasp0['class']) #雙手抓取點的cls一樣
                
                scence_obj.add_twohands_grasp(
                    TwoHandsGrasp(
                        center_point_2d=paired_grasp_center,
                        keypoint8_2d=keypoint8_2d,
                        cls = cls_id,
                        meta = meta, 
                        obj_ct_pred=obj_ct_anno,
                        width0 = width0,
                        width1 = width1
                    )
                )

            for single_grasp in single_grasp_list:

            
                # grasp kpt 以及訪設變換
                grasp_kpt = np.array(single_grasp['projected_keypoints'])[:5]
                width = single_grasp["width"]
                #===============================================================


                #single grasp center heatmap=========================
                cls_id == int(single_grasp['class'])
                    
                scence_obj.add_singlehand_grasp(
                    SingleHandGrasp(
                        center_point_2d = grasp_kpt[0],
                        keypoint4_2d = grasp_kpt[1:],
                        cls = cls_id,
                        meta = meta,
                        obj_ct_pred=obj_ct_anno,
                        width = width,
                    )
                )
                #keypoint-------------------------------------
            
            labelscene.add_obj(scene_obj=scence_obj)
        
        #labelscene.scene_info_show()
        return meta, labelscene
        

    
