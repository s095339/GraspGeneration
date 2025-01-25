
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

#
from utils.image import get_affine_transform, affine_transform
from utils.image import flip, color_aug
from utils.image import gaussian_radius, draw_umich_gaussian

from pose_solver.pose_recover_foup import PoseSolverFoup
from pose_solver.pose_recover import PoseSolver
from pose_solver.cubic_bbox_foup import Axis
class foup_dataset(Dataset):
    
    default_resolution = (800,600)
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    num_classes = 3
    def __init__(self, data_dir_pth ,opt, split, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.data_dir_pth = data_dir_pth

        self.train_data_list = []
        
        
        self._data_rng = np.random.RandomState(123)
        
        #數值來自於centerPose
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt


        self.max_objs = 10
        
        
        self.num_classes = 3
        
        self.depth = False
        self.get_train_data()
        print("total:",len(self.train_data_list))


    def get_train_data(self):
        data_dir_list = os.listdir(self.data_dir_pth)
        print(data_dir_list)
        i=0
        for dir_pth in data_dir_list:
            
            print("get traning data from", dir_pth)
            anns_path = os.path.join(dir_pth, "annotation")



            #每一個data_dir
            pth = os.path.join(self.data_dir_pth, anns_path)
            anno_list = [img for img in os.listdir(pth) if img.endswith(".json")]
            #print(imglist)

            
            for anno in anno_list:
                #anno_filename = img.replace('.png','.json')
                #print(anno_filename)
                
                #if not os.path.exists(os.path.join(pth,anno_filename)):
                #    print("not")
                #    continue
                with open(os.path.join(pth,anno)) as f:
                    anns = json.load(f)
                
                color_path = os.path.join(self.data_dir_pth,dir_pth, anns["color_data_path"])
                if self.depth:
                    depth_path = os.path.join(self.data_dir_pth,dir_pth, anns["depth_data_path"])
                else:
                    depth_path = None
                
                train_data_dict = {
                    "annotation": os.path.join(pth,anno),
                    "color_img": color_path,
                    "depth_img": depth_path
                }

                self.train_data_list.append(train_data_dict)
                #print(self.train_data_list)
                #抓出裡面所有的.png檔案



    def __len__(self):
        return len(self.train_data_list)
    def _check_3dbox(self, img, ct, kpt_disp):
        ct = ct.astype(np.int32)
        imgt = img.copy()
        print(kpt_disp)
        print("ct = ", ct)
        cv2.circle(imgt, ct ,2, [255,0,255],2 )
        for idx in range(8):
            dsp = kpt_disp[2* idx:2 * idx + 2].astype(np.int32)
            #print()
            cv2.circle(imgt, ct + dsp, 2, [255,0,255],2 )

        cv2.imshow("test", imgt)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def _get_2d_bbox(self, proj_bbox_3d_pts , img_w, img_h):
        """
        arg: proj_bbox_3d_pts: projected 3D bbox keypoints
        return : 4 points of 2d bbox [left top, righ top, left bottom , right botton]
        """
        x_min = 9999
        x_max = 0
        y_min = 9999
        y_max = 0
        for pt in proj_bbox_3d_pts:
            x,y = pt
            
            if(x<x_min):
                x_min = x
            if(x>x_max):
                x_max = x
            if(y<y_min):
                y_min = y
            if(y>y_max):
                y_max = y

        #為了不要讓2d bbox超出圖片範圍   
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)
        x_min = max(0, x_min)
        y_min = max(0, y_min)

        #計算2D bbox的長寬
        w = x_max - x_min
        h = y_max - y_min
        
        return[
            [x_min, y_min],
            [x_max, y_min],
            [x_min, y_max],
            [x_max, y_max]
        ], y_max-y_min, x_max-x_min
    def __getitem__(self, idx):
        
        #初始參數設定========================
        max_objs = 10
        input_res = 512
        output_res = 128 #512/4
        num_kps = 8 
        num_class = 1
        #===================================
        #=========================================================================#
        #               1. 讀出一筆訓練資料 包含其annotation                        #
        #=========================================================================#
        train_data  = self.train_data_list[idx]

        color_img_path = train_data["color_img"]
        depth_img_path = train_data["depth_img"]
        #print(img_path)
        anns_path = train_data["annotation"]
        #print(anns)
        print("file:", anns_path)
        
        #把訓練資料讀近來===========================
        with open(anns_path) as f:
            anns = json.load(f)
        
        color_img = None
        depth_img = None
        try:
            color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
            if self.depth: depth_img = cv2.imread(depth_img_path)
        except:
            return None
        
        #擷取資料=================================================================
        #圖片中心點位置
        height, width = color_img.shape[0], color_img.shape[1]
       

        num_objs = min(len(anns['objects']), max_objs)
        
        intr = anns["camera_data"]["intrinsics"]
        camera_intr = np.array(
            [
                [intr['fx'],0         ,intr['cx']],
                [0,         intr['fy'],intr['cy']],
                [0,         0,         1]
            ]
        )
        pose_solver = PoseSolverFoup(camera_intr)
        img_gt = color_img
        img_proj = color_img.copy()
        for obj_idx in range(num_objs):
            ann = anns['objects'][obj_idx]
        
            cls_id = ann['class']
            #if(cls_id!=0): continue
            #一個物件的9個keypoint 和size=====================================================
            nine_keypoint_anno = np.array(ann['projected_cuboid'])# 包含中心點在內的9個keypoints
            abs_scale = np.array(ann['scale']) #絕對size
            obj_ct_anno = nine_keypoint_anno[0]
            kps_anno = nine_keypoint_anno[1:]
            #================================================================================

            """ """
            ret, tvec, rvec, projected_points, reprojectionError = pose_solver.solve(nine_keypoint_anno[1:],abs_scale)    

            axis_obj = Axis()
            axis = np.array(axis_obj.get_vertices())
            axis_3d_point = []
            for pt in axis:
                axis_3d_point.append(pt)

            axis_3d_point = np.array(axis_3d_point, dtype = float)

            print("axis:",axis_3d_point)
            axis_proj, _ = cv2.projectPoints(axis_3d_point, rvec, tvec, camera_intr,
                                                        np.zeros((4, 1)))
            axis_proj = np.squeeze(axis_proj)
            axis_proj = axis_proj.astype(int)
            print("axis_proj = ",axis_proj)
            point_Id = 1
            projected_points =projected_points.astype(int)
            for pt in projected_points:
                print("pt = ", pt.astype(int))
                cv2.circle(img_proj, pt.astype(int), 3, [128,127,35], 3)
                #cv2.putText(img_proj, str(point_Id), pt.astype(int), 2 , 2 , [0, 200 , 250], 2)
                point_Id+=1
            
            cv2.line(img_proj, projected_points[0], projected_points[1], [255,0,0],2)
            cv2.line(img_proj, projected_points[0], projected_points[4], [255,0,0],2)
            cv2.line(img_proj, projected_points[5], projected_points[1], [255,0,0],2)
            cv2.line(img_proj, projected_points[4], projected_points[5], [255,0,0],2)

            cv2.line(img_proj, projected_points[2], projected_points[6], [0,255,0],2)
            cv2.line(img_proj, projected_points[3], projected_points[7], [0,255,0],2)
            cv2.line(img_proj, projected_points[2], projected_points[3], [0,255,0],2)
            cv2.line(img_proj, projected_points[6], projected_points[7], [0,255,0],2)

            cv2.line(img_proj, projected_points[0], projected_points[2], [0,0,255],2)
            cv2.line(img_proj, projected_points[3], projected_points[1], [0,0,255],2)
            cv2.line(img_proj, projected_points[7], projected_points[5], [0,0,255],2)
            cv2.line(img_proj, projected_points[6], projected_points[4], [0,0,255],2)

            cv2.arrowedLine(img_proj, axis_proj[0], axis_proj[1], [0,0,255],2)
            cv2.arrowedLine(img_proj, axis_proj[0], axis_proj[2], [0,255,0],2)
            cv2.arrowedLine(img_proj, axis_proj[0], axis_proj[3], [255,0,0],2)
            

            #gt
            key_point = np.array(nine_keypoint_anno)
            for i in range(key_point.shape[0]):
                pt = key_point[i]
                cv2.circle(img_gt, pt, 3, [128,127,35], 3)
                if(i>0):cv2.putText(img_gt, str(i), pt, 2 , 2 , [0, 200 , 250], 2)
                
            #cv2.circle(img_gt, pt, 2, c, 2)
            
            cv2.line(img_gt, key_point[1], key_point[2], [255,0,0],2)
            cv2.line(img_gt, key_point[1], key_point[5], [255,0,0],2)
            cv2.line(img_gt, key_point[6], key_point[2], [255,0,0],2)
            cv2.line(img_gt, key_point[5], key_point[6], [255,0,0],2)

            cv2.line(img_gt, key_point[3], key_point[7], [0,255,0],2)
            cv2.line(img_gt, key_point[4], key_point[8], [0,255,0],2)
            cv2.line(img_gt, key_point[3], key_point[4], [0,255,0],2)
            cv2.line(img_gt, key_point[7], key_point[8], [0,255,0],2)

            cv2.line(img_gt, key_point[1], key_point[3], [0,0,255],2)
            cv2.line(img_gt, key_point[4], key_point[2], [0,0,255],2)
            cv2.line(img_gt, key_point[8], key_point[6], [0,0,255],2)
            cv2.line(img_gt, key_point[7], key_point[5], [0,0,255],2)

            cv2.arrowedLine(img_gt, axis_proj[0], axis_proj[1], [0,0,255],2)
            cv2.arrowedLine(img_gt, axis_proj[0], axis_proj[2], [0,255,0],2)
            cv2.arrowedLine(img_gt, axis_proj[0], axis_proj[3], [255,0,0],2)
        img_result = np.concatenate([img_gt, img_proj],axis=1)
        cv2.imshow("test",img_result)
        cv2.waitKey(0)


if __name__ == "__main__":
    dataset = Objectron_dataset()



    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    next(iter(train_loader))

    print("end")