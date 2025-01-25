
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
#
from pose_solver.pose_recover_foup import PoseSolverFoup
from pose_solver.cubic_bbox_foup import Axis
from pose_solver.foup_pnp_shell import pnp_shell

import albumentations as A
class foup_dataset(Dataset):
    
    default_resolution = (1280,720)
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    #num_classes = 3
    
    num_classes = 10
    #grasp
    grasp_kpt_num = 4
    single_grasp_num = 20
    paired_grasp_num = 20
    grasp_type_num = 10
    # Mean and stddev of the RGB-D    (KGN)
    mean_rbgd = np.array([0.64326715, 0.64328622, 0.64328383, 0.7249166783727705], dtype=np.float32)
    std_rbgd = np.array([0.03159907, 0.03159791, 0.03159052, 0.06743566336832418], dtype=np.float32)
    
    mean_rgbd = mean_rbgd
    std_rgbd = std_rbgd

    class_name = ["foup12","foup8","magazin"]

    def __init__(self, data_dir_pth ,opt, split, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.opt = opt

        self.depth = self.opt.depth
        self.train_data_list = []

        self.data_dir_pth = data_dir_pth
        self._data_rng = np.random.RandomState(123)
        
        #數值來自於centerPose
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)


        if not self.depth:
            self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                        dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(1, 1, 3)
        else:
            self.mean = np.array([0.64326715, 0.64328622, 0.64328383, 0.7249166783727705], dtype=np.float32)
            self.std =  np.array([0.03159907, 0.03159791, 0.03159052, 0.06743566336832418], dtype=np.float32)
            self.mean_rgbd = self.mean
            self.std_rgbd = self.std

        self.split = split
        self.class_name = ["foup12","foup8","magazin"]

        
        self.max_objs = 10
        
        
        self.num_classes = self.opt.num_classes
        print("foup dataset number of classes: ",self.num_classes)
        
        #grasp
        self.grasp_kpt_num = 4
        self.single_grasp_num = 20
        self.paired_grasp_num = 20
        self.grasp_type_num = 10

        self.get_train_data()
        print("total:",len(self.train_data_list))
        
        print("training class:", self.opt.train_class)
    def get_train_data(self):
        print("collecting data from: ",self.data_dir_pth)
        data_dir_list = os.listdir(self.data_dir_pth)
        print(data_dir_list)
        i=0

        data_idx = 0

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

                data_idx+=1
                if(self.split == "train" and data_idx % (self.opt.spilt_ratio+1) != 0):
                    self.train_data_list.append(train_data_dict)
                elif( (self.split == "val" or self.split == "test" ) and data_idx % (self.opt.spilt_ratio+1) == 0):
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

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config
        Arguments:
            depth {np.ndarray} -- depth map
        Returns:
            np.ndarray -- augmented depth map
        Source: https://github.com/NVlabs/contact_graspnet/blob/main/contact_graspnet/data.py#L538-L560
        """
        # from KGN

        if self.opt.depth_aug_sigma > 0:
            clip = self.opt.depth_aug_clip
            sigma = self.opt.depth_aug_sigma
            noise = np.clip(sigma*np.random.randn(*depth.shape), -clip, clip)
            depth += noise
        if self.opt.depth_aug_gaussian_kernel > 0:
            kernel = self.opt.depth_aug_gaussian_kernel 
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth,(kernel,kernel),0)
            depth[depth_copy==0] = depth_copy[depth_copy==0]
                
        return depth
    def _get_aug_param(self, c_ori, s, width, height, disturb=False):
        # borrowed from center pose
        c = c_ori.copy()
        if (not self.opt.not_rand_crop) and not disturb:
            # Training for current frame
            aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            # Training for previous frame
            sf = self.opt.scale
            cf = self.opt.shift

            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.aug_rot:
            rf = self.opt.rotate  # 0 - 180
            rot = 2 * (np.random.rand() - 0.5) * rf
            # rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0

        return c, aug_s, rot
    

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
    def _get_border(self, border, size):
        #center pose
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, idx):
        
        #初始參數設定========================
        max_objs = 10
        input_res = 512
        output_res = 128 #512/4
        num_kps = 8 
        num_class = 1

        # grasp
        paired_grasp_num = self.paired_grasp_num 
        single_grasp_num = self.single_grasp_num
        #print("single_grasp_num:",single_grasp_num)
        #print("paired_grasp_num:", paired_grasp_num)
        grasp_num_kps = 4
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
        #print("file:", anns_path)

        #把訓練資料讀近來===========================
        with open(anns_path) as f:
            anns = json.load(f)
        
        color_img = None
        depth_img = None
        try:
            color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
            if self.depth: 
                depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
                #print("depth.type", depth_img.dtype) #np.uint16

        except:
            return None
        

        
        if self.depth and self.split == "train" and self.opt.data_augmentation:
            #print("dep type = ", depth_img.dtype)
            depth_img = self._augment_depth(depth_img.astype(np.float64))
        #擷取資料=================================================================
        #圖片中心點位置
        height, width = color_img.shape[0], color_img.shape[1]
        image_input_center = np.array([color_img.shape[1] / 2., color_img.shape[0] / 2.], dtype=np.float32)
        # maximum dimension s
        image_input_shape = max(color_img.shape[0], color_img.shape[1]) * 1.0
        rot = 0

        num_objs = min(len(anns['objects']), max_objs)
        
        


        #========================================================================#
        #                       2. traning data add noise                        #
        #========================================================================#
        #TODO: 一些資料增強 平移 加雜訊等等 borrow from centerpose

        if self.opt.data_augmentation == True:
            # Only apply albumentations on spatial data augmentation, nothing to do with gt label
            transform = A.Compose([
                A.MotionBlur(blur_limit=3, p=0.1),
                A.Downscale(scale_min=0.6, scale_max=0.8, p=0.1),
                A.GaussNoise(p=0.2),
                # A.Blur(p=0.2),
                # A.RandomBrightnessContrast(p=0.2),
            ],
            )
            transformed = transform(image=color_img)
            # Update image
            color_img = transformed["image"]

        try:
            height, width = color_img.shape[0], color_img.shape[1]
        except:
            return None
        

        c_ori = np.array([color_img.shape[1] / 2., color_img.shape[0] / 2.], dtype=np.float32)  #抓取圖片的中心點
        s_ori = max(color_img.shape[0], color_img.shape[1]) * 1.0 #圖片的shape的最大值
        rot = 0

        if self.split == 'train':

            c, aug_s, rot = self._get_aug_param(c_ori, s_ori, width, height, disturb=False)
            s = s_ori * aug_s

            #if np.random.random() < self.opt.flip:
            #    flipped = True
            #    img = img[:, ::-1, :]

            #    c[0] = width - c[0] - 1
        else:
            c = c_ori
            s = s_ori

        #=========================================================================#
        #           3. 仿設變換 from 600X800 => 512X512                           #
        #=========================================================================#
        # from CenterNet
        affine_tran_inp = get_affine_transform(
            c, s, rot , [input_res, input_res]
        )

        affine_tran_out = get_affine_transform(
            c, s, rot , [output_res, output_res]
        )


        inp = cv2.warpAffine(color_img, affine_tran_inp, 
                        (input_res, input_res),
                        flags=cv2.INTER_LINEAR)
        
        inp = (inp.astype(np.float32) / 255.)

        #depth
        depth_inp = None
        if self.depth:
            depth_inp = cv2.warpAffine(depth_img, affine_tran_inp, 
                        (input_res, input_res),
                        flags=cv2.INTER_LINEAR)
        
            depth_inp = (depth_inp.astype(np.float32) / 65535.) #因為它是用uint16去存的
            #print("color max:",inp.max())
            #print("depth max:",depth_inp.max())
        

        test_onp = cv2.warpAffine(color_img, affine_tran_out, 
                        (output_res, output_res),
                        flags=cv2.INTER_LINEAR)

        if self.split == 'train': # and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        #print("depth.shape = ", depth_inp.shape)
        if self.depth:
            inp =  np.dstack((inp, depth_inp))
            #print("inp.shape = ", inp.shape)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        
        #print("inp.shpae = ", inp.shape)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Normalize the depth image for visualization
        """
        depth_image_normalized = cv2.normalize(depth_inp, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_normalized = np.uint8(depth_image_normalized)

        # Apply a colormap to the normalized depth image
        depth_image_colored = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)
        """
        # Display the depth image
        #cv2.imshow("inp", inp)
        #cv2.imshow("Depth Image", depth_image_colored)
        #cv2.waitKey(0)  # Wait for a key press to close the window

        #=========================================================================#
        #4. 宣告所有的groud truth 不考慮symmetry 
        #=========================================================================#
        # TODO:考慮symmetry
        #center point
        
        ct_hm = np.zeros((1, self.num_classes, output_res, output_res), dtype = np.float32) #center heatmap
        ind = np.zeros((1, self.max_objs), dtype=np.int64) #用於feature gather (不知道要幹嘛)
        ct_sbpxl_off = np.zeros((1,self.max_objs, 2), dtype=np.float32) #centerpoint subpixel displacement #reg
        ct_sbpxl_mask = np.zeros((1,self.max_objs), dtype=np.uint8) # reg_mask
        #2D bbox wh
        bbox_wh = np.zeros((1, self.max_objs, 2), dtype = np.float32) #2d bounding box的長寬
        

        
        #參考 keypoint-based Grasp Net (KGN) 偵測Center-Kpts Offsets
        #center-keypoint-displacement
        
        #heatmap
        kps_hm = np.zeros((1, self.max_objs, num_kps, output_res, output_res), dtype = np.float32) #keypoint heatmap
        kps_ind = np.zeros((1,self.max_objs, num_kps), dtype = np.int64)
        kps_sbpxl_dsp = np.zeros((1,self.max_objs, num_kps,2), dtype=np.float32) #keypoint subpixel displacement
        kps_sbpxl_mask = np.zeros((1,self.max_objs, num_kps), dtype=np.uint8) #keypoint subpixel displacement
        
        #keypoint-center displacement
        kps_ct_dsp = np.zeros((1, self.max_objs, num_kps * 2), dtype=np.float32) #keypoint-center displacement
        kps_ct_dsp_off = np.zeros((1, self.max_objs, num_kps * 2), dtype=np.float32)
        kps_ct_dsp_mask = np.zeros((1, self.max_objs), dtype=np.uint8)

        #inspired from centerpose: relative scale
        re_scale = np.zeros((1, self.max_objs, 3), dtype=np.float32)
        absolute_scale = np.zeros((1, self.max_objs, 3), dtype=np.float32)
        scale_mask = np.zeros((1, self.max_objs), dtype=np.uint8)

    
        draw_gaussian = draw_umich_gaussian

        #Grasp=======================================================----
        #single grasp
        single_grasp_center_hm = np.zeros((1, 1, output_res, output_res), dtype = np.float32)
        single_grasp_center_offset  = np.zeros((1, self.single_grasp_num, 2), dtype = np.float32)

        single_grasp_kpt_dsp  = np.zeros((1, self.single_grasp_num, 5*2), dtype = np.float32)
        single_grasp_width = np.zeros((1, self.single_grasp_num, 1), dtype = np.float32)
        single_grasp_kpt_mask = np.zeros((1, self.single_grasp_num), dtype=np.uint8)
        single_grasp_type  = np.zeros((1, self.single_grasp_num,  self.grasp_type_num), dtype = np.uint8)
        single_graspct_objct_dsp  = np.zeros((1, self.single_grasp_num,  2), dtype = np.float32)
        single_grasp_ind = np.zeros((1, self.single_grasp_num), dtype=np.int64) 
        #paired grasp
        paired_grasp_center_hm = np.zeros((1, 1, output_res, output_res), dtype = np.float32)
        paired_grasp_center_offset  = np.zeros((1, self.single_grasp_num, 2), dtype = np.float32)

        paired_grasp_kpt_dsp  = np.zeros((1, self.paired_grasp_num, 9*2), dtype = np.float32)
        paired_grasp_kpt_dsp_reverse = np.zeros((1, self.paired_grasp_num, 9*2), dtype = np.float32) # !Note 兩個抓取點互換

        paired_grasp_width = np.zeros((1, self.paired_grasp_num, 2), dtype = np.float32)
        paired_grasp_kpt_mask = np.zeros((1, self.paired_grasp_num), dtype=np.uint8)
        paired_grasp_type  = np.zeros((1, self.paired_grasp_num,  self.grasp_type_num), dtype = np.uint8)
        paired_graspct_objct_dsp  = np.zeros((1, self.paired_grasp_num,  2), dtype = np.float32)
        paired_grasp_ind = np.zeros((1, self.paired_grasp_num), dtype=np.int64) 
        
        single_grasp_id = 0
        paired_grasp_id = 0
        #print("====================================================================================")
        for obj_idx in range(num_objs):
            ann = anns['objects'][obj_idx]
            
            #一個物件的9個keypoint 和size=====================================================
            nine_keypoint_anno = np.array(ann['projected_cuboid'])# 包含中心點在內的9個keypoints
            abs_scale = np.array(ann['scale']) #絕對size
            obj_ct_anno = nine_keypoint_anno[0]
            kps_anno = nine_keypoint_anno[1:]
            cls_id = ann['class']
            #================================================================================

           

            #因為輸出的feature map只有原來的1/4 所以ground truth需要做彷射變換===================
            #affine_tran_out
            temp = []
            for pt in nine_keypoint_anno:
                temp.append(affine_transform(np.array(pt).astype(dtype='int64'), affine_tran_out))
            pts = np.array(temp) #九個點

            #1. 2d bbox
            #annotation pts -> output_res pts
            ct = pts[0]
            kps = pts[1:]
            bbox_2d, h, w = self._get_2d_bbox(kps, output_res, output_res)
        
            if h>0 and w>0:
                #from center net
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius)) #focal loss
                #centernet: 
                #ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                if (ct_int[0] < 0  or ct_int[0] >127 ) or (ct_int[1] < 0  or ct_int[1] >127 ):
                    continue
                draw_gaussian(ct_hm[0][cls_id], ct_int, radius)#center point heatmap
                #print(ct_hm.shape)
                #cv2.imshow("ct_hm", ct_hm[0,0])
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                bbox_wh[0][obj_idx] = 1. * w, 1. * h
                ind[0][obj_idx] = ct_int[1] * output_res + ct_int[0]
                """
                ind是做甚麼的 : 詳細來講需要去看Loss那邊怎麼搞的,總之ind的用途是圍繞在一個函數:torch.gather()
                網路會輸出reg 也就是center point subpixel offset,
                我們在資料集上面，是希望 ct_sbpxl_off = np.zeros((1,self.max_objs, 2), dtype=np.float32)
                最後一個維度寫2 就是為了分別記錄與center point的subpxl offset x跟y
                但是網路實際在輸出這個center point subpxl offset (reg)時 因為是CNN 所以shape會是 [batch, max_objs, 128, 128]
                那我們ind紀錄了中心點所在的位置 但是適用y*128+x的方式紀錄 我們期望網路輸出reg的feature map中在跟中心點相同位置的feature上記錄著
                中心點的subpixel offset, 而 ind就是記錄這個subpixel offset的位置

                在計算loss的時候 會把ind跟output[reg]做transposed and gather feature 把output[reg]裡面
                ind位置的feature抓出來 形成一個新的tensor  其大小會跟dataset的ct_subpxl_off一樣大
                這樣 ct_subpxl_off 和 output[reg]就可以直接去計算L1 loss
                
                """
                ct_sbpxl_off[0][obj_idx] = ct - ct_int #center point subpixel displacement
                ct_sbpxl_mask[0][obj_idx] = 1 #center point mask

                kps_radius = radius
                test_pts = []

                #size====================================
                relative_size = abs_scale/abs_scale[1]
                re_scale[0][obj_idx] = relative_size
                absolute_scale[0][obj_idx] = abs_scale
                scale_mask[0][obj_idx] = 1
                #print("size: ",scale[0][obj_idx])
                #print("relative size", relative_size)
                #raise ValueError("迷子でもすすめ")
                #=======================================
            
                for kp_idx in range(num_kps):
                    #若先只偵測center-keypoint displacement
                    kpx, kpy = kps[kp_idx].astype(np.int32)
                    
                    #把x跟y都加進去
                    
                    test_pts.append(kps[kp_idx]-ct_int)
                    kps_ct_dsp[0, obj_idx, kp_idx*2:kp_idx*2+2] = kps[kp_idx]-ct_int
                    kps_ct_dsp_mask[0, obj_idx] = 1
            
                #self._check_3dbox(test_onp,  ct, kps_ct_dsp[0, obj_idx])

                #注意 這邊跟object是分開的
                #grasp=============================================================================
                paired_grasp_list = ann['paired_grasp_list']
                single_grasp_list = ann['single_grasp_list']
                #================================================================================

                
                

                
                for paired_grasp0, paired_grasp1 in paired_grasp_list:
                    
                    paired_grasp_kpt_mask[0, paired_grasp_id] = 1
                    if paired_grasp_id == paired_grasp_num: break; 
                    # grasp0 kpt ===================================================
                    # 訪設變換
                    grasp0_kpt = np.array(paired_grasp0['projected_keypoints'])
                    
                    temp = []
                    for pt in grasp0_kpt:
                        temp.append(affine_transform(np.array(pt).astype(dtype='int64'), affine_tran_out))
                    grasp0_kpt = np.array(temp) #九個點

                    
                    # grasp1 kpt =================================================== 
                    # 以及訪設變換
                    grasp1_kpt = np.array(paired_grasp1['projected_keypoints'])
                    temp = []
                    for pt in grasp1_kpt:
                        temp.append(affine_transform(np.array(pt).astype(dtype='int64'), affine_tran_out))
                    grasp1_kpt = np.array(temp) #九個點
                    #=================================================================
                    
                    #grasp width=====================================================
                    paired_grasp_width[0, paired_grasp_id, 0] = paired_grasp0["width"]
                    paired_grasp_width[0, paired_grasp_id, 1] = paired_grasp1["width"]
                    #=================================================================



                    #paired grasp center heatmap=========================
                    paired_grasp_ct = (grasp0_kpt[0] + grasp1_kpt[0])/2.0
                    paired_grasp_ct_int = paired_grasp_ct.astype(np.int32)
                    if (paired_grasp_ct_int[0] < 0  or paired_grasp_ct_int[0] >127 ) or (paired_grasp_ct_int[1] < 0  or paired_grasp_ct_int[1] >127 ):
                        continue
                    draw_gaussian(paired_grasp_center_hm[0][0], paired_grasp_ct_int, radius)#center point heatmap
                    paired_grasp_ind[0][paired_grasp_id] = paired_grasp_ct_int[1] * output_res +paired_grasp_ct_int[0]
                    paired_grasp_center_offset[0][paired_grasp_id] = paired_grasp_ct - paired_grasp_ct_int
                    #paired grasp type=================================================
                    for cls_id in range(self.grasp_type_num):
                        #one hot code
                        if cls_id == int(paired_grasp0['class']):
                            paired_grasp_type[0, paired_grasp_id, cls_id] = 1
                        else:
                            paired_grasp_type[0, paired_grasp_id, cls_id] = 0
                    #print('paired_grasp_type: ', paired_grasp_type)

                    #keypoint-------------------------------------
                    paired_grasp_kpt_dsp[0, paired_grasp_id, 0:2] = paired_grasp_ct_int-paired_grasp_ct_int
                    #grasp0的四個點
                    for kp_idx in range(grasp_num_kps):
                        kp_idx_grasp0 = kp_idx+1  #1~4 
                        paired_grasp_kpt_dsp[0, paired_grasp_id, kp_idx_grasp0*2:kp_idx_grasp0*2+2] = grasp0_kpt[kp_idx+1]-paired_grasp_ct_int #2:3 4:5 6:7 8:9
                    #grasp1的四個點
                    for kp_idx in range(grasp_num_kps):
                        kp_idx_grasp1 = kp_idx+5  #5~8
                        paired_grasp_kpt_dsp[0, paired_grasp_id, kp_idx_grasp1*2:kp_idx_grasp1*2+2] = grasp1_kpt[kp_idx+1]-paired_grasp_ct_int
                    

                    paired_grasp_kpt_dsp_reverse[0, paired_grasp_id, 0:2] = paired_grasp_ct_int-paired_grasp_ct_int
                    #grasp0的四個點
                    for kp_idx in range(grasp_num_kps):
                        kp_idx_grasp0 = kp_idx+1  #1~4 
                        paired_grasp_kpt_dsp_reverse[0, paired_grasp_id, kp_idx_grasp0*2:kp_idx_grasp0*2+2] = grasp1_kpt[kp_idx+1]-paired_grasp_ct_int #2:3 4:5 6:7 8:9
                    #grasp1的四個點
                    for kp_idx in range(grasp_num_kps):
                        kp_idx_grasp1 = kp_idx+5  #5~8
                        paired_grasp_kpt_dsp_reverse[0, paired_grasp_id, kp_idx_grasp1*2:kp_idx_grasp1*2+2] = grasp0_kpt[kp_idx+1]-paired_grasp_ct_int


                    #=============================================

                    paired_graspct_objct_dsp[0, paired_grasp_id,  :] = paired_grasp_ct-ct;
                    #print("paired_grasp_id:", paired_grasp_id)
                    paired_grasp_id += 1
                
                


                for single_grasp in single_grasp_list:
                    single_grasp_kpt_mask[0, single_grasp_id] = 1
                    if single_grasp_id == single_grasp_num: break;

                    # grasp kpt 以及訪設變換
                    grasp_kpt = np.array(single_grasp['projected_keypoints'])
                    temp = []
                    for pt in grasp_kpt:
                        temp.append(affine_transform(np.array(pt).astype(dtype='int64'), affine_tran_out))
                    grasp_kpt = np.array(temp) #九個點

                    #grasp width====================================================
                    single_grasp_width[0, single_grasp_id, 0] = single_grasp["width"]
                    #===============================================================


                    #single grasp center heatmap=========================
                    single_grasp_ct = grasp_kpt[0]
                    single_grasp_ct_int = single_grasp_ct.astype(np.int32)
                    if (single_grasp_ct_int[0] < 0  or single_grasp_ct_int[0] >127 ) or (single_grasp_ct_int[1] < 0  or single_grasp_ct_int[1] >127 ):
                        continue
                    draw_gaussian(single_grasp_center_hm[0][0], single_grasp_ct_int, radius)#center point heatmap
                    single_grasp_center_offset[0][single_grasp_id] = single_grasp_ct - single_grasp_ct_int

                    single_grasp_ind[0][single_grasp_id] = single_grasp_ct_int[1] * output_res + single_grasp_ct_int[0]
                    #single grasp type==============================================
                    for cls_id in range(self.grasp_type_num):
                        #one hot code
                        if cls_id == int(single_grasp['class']):
                            single_grasp_type[0, single_grasp_id, cls_id] = 1
                        else:
                            single_grasp_type[0, single_grasp_id, cls_id] = 0
                    #print('single_grasp_type: ', single_grasp_type)
                    
                    
                    #keypoint-------------------------------------
                    for kp_idx in range(grasp_num_kps+1):#+1包含中心點
                        single_grasp_kpt_dsp[0, single_grasp_id, kp_idx*2:kp_idx*2+2] = grasp_kpt[kp_idx]-single_grasp_ct
                    
                    single_graspct_objct_dsp[0, single_grasp_id,  :] = single_grasp_ct-ct;
                    single_grasp_id += 1
                #==================================================================================


        #print(ct_hm[0].shape)
        #print("ind:", ind[0])
       
        ret = {
            'input': inp, 'hm': ct_hm[0], 'ind': ind[0], 'reg' :ct_sbpxl_off[0], 'reg_mask':ct_sbpxl_mask[0],
            'wh': bbox_wh[0],
            'kps': kps_ct_dsp[0], 'kps_mask': kps_ct_dsp_mask[0],
            'relative_scale': re_scale[0], 'scale_mask': scale_mask[0],
            'absolute_scale':absolute_scale[0],
            #'kps_hm':kps_hm, 'kps_sbpxl_dsp':kps_sbpxl_dsp, 'kps_sbpxl_mask':kps_sbpxl_mask, 'kp_ind':kps_ind
            #=================================================================================================
            'paired_grasp_center_hm':paired_grasp_center_hm[0],
            'paired_grasp_center_offset':paired_grasp_center_offset[0], #subpixel offset
            
            'paired_grasp_ct_kps_dsp': paired_grasp_kpt_dsp[0],  # center to keypoint displacement
            'paired_grasp_ct_kps_dsp_reverse':paired_grasp_kpt_dsp_reverse[0],

            'paired_grasp_type': paired_grasp_type[0],
            'paired_grasp_kpt_mask':paired_grasp_kpt_mask[0],
            'paired_grasp_width': paired_grasp_width[0], 
            'paired_graspct_objct_dsp':paired_graspct_objct_dsp[0],
            'paired_grasp_ind':paired_grasp_ind[0],

            'single_grasp_center_hm':single_grasp_center_hm[0],
            'single_grasp_center_offset':single_grasp_center_offset[0],

            'single_grasp_ct_kps_dsp': single_grasp_kpt_dsp[0], 
            
            'single_grasp_type':single_grasp_type[0],
            'single_grasp_kpt_mask':single_grasp_kpt_mask[0],
            'single_grasp_width':single_grasp_width[0],
            'single_graspct_objct_dsp':single_graspct_objct_dsp[0],
            'single_grasp_ind':single_grasp_ind[0]
            }
        

        return ret

        
          

    

if __name__ == "__main__":
    dataset = Objectron_dataset()



    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    next(iter(train_loader))

    print("end")