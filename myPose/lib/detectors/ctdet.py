from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

#centerpose======================
from pnp.cuboid_pnp_solver import CuboidPNPSolver
from pnp.cuboid_objectron import Cuboid3d
#================================

#====================================
from pose_solver.pose_recover_foup import PoseSolverFoup
from pose_solver.cubic_bbox_foup import Axis
#====================================
try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector
"""
    [因為懶得改名]
    若網路有使用深度資訊的話，那麼這裡的relative_scale, re_scale之類的東西 通通都是absolute_scale
    
"""
class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]  #把照片丟進去網路

      #把feature 從網路裡面抓出來=============
      hm = output['hm'].sigmoid_() #heatmap要做sigmoid
      wh = output['wh']
      #=====================================
      reg = output['reg'] if self.opt.reg_offset else None

      #mycode===============================
      kps = output['kps']
      relative_scale = output['scale']

      #grasp
      single_grasp_center_hm = output['single_grasp_center_hm']
      single_grasp_center_offset = output['single_grasp_center_offset']

      single_grasp_ct_kps_dsp = output['single_grasp_ct_kps_dsp']
      single_grasp_type = output['single_grasp_type']
      #single_grasp_kpt_mask = output['single_grasp_kpt_mask']
      single_grasp_width = output['single_grasp_width']
      single_graspct_objct_dsp = output['single_graspct_objct_dsp']

      paired_grasp_center_hm = output['paired_grasp_center_hm']
      paired_grasp_center_offset = output['paired_grasp_center_offset']

      paired_grasp_ct_kps_dsp = output['paired_grasp_ct_kps_dsp']
      paired_grasp_type = output['paired_grasp_type']
      #paired_grasp_kpt_mask = output['paired_grasp_kpt_mask']
      paired_grasp_width = output['paired_grasp_width']
      paired_graspct_objct_dsp = output['paired_graspct_objct_dsp']


      #=====================================

      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()

      #解析輸出


      #original code
      #dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

      #mycode
      dets,center_and_kps_coor,re_scale,\
      single_grasp_ct_kps_dsp, single_grasp_type, single_grasp_width,single_grasp_objct_ct,\
      paired_grasp_ct_kps_dsp, paired_grasp_type, paired_grasp_width,paired_grasp_objct_ct \
        = ctdet_decode(
        hm, wh, reg=reg, kps = kps, relative_scale = relative_scale, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K,
        single_grasp_center_hm = single_grasp_center_hm,
        single_grasp_center_offset = single_grasp_center_offset,
        
        single_grasp_ct_kps_dsp = single_grasp_ct_kps_dsp,
        single_grasp_type = single_grasp_type,
        #single_grasp_kpt_mask = single_grasp_kpt_mask,
        single_grasp_width = single_grasp_width,
        single_graspct_objct_dsp = single_graspct_objct_dsp,

        paired_grasp_center_hm = paired_grasp_center_hm,
        paired_grasp_center_offset = paired_grasp_center_offset,
        
        paired_grasp_ct_kps_dsp = paired_grasp_ct_kps_dsp,
        paired_grasp_type = paired_grasp_type,
        #paired_grasp_kpt_mask = paired_grasp_kpt_mask,
        paired_grasp_width = paired_grasp_width,
        paired_graspct_objct_dsp = paired_graspct_objct_dsp
        
        
        )
      
      # det = [batch. K, 6] 6分別是: bbox的左上x 左上y 右下x 右下y score classes

      
    if return_time:
      #return output, dets, forward_time
      return output, dets, forward_time, center_and_kps_coor, re_scale,\
      single_grasp_ct_kps_dsp, single_grasp_type, single_grasp_width,single_grasp_objct_ct,\
      paired_grasp_ct_kps_dsp, paired_grasp_type, paired_grasp_width,paired_grasp_objct_ct 
    else:
      #return output, dets
      return output, dets, center_and_kps_coor, re_scale,\
      single_grasp_ct_kps_dsp, single_grasp_type, single_grasp_width,single_grasp_objct_ct,\
      paired_grasp_ct_kps_dsp, paired_grasp_type, paired_grasp_width,paired_grasp_objct_ct 

  def post_process(self, dets, meta, scale=1, center_and_kps_coor  = None, relative_scale=None,
                   single_grasp_ct_kps_dsp=None,
                   single_grasp_type=None,
                   single_grasp_width=None,
                   single_grasp_objct_ct=None,
                   paired_grasp_ct_kps_dsp=None,
                   paired_grasp_type=None,
                   paired_grasp_width=None,
                   paired_grasp_objct_ct=None 
                   ):
    #print("dets = ",dets.shape) [batch = 1, 100, 6]
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2]) #好像是沒啥變化
    
    #mycode ===
    if center_and_kps_coor != None:
      #keypoints
      center_and_kps_coor = center_and_kps_coor.detach().cpu().numpy()
      center_and_kps_coor = center_and_kps_coor.reshape(1,-1,center_and_kps_coor.shape[2])

      #relative scale===========
      relative_scale = relative_scale.detach().cpu().numpy()
      relative_scale = relative_scale.reshape(1,-1,relative_scale.shape[2])
      #========================
    
      
      #grasp
      single_grasp_ct_kps_dsp=single_grasp_ct_kps_dsp.detach().cpu().numpy()
      single_grasp_ct_kps_dsp=single_grasp_ct_kps_dsp.reshape(1,-1,single_grasp_ct_kps_dsp.shape[2])

      single_grasp_type=single_grasp_type.detach().cpu().numpy()
      single_grasp_type=single_grasp_type.reshape(1,-1,single_grasp_type.shape[2])
      
      single_grasp_width=single_grasp_width.detach().cpu().numpy()
      single_grasp_width=single_grasp_width.reshape(1,-1,single_grasp_width.shape[2])
      
      single_grasp_objct_ct=single_grasp_objct_ct.detach().cpu().numpy()
      single_grasp_objct_ct=single_grasp_objct_ct.reshape(1,-1,single_grasp_objct_ct.shape[2])
      
      paired_grasp_ct_kps_dsp=paired_grasp_ct_kps_dsp.detach().cpu().numpy()
      paired_grasp_ct_kps_dsp=paired_grasp_ct_kps_dsp.reshape(1,-1,paired_grasp_ct_kps_dsp.shape[2])
      
      paired_grasp_type=paired_grasp_type.detach().cpu().numpy()
      paired_grasp_type=paired_grasp_type.reshape(1,-1,paired_grasp_type.shape[2])
      
      paired_grasp_width=paired_grasp_width.detach().cpu().numpy()
      paired_grasp_width=paired_grasp_width.reshape(1,-1,paired_grasp_width.shape[2])
      
      paired_grasp_objct_ct=paired_grasp_objct_ct.detach().cpu().numpy()
      paired_grasp_objct_ct=paired_grasp_objct_ct.reshape(1,-1,paired_grasp_objct_ct.shape[2])

      dets, ct_kps_coor, re_scale,\
      single_grasp_ct_kps_dsp,single_grasp_type,single_grasp_width,single_grasp_objct_ct,\
      paired_grasp_ct_kps_dsp,paired_grasp_type,paired_grasp_width,paired_grasp_objct_ct\
      = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes, center_and_kps_coor  = center_and_kps_coor, relative_scale = relative_scale,
        
        single_grasp_ct_kps_dsp=single_grasp_ct_kps_dsp,
        single_grasp_type=single_grasp_type,
        single_grasp_width=single_grasp_width,
        single_grasp_objct_ct=single_grasp_objct_ct,

        paired_grasp_ct_kps_dsp=paired_grasp_ct_kps_dsp,
        paired_grasp_type=paired_grasp_type,
        paired_grasp_width=paired_grasp_width,
        paired_grasp_objct_ct=paired_grasp_objct_ct
        )
      
    else:
    #================
      dets = ctdet_post_process(
          dets.copy(), [meta['c']], [meta['s']],
          meta['out_height'], meta['out_width'], self.opt.num_classes)
    #這時候的dets是list 裡面放一個dict
    """
    [
      {  1(類別): [   
                      [5筆資料(左上角xy 右下角xy 偵測分數)     ],
                      [     ]
                      ...
                  ] ,

        2(類別): [
        
                ]
      ....
      }  

    ]
    
    """
    
 
    # j是從1 開始
    #print(dets)
    # print("tttetetat:",dets[0][1]) [0]是這個list就只有1個元素 [1] 是第一個類別 底下有好多個偵測物
    for j in range(1, self.num_classes + 1):
      #print("dets[0][j]=", dets[0][j])
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5) #把每一個類別底下的偵測物，全部用array形式表示
      # print("array.shape = ",dets[0][j].shape)   [該類別下的偵測出來的topk物件總數,   5]
      dets[0][j][:, :4] /= scale  #scale是1
      #mycode======================
      if np.any(center_and_kps_coor):
        #print("cocoL:",np.array(ct_kps_coor[0][j], dtype = np.float32).shape)
        ct_kps_coor[0][j] = np.array(ct_kps_coor[0][j], dtype = np.float32).reshape(-1,18)
        ct_kps_coor[0][j][:,:18] /= scale

        re_scale[0][j] = np.array(re_scale[0][j], dtype = np.float32).reshape(-1, 3)
        #relative scale應該就不需要再除以scale

        #grasp
    for j in range(1,2):
      if np.any(center_and_kps_coor):
        single_grasp_ct_kps_dsp[0][j] = np.array(single_grasp_ct_kps_dsp[0][j], dtype = np.float32).reshape(-1,10)
        single_grasp_type[0][j] = np.array(single_grasp_type[0][j], dtype = np.float32).reshape(-1,10)
        single_grasp_width[0][j] = np.array(single_grasp_width[0][j], dtype = np.float32).reshape(-1,1)
        single_grasp_objct_ct[0][j] = np.array(single_grasp_objct_ct[0][j], dtype = np.float32).reshape(-1,2)

        paired_grasp_ct_kps_dsp[0][j] = np.array(paired_grasp_ct_kps_dsp[0][j], dtype = np.float32).reshape(-1,18)
        paired_grasp_type[0][j] = np.array(paired_grasp_type[0][j], dtype = np.float32).reshape(-1,10)
        paired_grasp_width[0][j] = np.array(paired_grasp_width[0][j], dtype = np.float32).reshape(-1,2)
        paired_grasp_objct_ct[0][j] = np.array(paired_grasp_objct_ct[0][j], dtype = np.float32).reshape(-1,2)

      #=============================
    
    #mycode======================
    if np.any(center_and_kps_coor):
      return dets[0], ct_kps_coor[0], re_scale[0],\
      single_grasp_ct_kps_dsp[0],single_grasp_type[0],single_grasp_width[0],single_grasp_objct_ct[0],\
      paired_grasp_ct_kps_dsp[0],paired_grasp_type[0],paired_grasp_width[0],paired_grasp_objct_ct[0]
    else:
    #=============================
      return dets[0]

  def merge_outputs(self, detections, ct_kps_coors, relative_scale_dets,
                    single_grasp_ct_kps_dsp_dets,
                    single_grasp_type_dets,
                    single_grasp_width_dets,
                    single_grasp_objct_ct_dets,
                    
                    paired_grasp_ct_kps_dsp_dets ,
                    paired_grasp_type_dets,
                    paired_grasp_width_dets ,
                    paired_grasp_objct_ct_dets 
                    
                    ):
    results = {}
    ct_kps_coor_results = {}
    relative_scale_results = {}

    single_grasp_ct_kps_dsp_results = {}
    single_grasp_type_results = {}
    single_grasp_width_results = {}
    single_grasp_objct_ct_results = {}
    
    paired_grasp_ct_kps_dsp_results = {}
    paired_grasp_type_results = {}
    paired_grasp_width_results = {}
    paired_grasp_objct_ct_results = {}
    #print("len(detection),", len(detections)) # 1 而已
    """
    detections[scale] 阿scale只有1 所以detections是只有長度唯一的list
    detections[0]:
    {
        1. : np.array([
              [五筆資料],
              [五筆資料]
        ])

        2. :

    }
    
    """
    #print(detections[0])
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)  
      #  應該吧這段應該可以無視 畢竟detections長度只有1
      #但如果scale不只一個 也就是detection長度不只一 那應該就是把不同scale的相同類別的輸出都放在一起
      if len(self.scales) > 1 or self.opt.nms:
        #print("scalesss???")  
        #這邊沒有近來
        soft_nms(results[j], Nt=0.5, method=2)
#mycode==================================================================
    for j in range(1, self.num_classes + 1):
      ct_kps_coor_results[j] = np.concatenate(
        [ct_kps_coor[j] for ct_kps_coor in ct_kps_coors], axis=0).astype(np.float32)  
      #這段應該可以無視 畢竟detections長度只有1
      #但如果scale不只一個 也就是detection長度不只一 那應該就是把不同scale的相同類別的輸出都放在一起

      #re-scale==
      relative_scale_results[j] = np.concatenate(
        [relative_scale_det[j] for relative_scale_det in relative_scale_dets], axis=0).astype(np.float32)
      #==========

      if len(self.scales) > 1 or self.opt.nms:
        print("scalesss???")  
        #這邊沒有近來
        soft_nms(ct_kps_coor_results[j], Nt=0.5, method=2)
    
    for j in range(1,2):
      single_grasp_ct_kps_dsp_results[j] = np.concatenate(
        [single_grasp_ct_kps_dsp_det[j] for single_grasp_ct_kps_dsp_det in single_grasp_ct_kps_dsp_dets], axis=0).astype(np.float32)  
      single_grasp_type_results[j] = np.concatenate(
        [single_grasp_type_det[j] for single_grasp_type_det in single_grasp_type_dets], axis=0).astype(np.float32)  
      single_grasp_width_results[j] = np.concatenate(
        [single_grasp_width_det[j] for single_grasp_width_det in single_grasp_width_dets], axis=0).astype(np.float32)  
      single_grasp_objct_ct_results[j] = np.concatenate(
        [single_grasp_objct_ct_det[j] for single_grasp_objct_ct_det in single_grasp_objct_ct_dets], axis=0).astype(np.float32)  
      
      paired_grasp_ct_kps_dsp_results[j] = np.concatenate(
        [paired_grasp_ct_kps_dsp_det[j] for paired_grasp_ct_kps_dsp_det in paired_grasp_ct_kps_dsp_dets], axis=0).astype(np.float32)  
      paired_grasp_type_results[j] = np.concatenate(
        [paired_grasp_type_det[j] for paired_grasp_type_det in paired_grasp_type_dets], axis=0).astype(np.float32)  
      paired_grasp_width_results[j] = np.concatenate(
        [paired_grasp_width_det[j] for paired_grasp_width_det in paired_grasp_width_dets], axis=0).astype(np.float32)  
      paired_grasp_objct_ct_results[j] = np.concatenate(
        [paired_grasp_objct_ct_det[j] for paired_grasp_objct_ct_det in paired_grasp_objct_ct_dets], axis=0).astype(np.float32)  
#==========================================================================

    #這是要限制輸出的
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])  #hstack 沿著cloumn把所有類別的輸出都疊起來變成一個維度唯一的array
    
    #print("scores,shape=",scores.shape) #   (K, )  
    #
 
    if len(scores) > self.max_per_image: #100 
      #不會進來
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
        #mycode==============
        ct_kps_coor_results[j] = ct_kps_coor_results[j][keep_inds]
        relative_scale_results[j] = relative_scale_results[j][keep_inds]

      for j in range(1):
        single_grasp_ct_kps_dsp_results[j] = single_grasp_ct_kps_dsp_results[j][keep_inds]
        single_grasp_type_results[j] = single_grasp_type_results[j][keep_inds]
        single_grasp_width_results[j] =  single_grasp_width_results[j][keep_inds]
        single_grasp_objct_ct_results[j] =  single_grasp_objct_ct_results[j][keep_inds]
     
        paired_grasp_ct_kps_dsp_results[j] = paired_grasp_ct_kps_dsp_results[j][keep_inds]
        paired_grasp_type_results[j] = paired_grasp_type_results[j][keep_inds]
        paired_grasp_width_results[j] =paired_grasp_width_results[j][keep_inds]
        paired_grasp_objct_ct_results[j] = paired_grasp_objct_ct_results[j][keep_inds]

        #====================

    
    return results, ct_kps_coor_results, relative_scale_results,\
      single_grasp_ct_kps_dsp_results,single_grasp_type_results,single_grasp_width_results,single_grasp_objct_ct_results,\
      paired_grasp_ct_kps_dsp_results,paired_grasp_type_results,paired_grasp_width_results,paired_grasp_objct_ct_results,


  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results, ct_kps_coor_results, relative_scale_results, camera_instrinic,
                   single_grasp_ct_kps_dsp_results,single_grasp_type_results,
                    single_grasp_width_results,single_grasp_objct_ct_results,
                    paired_grasp_ct_kps_dsp_results,paired_grasp_type_results,
                    paired_grasp_width_results,paired_grasp_objct_ct_results
                        
                   ):
    #mycode=======
    #camera_instrinic = np.array([
    #  [camera_instrinic['fx'], 0, camera_instrinic['cx']],
    #  [0, camera_instrinic['fy'], camera_instrinic['cy']],
    #  [0,0,1]
    #])
    
    
    # N * 2
    #print("scale:", re_scale)
    #=====================================================================================
    #print("camera_instrinic:", camera_instrinic)
    slover = PoseSolverFoup(camera_intrinsic=camera_instrinic)
    #=============
    debugger.add_img(image, img_id='ctdet')
    #pose
    box_center_list = [] 
    box_id = 0
    for j in range(1, self.num_classes + 1):
      
      for bbox_idx in range(len(results[j])):
        bbox = results[j][bbox_idx]
        pts = ct_kps_coor_results[j][bbox_idx]
        re_scale = relative_scale_results[j][bbox_idx]
        #print("pts shape = ", pts.shape)
        box_scale = None
        if self.opt.depth:
          
          box_scale = re_scale
          #print("box_scale:", box_scale)
        else:
          box_scale = re_scale/re_scale[1]

        
        if bbox[4] > self.opt.vis_thresh: #把輸出設了一個thresh
          ret, tvec_new, rvec_new, projected_points, reprojectionError ,quaternion= slover.solve(pts, box_scale)
          #debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
          #mycode=========
          #debugger.add_ct_pt(pts[0:2], j-1, bbox[4], img_id='ctdet')
          #print(projected_points)
          debugger.add_3d_bbox(
            pts, box_id, bbox[4], img_id='ctdet'
          )
          box_id += 1
          box_center_point = np.array(pts).reshape((9,2))[0]
          box_center_list.append(box_center_point)
          
          axis_obj = Axis()
          axis = np.array(axis_obj.get_vertices())
          axis_3d_point = []
          
          for pt in axis:
            axis_3d_point.append(pt)

          axis_3d_point = np.array(axis_3d_point, dtype = float)

          #print("axis:",axis_3d_point)
          axis_proj, _ = cv2.projectPoints(axis_3d_point, rvec_new, tvec_new, camera_instrinic,
                                                      np.zeros((4, 1)))
          axis_proj = np.squeeze(axis_proj)
          axis_proj = axis_proj.astype(int)
          #print("axis_proj = ",axis_proj)
          
          debugger.add_axis(
            axis_proj, j-1, img_id = 'ctdet'
          )
            
    #grasp       
    
    for j in range(1,2):

      for sg_idx in range(len(single_grasp_ct_kps_dsp_results[j])):
        pts = single_grasp_ct_kps_dsp_results[j][sg_idx] 
        grasp_center_point = np.array(pts).reshape((5,2))[0]
        box_center_array = np.array(box_center_list) 
        obj_ct = single_grasp_objct_ct_results[j][sg_idx]
        
        min = 10000
        argmin = None
        box_id = 0
        for ct in box_center_array:
          dist = np.linalg.norm(obj_ct-ct)
          if(dist<min): 
            min = dist
            argmin = box_id
            box_id+=1
        grasp_box_id = argmin
      
        #print('single grasp_box_id:', grasp_box_id)
        #print(box_center_array)
        #print(obj_ct)
        debugger.add_single_grasp(
          pts,
          cat = grasp_box_id,
          img_id = 'ctdet',
          show_obj_ct = True,
          obj_ct = obj_ct
        )
        
    for j in range(1,2):
      for pg_idx in range(len(paired_grasp_ct_kps_dsp_results[j])):
        pts = paired_grasp_ct_kps_dsp_results[j][pg_idx]
        grasp_center_point = np.array(pts).reshape((9,2))[0]
        
        
        box_center_array = np.array(box_center_list) 
        obj_ct = paired_grasp_objct_ct_results[j][pg_idx]
        
        min = 10000
        argmin = None
        box_id = 0
        for ct in box_center_array:
          dist = np.linalg.norm(obj_ct-ct)
          if(dist<min): 
            min = dist
            argmin = box_id
            box_id+=1
        grasp_box_id = argmin
        
        
        #print('paired grasp_box_id:', grasp_box_id)
        #print(box_center_array)
        #print(obj_ct)
        debugger.add_paired_grasp(
          pts,
          cat = grasp_box_id,
          img_id = 'ctdet',
          show_obj_ct = True,
          obj_ct =obj_ct
        )
            
    debugger.show_all_imgs(pause=self.pause)
