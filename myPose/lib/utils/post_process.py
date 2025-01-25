from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes, center_and_kps_coor = None, relative_scale=None,
                      single_grasp_ct_kps_dsp=None,
                      single_grasp_type=None,
                      single_grasp_width=None,
                      single_grasp_objct_ct=None,

                      paired_grasp_ct_kps_dsp=None,
                      paired_grasp_type=None,
                      paired_grasp_width=None,
                      paired_grasp_objct_ct=None
                       
                       ):
  # dets: batch x max_dets x dim                      #mycode 
  # return 1-based class det dict
  """
  #dets = dets.copy(), 
  c,s = [meta['c']], [meta['s']],
  h,w = meta['out_height'], meta['out_width'], 
  num_classes = self.opt.num_classes
  """

  # NOTE! 在這裡，會把圖片轉回原本的形狀。所以bbox的keypoint也會在這邊做彷射變換
  #print("num_classes:", num_classes)
  ret = []
  #mycode==
  ct_coor_ret = [] #包含中心點和八個keypoint
  rescale_ret = []
  #single grasp
  single_grasp_ct_kps_dsp_ret = []
  single_grasp_type_ret = []
  single_grasp_width_ret = []
  single_grasp_objct_ct_ret = []
  #paired grasp
  paired_grasp_ct_kps_dsp_ret = []
  paired_grasp_type_ret = []
  paired_grasp_width_ret = []
  paired_grasp_objct_ct_ret = []
  
  
  #========
  for i in range(dets.shape[0]): #對所有batch底下的所有偵測物
    top_preds = {}
    #mycode===
    top_ct_preds = {} #包含中心點跟8個keypoint
    top_rescale_preds = {} #relative scale
    #single grasp
    top_single_grasp_ct_kps_dsp_preds = {}
    top_single_grasp_type_preds = {}
    top_single_grasp_width_preds = {}
    top_single_grasp_objct_ct_preds = {}
    #paired grasp
    top_paired_grasp_ct_kps_dsp_preds = {}
    top_paired_grasp_type_preds = {}
    top_paired_grasp_width_preds = {}
    top_paired_grasp_objct_ct_preds = {}
      
    #=========
    #應該是把圖片轉回原本的大小 所以點點也要轉回去
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h)) # 把所有 2bbox的左上角的點 轉換回去
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h)) # 把所有 2bbox的右下角的點  轉換回去
    
    #mycode ===============================
    if np.any(center_and_kps_coor) :
      #3d box keypoint
      for idx in range(2,20,2):
        center_and_kps_coor[i,:,idx-2:idx] = transform_preds(
          center_and_kps_coor[i,:,idx-2:idx], c[i], s[i], (w, h)
        )

      #sg grasp kpt
      for idx in range(0,10,2):
        single_grasp_ct_kps_dsp[i,:,idx:idx+2] = transform_preds(
          single_grasp_ct_kps_dsp[i,:,idx:idx+2], c[i], s[i], (w, h)
        )
      #print("single_grasp_objct_ct345:", single_grasp_objct_ct.shape)
      for idx in range(0,2,2):
        single_grasp_objct_ct= transform_preds(
            single_grasp_objct_ct[i,:,idx:idx+2], c[i], s[i], (w, h)
          )
      single_grasp_objct_ct = np.expand_dims(single_grasp_objct_ct, axis=0)
      #print("single_grasp_objct_ct345:", single_grasp_objct_ct.shape)
      #pg grasp kpt
      for idx in range(0,18,2):
        paired_grasp_ct_kps_dsp[i,:,idx:idx+2] = transform_preds(
          paired_grasp_ct_kps_dsp[i,:,idx:idx+2], c[i], s[i], (w, h)
        )
      for idx in range(0,2,2):
        paired_grasp_objct_ct= transform_preds(
            paired_grasp_objct_ct[i,:,idx:idx+2], c[i], s[i], (w, h)
          )
      paired_grasp_objct_ct = np.expand_dims(paired_grasp_objct_ct, axis=0)
    #======================================
    classes = dets[i, :, -1]  #topK當中 class的list, [1, 100 , 1]
                              #把紀錄class的那個提出來

    # print("det:", dets.shape)  [1, K, 6]
    for j in range(num_classes): #對每一個class
      inds = (classes == j)   #抓出所有det當中，class == j的資料所在的index

      #print("inds:",inds)

      #print("dets1",dets[i, inds, :4].shape)   [K,4]
      #print("dets2",dets[i, inds, 4:5].shape)  [K,1]
      top_preds[j + 1] = np.concatenate([
        #這時候的dets[i, inds, :] 裡面的內容是 class == j的資料
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1)
      #print(top_preds[j+1].shape)  [100,5] 因為只有一個類別 
      #假設K 有 100 而在這個類別中只有2個的話 形狀就會是 [2,5]
      top_preds[j + 1] = top_preds[j + 1].tolist()
      
      #nycode=========================================
      if np.any(center_and_kps_coor) :
        #keypoint
        top_ct_preds[j+1] = center_and_kps_coor[i, inds, 0:].astype(np.float32).tolist()
        #relative scale
        top_rescale_preds[j+1] = relative_scale[i, inds, 0:].astype(np.float32).tolist()
      #===============================================
      """
      import numpy as np
      a = np.array([0,1,2,3,4,5,6,7,8,9,10])
      b = np.array([0,1,1,0,0,0,1,1,0,1,1])
      c = (b == 1)
      c
      >>> array([False,  True,  True, False, False, False,  True,  True, False,
              True,  True])
      a[c]
      >>> array([ 1,  2,  6,  7,  9, 10])
      """
    # grasp 也可以做一樣的事情 但他只有一個type
    for j in range(1):
      
      """
      single_grasp_ct_kps_dsp
      single_grasp_type
      single_grasp_width
      single_grasp_objct_ct

      paired_grasp_ct_kps_dsp
      paired_grasp_type
      paired_grasp_width
      paired_grasp_objct_ct
      """
      top_single_grasp_ct_kps_dsp_preds[j+1] = single_grasp_ct_kps_dsp[i, :, 0:].astype(np.float32).tolist()
      top_single_grasp_type_preds[j+1] = single_grasp_type[i, :, 0:].astype(np.float32).tolist()
      top_single_grasp_width_preds[j+1] = single_grasp_width[i, :, 0:].astype(np.float32).tolist()
      #print("single_grasp_objct_ct last:", single_grasp_objct_ct.shape)
      top_single_grasp_objct_ct_preds[j+1] = single_grasp_objct_ct[i, :, 0:].astype(np.float32).tolist() #錯這邊
      #paired grasp
      top_paired_grasp_ct_kps_dsp_preds[j+1] = paired_grasp_ct_kps_dsp[i, :, 0:].astype(np.float32).tolist()
      top_paired_grasp_type_preds[j+1] = paired_grasp_type[i, :, 0:].astype(np.float32).tolist()
      top_paired_grasp_width_preds[j+1] = paired_grasp_width[i, :, 0:].astype(np.float32).tolist()
      top_paired_grasp_objct_ct_preds[j+1] = paired_grasp_objct_ct[i, :, 0:].astype(np.float32).tolist()


      #print(type(top_preds))  top_pred是字典的格式
    ret.append(top_preds) 
    #mycode===
    ct_coor_ret.append(top_ct_preds)
    rescale_ret.append(top_rescale_preds)

     #single grasp
    single_grasp_ct_kps_dsp_ret.append(top_single_grasp_ct_kps_dsp_preds)
    single_grasp_type_ret.append(top_single_grasp_type_preds)
    single_grasp_width_ret.append(top_single_grasp_width_preds)
    single_grasp_objct_ct_ret.append(top_single_grasp_objct_ct_preds)
    #paired grasp
    paired_grasp_ct_kps_dsp_ret.append(top_paired_grasp_ct_kps_dsp_preds)
    paired_grasp_type_ret.append(top_paired_grasp_type_preds)
    paired_grasp_width_ret.append(top_paired_grasp_width_preds)
    paired_grasp_objct_ct_ret.append(top_paired_grasp_objct_ct_preds)
    #=========
    #print("top_preds:", top_preds.shape)
    #print(ret)
  
  #mycode=====================
  if np.any(center_and_kps_coor) :
    return ret, ct_coor_ret, rescale_ret,\
    single_grasp_ct_kps_dsp_ret,single_grasp_type_ret,single_grasp_width_ret,single_grasp_objct_ct_ret,\
    paired_grasp_ct_kps_dsp_ret,paired_grasp_type_ret,paired_grasp_width_ret,paired_grasp_objct_ct_ret
  else:
  #=============================
    return ret


def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
