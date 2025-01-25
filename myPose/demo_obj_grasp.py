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

class Evaluator:
  def __init__(self, opt):
    
    self.opt = opt
    self.obj_class_list = [i for i in range(opt.num_classes)]

    self.box_matching_grasp_acc = 0.0

    self.box_matching_s_g_grasp_acc = 0.0
    self.box_matching_s_g_grasp_num = 0 #pred
    self.box_matching_s_g_grasp_success_num = 0

    self.box_matching_s_g_gt_coveraged_rate = 0.0
    self.box_matching_s_g_gt_coveraged_num = 0 #pred
    self.box_matching_s_g_gt_num = 0


    self.box_matching_s_g_obj_success_num = [0.0 for i in range(opt.num_classes)]
    self.box_matching_s_g_obj_grasp_num = [0.0 for i in range(opt.num_classes)]
    self.box_matching_s_g_obj_success_rate = [0.0 for i in range(opt.num_classes)]
    
    self.box_matching_s_g_obj_gt_num = [0.0 for i in range(opt.num_classes)]
    self.box_matching_s_g_obj_gt_coveraged_num = [0.0 for i in range(opt.num_classes)]
    self.box_matching_s_g_obj_gt_coveraged_rate = [0.0 for i in range(opt.num_classes)]



    self.box_matching_p_g_grasp_acc = 0.0
    self.box_matching_p_g_grasp_num = 0 #pred
    self.box_matching_p_g_grasp_success_num = 0

    self.box_matching_p_g_gt_coveraged_rate = 0.0
    self.box_matching_p_g_gt_coveraged_num = 0 #pred
    self.box_matching_p_g_gt_num = 0



    self.box_matching_p_g_obj_grasp_success_num = [0.0 for i in range(opt.num_classes)]
    self.box_matching_p_g_obj_grasp_num = [0.0 for i in range(opt.num_classes)]
    self.box_matching_p_g_obj_grasp_success_rate = [0.0 for i in range(opt.num_classes)]

    self.box_matching_p_g_obj_gt_num = [0.0 for i in range(opt.num_classes)]
    self.box_matching_p_g_obj_gt_coveraged_num = [0.0 for i in range(opt.num_classes)]
    self.box_matching_p_g_obj_gt_coveraged_rate = [0.0 for i in range(opt.num_classes)]



    self.opt = opt
    return
  def _get_SO3_dist(self, rotMat1, rotMat2):
        """Get the SO3 distance from 2 set of rotation matrices

        Args:
            rotMat1 (N1, 3, 3)
            rotMat2 (N2, 3, 3)
        Returns:
            angle_dists (N1, N2)
        """
        rotMat1_inv = np.linalg.inv(rotMat1)[
            :, np.newaxis, :, :]  # (N1, 1, 3, 3)
        rotMat2_tmp = rotMat2[np.newaxis, :, :, :]  # (1, N2, 3, 3)
        # (N_pred, N_gt, 3, 3)
        mul_result = np.matmul(rotMat1_inv, rotMat2_tmp)
        # (N_pred, N_gt)
        trace = np.trace(mul_result, axis1=2, axis2=3)

        cos_angles = (trace - 1) / 2
        # improve the numerical stability
        cos_angles[cos_angles <= -1] = -1 + 1e-5
        # improve the numerical stability
        cos_angles[cos_angles >= 1] = 1 - 1e-5
        angle_dists = np.arccos(
            cos_angles
        )

        return angle_dists

  def _to_float(self, x):
      return float("{:.4f}".format(x))

  def _rotate_poses_180_by_x(self, poses):
        poses_new = np.copy(poses)
        # correct the pose correspondingly. Rotate along the x axis by 180 degrees
        M_rot = create_homog_matrix(
            R_mat=create_rot_mat_axisAlign([1, -2, -3]),
            T_vec=np.zeros((3, )) 
        )
        poses_new = poses_new @ M_rot 
        return poses_new
  def _eval_grasps_with_gt(self, grasp_poses_pred, grasp_widths_pred, grasp_types_pred,
                                 grasp_poses_gt_obj, grasp_widths_gt_obj, grasp_types_gt_obj,

                            ignore_mask = None,
                             angle_th=np.pi/4, dist_th=0.02, width_th = 0.01, test_width_type = True, **kwargs):
        #print("angle_th = ", angle_th)
        """Evaluate a set of predicted grasps by comparing to a set of ground truth grasps

        Args:
            grasp_poses_pred (array, (N_pred, 4, 4)):   The predicted homogeneous grasp poses in the world frame. 
            grasp_widths_pred (array, (N_pred, )): The predicted grasp open widths
            grasp_poses_gt_obj (N_grasp_poses): The ground truth homogeneous grasp poses in the world frame
            grasp_widths_gt_obj (_type_):  The ground truth grasp open widths. 
            ignore_mask (array (N_pred)):   The mask for ignoring the predicted grasps during the pred_succ_num counting
            angle_th (float): The threshold for the angular difference
            dist_th (float): The threshold for the translation difference

        Returns:
            pred_num (int):             The number of the predicted grasps
            pred_succ_num (int):        The numebr of the successfully predicted grasps
            gt_num (int):               The number of the GT grasps
            gt_cover_num (int):         The numebr of the GT grasps that is covered by the predicted set
        """
        pred_rotations = grasp_poses_pred[:, :3, :3]  # (N_pred, 3, 3)
        pred_translates = grasp_poses_pred[:, :3, 3]  # (N_pred, 3)
        gt_rotations_1 = grasp_poses_gt_obj[:, :3, :3]  # (N_gt, 3, 3)
        gt_translates = grasp_poses_gt_obj[:, :3, 3]  # (N_gt, 3)
        grasp_poses_gt_obj_2 = self._rotate_poses_180_by_x(grasp_poses_gt_obj)
        gt_rotations_2 = grasp_poses_gt_obj_2[:, :3, :3]  # (N_gt, 3, 3)



        # the numbers
        pred_num = pred_rotations.shape[0]
        gt_num = gt_translates.shape[0]

        # SO(3) distances - minimum rotation angle
        angle_dist_1 = self._get_SO3_dist(pred_rotations, gt_rotations_1)
        angle_dist_2 = self._get_SO3_dist(pred_rotations, gt_rotations_2)
        

        # Translation distance
        translates_diff = pred_translates[:, np.newaxis,
                                          :] - gt_translates[np.newaxis, :, :]
        translates_dist = np.linalg.norm(
            translates_diff, axis=2)   # (N_pred, N_gt)

        # match matrix - (N_pred, N_gt)
        matches = np.logical_and(
            np.logical_or(angle_dist_1 < angle_th, angle_dist_2 < angle_th),
            translates_dist < dist_th
        )

        #!Note::#############################################
        #!  注意!!!!                                        #
        #! 這段的目的是為了要測說，如果預測出來的grasp type    #
        #! 和label不一樣 以及預測出來的graps width和label的   #
        #! 差太多，就視為失敗的抓取。
        #! 之所以這行會動的原因，是因為資料集中每一個物件我只有
        #! 標註一個單手抓取點，若學弟妹未來要標註新的資料集，一
        #! 個物件可能有兩個以上的單手抓取候選，那麼這行就會錯! 
        #! 
        #! 所以建議以後這個副程式可以改成一次就一個pred跟label
        #! 下來比
        #! 這個副程式 _eval_grasps_with_gt() 是可以支援多個
        #! 預測抓取點和多個抓取點label相互筆記的              #
        #!###################################################
        #print("grasp_widths_pred:", grasp_widths_pred)
        #print("grasp_widths_gt:", grasp_widths_gt_obj)
        #print("grasp_type_pred:", grasp_types_pred)
        #print("grasp_type_gt:", grasp_types_gt_obj)
        if test_width_type:
          if abs(grasp_widths_pred - grasp_widths_gt_obj) > width_th or grasp_types_pred != grasp_types_gt_obj:
           #print("bad matches: ", matches)
           matches = matches & False;
           #print("matches false: ", matches)
        


        #! ####################################################


        # get the success number
        
        pred_succ = np.any(matches, axis=1)
        gt_covered = np.any(matches, axis=0)
        #return
        assert pred_succ.size == pred_num
        assert gt_covered.size == gt_num

        if ignore_mask is None:
            ignore_mask = np.zeros(pred_num, dtype=bool)
        
        pred_succ_num = np.count_nonzero(pred_succ[np.logical_not(ignore_mask)])
        gt_cover_num = np.count_nonzero(gt_covered)

        # # debug
        # if np.any(~pred_succ) or np.any(~gt_covered):
        #     print("The angular distance: {}\n".format(angle_dist))
        #     print("\n The trace: {} \n".format(trace))

        #     print("The translate distance: {}".format(translates_dist))

        return pred_num, pred_succ_num, gt_num, gt_cover_num, pred_succ, gt_covered
  def _eval_two_grasps_with_gt(self, 
                              grasp_poses_pred0, grasp_widths_pred0, grasp_types_pred0,
                              grasp_poses_pred1, grasp_widths_pred1, grasp_types_pred1,

                              grasp_poses_gt_obj0, grasp_widths_gt_obj0, grasp_types_gt_obj0,
                              grasp_poses_gt_obj1, grasp_widths_gt_obj1, grasp_types_gt_obj1,
                              
                              ignore_mask = None,
                              angle_th=np.pi/4, dist_th=0.02, 
                              width_th = 0.01, test_width_type = True,
                              **kwargs):
    """Evaluate a set of predicted grasps by comparing to a set of ground truth grasps

    Args:
        grasp_poses_pred (array, (1, 4, 4)):   The predicted homogeneous grasp poses in the world frame. 
        grasp_widths_pred (array, (1, )): The predicted grasp open widths
        grasp_poses_gt_obj (1): The ground truth homogeneous grasp poses in the world frame
        grasp_widths_gt_obj (_type_):  The ground truth grasp open widths. 
        ignore_mask (array (1)):   The mask for ignoring the predicted grasps during the pred_succ_num counting
        angle_th (float): The threshold for the angular difference
        dist_th (float): The threshold for the translation difference

    Returns:
        pred_num (int):             The number of the predicted grasps
        pred_succ_num (int):        The numebr of the successfully predicted grasps
        gt_num (int):               The number of the GT grasps
        gt_cover_num (int):         The numebr of the GT grasps that is covered by the predicted set
    """
    #! =====================================================#
    #! 在我的資料集當中，所有的雙手抓取點的grasp type標註都是0
    #! 但有些我給她標錯了= =(需要再做修改) 所以我在這個副程式
    #! 其實就沒有去比較grasp type，但依然有比較grasp width
    #!======================================================#
    # pred ====================================================
 
    pred0_rotations = grasp_poses_pred0[:, :3, :3]  # (1, 3, 3)
    pred0_translates = grasp_poses_pred0[:, :3, 3]  # (1, 3)
    pred1_rotations = grasp_poses_pred1[:, :3, :3]  # (1, 3, 3)
    pred1_translates = grasp_poses_pred1[:, :3, 3]  # (1, 3)
    # gt  =========================================================
    gt0_rotations_1 = grasp_poses_gt_obj0[:, :3, :3]  # (1, 3, 3)
    gt0_rotations_2 = self._rotate_poses_180_by_x(grasp_poses_gt_obj0)[:, :3, :3]# (1, 3, 3)

    gt0_translates = grasp_poses_gt_obj0[:, :3, 3]  # (1, 3)


    gt1_rotations_1 = grasp_poses_gt_obj1[:, :3, :3]  # (1, 3, 3)
    gt1_rotations_2 = self._rotate_poses_180_by_x(grasp_poses_gt_obj1)[:, :3, :3]# (1, 3, 3)
 
    gt1_translates = grasp_poses_gt_obj1[:, :3, 3]  # (1, 3)

    # pred0_rotations pred0_translates   : pred0
    # pred1_rotations pred1_translates   : pred1

    # gt0_rotations_1, gt0_translates_1,  : gt0_1
    # gt0_rotations_2, gt0_translates_2   : gt0_2
  
    # gt1_rotations_1, gt1_translates_1,  : gt1_1
    # gt1_rotations_2, gt1_translates_2   : gt1_2

    pred_num = 1
    gt_num = 1


    # ==================================================================
    #  0比0 1比1
    
    # pred0_rotations pred0_translates   : pred0
    # pred1_rotations pred1_translates   : pred1

    # gt0_rotations_1, gt0_translates,  : gt0_1
    # gt0_rotations_2, gt0_translates   : gt0_2  #gt0轉180

    # gt1_rotations_1, gt1_translates,  : gt1_1
    # gt1_rotations_2, gt1_translates   : gt1_2  #gt1轉180
    # ==================================================================


    # 0跟0  比 ==============================
    # pred0 vs gt0
    angle_dist_1 = self._get_SO3_dist(pred0_rotations, gt0_rotations_1)
    angle_dist_2 = self._get_SO3_dist(pred0_rotations, gt0_rotations_2)
    
    translates_diff= pred0_translates[:, np.newaxis,:] - gt0_translates[np.newaxis, :, :]
    translates_dist = np.linalg.norm(translates_diff, axis=2)   # (N_pred, N_gt)
    
    
    # match matrix - (N_pred = 1, N_gt = 1)
    matches_1 = np.logical_and(
        np.logical_or(angle_dist_1 < angle_th, angle_dist_2 < angle_th),
        translates_dist < dist_th
    )

    if test_width_type:
      if abs(grasp_widths_pred0 - grasp_widths_gt_obj0) > width_th: #TODO:  or grasp_types_pred0 != grasp_types_gt_obj0:
        #print("bad matches00: ", matches_1)
        matches_1 = matches_1 & False;
        #print("matches false00: ", matches_1)

    # pred 1 vs gt1
    angle_dist_1 = self._get_SO3_dist(pred1_rotations, gt1_rotations_1)
    angle_dist_2 = self._get_SO3_dist(pred1_rotations, gt1_rotations_2)
    
    translates_diff = pred1_translates[:, np.newaxis,:] - gt1_translates[np.newaxis, :, :]
    translates_dist = np.linalg.norm(translates_diff, axis=2)   # (N_pred, N_gt)
  
    # match matrix - (N_pred = 1, N_gt = 1)
    matches_2 = np.logical_and(
        np.logical_or(angle_dist_1 < angle_th, angle_dist_2 < angle_th),
        translates_dist < dist_th
    )

    if test_width_type:
      if abs(grasp_widths_pred1 - grasp_widths_gt_obj1) > width_th: #TODO:  or grasp_types_pred1 != grasp_types_gt_obj1:
        #print("bad matches11: ", matches_2)
        matches_2 = matches_2 & False;
        #print("matches false11: ", matches_2)

    match_00_11 = matches_1 & matches_2

    

    # 0跟0  比 ==============================
    # pred0 vs gt1
    angle_dist_1 = self._get_SO3_dist(pred0_rotations, gt1_rotations_1)
    angle_dist_2 = self._get_SO3_dist(pred0_rotations, gt1_rotations_2)
    
    translates_diff= pred0_translates[:, np.newaxis,:] - gt1_translates[np.newaxis, :, :]
    translates_dist = np.linalg.norm(translates_diff, axis=2)   # (N_pred, N_gt)
    
    
    # match matrix - (N_pred = 1, N_gt = 1)
    matches_1 = np.logical_and(
        np.logical_or(angle_dist_1 < angle_th, angle_dist_2 < angle_th),
        translates_dist < dist_th
    )

    if test_width_type:
      if abs(grasp_widths_pred0 - grasp_widths_gt_obj1) > width_th: #TODO: or grasp_types_pred0 != grasp_types_gt_obj1:
        #print("bad matches01: ", matches_1)
        matches_1 = matches_1 & False;
        #print("matches false01: ", matches_1)


    # pred 1 vs gt0
    angle_dist_1 = self._get_SO3_dist(pred1_rotations, gt0_rotations_1)
    angle_dist_2 = self._get_SO3_dist(pred1_rotations, gt0_rotations_2)
    
    translates_diff = pred1_translates[:, np.newaxis,:] - gt0_translates[np.newaxis, :, :]
    translates_dist = np.linalg.norm(translates_diff, axis=2)   # (N_pred, N_gt)
  
    # match matrix - (N_pred = 1, N_gt = 1)
    matches_2 = np.logical_and(
        np.logical_or(angle_dist_1 < angle_th, angle_dist_2 < angle_th),
        translates_dist < dist_th
    )

    if test_width_type:
      if abs(grasp_widths_pred1 - grasp_widths_gt_obj0) > width_th: #TODO: or grasp_types_pred1 != grasp_types_gt_obj0:
        #print("bad matches10: ", matches_2)
        matches_2 = matches_2 & False;
        #print("matches false10: ", matches_2)


    match_01_10 = matches_1 & matches_2

    matches = match_01_10 | match_00_11
    #print("match_01_10====")
    #print(matches_1)
    #print(matches_2)
    #print(match_01_10)
    #print("==============")
    #print("===============")





    # get the success number
    pred_succ = np.any(matches, axis=1)
    gt_covered = np.any(matches, axis=0)
    assert pred_succ.size == pred_num
    assert gt_covered.size == gt_num

    if ignore_mask is None:
        ignore_mask = np.zeros(pred_num, dtype=bool)
    
    pred_succ_num = np.count_nonzero(pred_succ[np.logical_not(ignore_mask)])
    gt_cover_num = np.count_nonzero(gt_covered)

    # # debug
    # if np.any(~pred_succ) or np.any(~gt_covered):
    #     print("The angular distance: {}\n".format(angle_dist))
    #     print("\n The trace: {} \n".format(trace))

    #     print("The translate distance: {}".format(translates_dist))  
    return pred_num, pred_succ_num, gt_num, gt_cover_num, pred_succ, gt_covered
  

  def eval_box_matching(self, label:LabelScene, predict:ImageScene, dist_threshold = 0.02, angle_th=np.pi/4):
    """
    Box Matching
    Matching every predicted obj(3d bounding box) to label obj.
    Compare the label grasps in the label obj with the predicted grasps in the predict obj
    """
    label_obj_list = label.get_scene_obj_list() # 這個場警的所有label物件
    predict_obj_list = predict.get_scene_obj_list() #這個場景的所有predict物件

    labelobj_list = label.get_obj_position_list(type = "2d") #用2d的中心點座標來配對label跟predict
    imageobj_list = predict.get_obj_position_list(type = "2d")
    #print("labelobj_list:", labelobj_list)

    
    matching_list = []
    
    if(len(imageobj_list) == 0): #沒偵測到東西
      print("-----no object is detected-------")
       #沒偵測到東西的話 就代表這張測試資料的label全部都沒有cover到
      for label_obj in label_obj_list:
        s_g_label_num = len(label_obj.single_grasp_list)
        p_g_label_num = len(label_obj.paired_grasp_list)
        self.box_matching_s_g_gt_num += s_g_label_num
        self.box_matching_s_g_obj_gt_num[label_obj.cls]+= s_g_label_num

        self.box_matching_p_g_gt_num += p_g_label_num
        self.box_matching_p_g_obj_gt_num[label_obj.cls]+= p_g_label_num
      return
  
    #=======================================#
    # Matching label obj and predicted obj  #
    #=======================================#
    for label_obj_position in labelobj_list: #把label物件一個一個拿出來
      
      #配對predict和label物件

      image_label_obj_dist_list = [#計算每一個predict物件和我這一個label物件的2d中心點距離
        np.linalg.norm(label_obj_position - image_obj_position) for image_obj_position in imageobj_list
      ]
      argmin = np.argmin(image_label_obj_dist_list)# 和這個label距離最短的predict物件 
      matching_list.append(argmin)
      imageobj_list[argmin] = np.array([[999],[999],[999]],dtype = np.float32)


    # 配對完之後 若有多出來的predicted obj
    #為甚麼要做這個 因為下面的code是看每一個label obj 然後抓出對應的predict object 也就是說多餘的predict obj會被忽略
    
    if len(labelobj_list) < len(imageobj_list):
      #print("いらないもの！")
      for i in range(len(imageobj_list)):
        if i not in matching_list: #代表這一個編號的物件沒有被配對到任何一個label
          pred_obj = predict_obj_list[i]
          s_g_pred_num = len(pred_obj.single_grasp_list)
          p_g_pred_num = len(pred_obj.paired_grasp_list)
          self.box_matching_s_g_grasp_num += s_g_pred_num
          self.box_matching_s_g_obj_grasp_num[pred_obj.cls] += s_g_pred_num
          self.box_matching_p_g_grasp_num += p_g_pred_num 
          self.box_matching_p_g_obj_grasp_num[pred_obj.cls] += p_g_pred_num 
    
    #==================================================#
    # Calculate Single Grasp Accuracy GSR and GCR      #
    #==================================================#

    

    #print("match list:",matching_list)
    """
    obj_idx = 0
    for predict_obj in predict_obj_list:
      label_obj = None
      try:
        label_obj  = label_obj_list[matching_list[obj_idx]]
      except:
        #這個predict沒有找到對應的人 那他的predicted grasp就算錯
        self.box_matching_s_g_grasp_num += len(predict_obj.single_grasp_list)
        continue
    """

    obj_idx = 0
    for label_obj in label_obj_list: #拿出每一個label物件
      predict_obj = None
      try:
        predict_obj  = predict_obj_list[matching_list[obj_idx]] #找出對應的predict 物件
      except:
        #如果沒有找到predict object  代表gt全部沒cover到 把這個label的ground truth全部記錄下來 並加進metric的計算當中 然後continue不做這個迴圈
        #會跳到這裡的原因是因為 網路少偵測object 但不直接break迴圈 而是把迴圈跑完 把所有的label object的grasp label給記錄下來  
        self.box_matching_s_g_gt_num += len(predict_obj.single_grasp_list)
        self.box_matching_s_g_obj_gt_num[label_obj.cls] += len(predict_obj.single_grasp_list)
        continue
 
      #label_obj v.s. predict_obj
      
      # evaluate single grasp

      label_obj_single_grasp_poses = np.array([
        singlegrasp.get_grasp_info(type = "Mat")["T"] for singlegrasp in label_obj.single_grasp_list
      ])
      #print("label_obj_single_grasp_poses:", label_obj_single_grasp_poses)
      label_obj_single_widths = np.array([
        singlegrasp.get_grasp_info(type = "Mat")["width"] for singlegrasp in label_obj.single_grasp_list
      ])
      label_obj_single_types = np.array([
        singlegrasp.get_grasp_info(type = "Mat")["grasp_type"] for singlegrasp in label_obj.single_grasp_list
      ])
    
      predict_obj_single_grasp_poses = np.array([
        singlegrasp.get_grasp_info(type = "Mat")["T"] for singlegrasp in predict_obj.single_grasp_list
      ])
      #print("predict_obj_single_grasp_poses:", predict_obj_single_grasp_poses)
      predict_obj_single_widths = np.array([
        singlegrasp.get_grasp_info(type = "Mat")["width"] for singlegrasp in predict_obj.single_grasp_list
      ])
      predict_obj_single_types = np.array([
        singlegrasp.get_grasp_info(type = "Mat")["grasp_type"] for singlegrasp in label_obj.single_grasp_list
      ])


      pred_num, pred_succ_num, gt_num, gt_cover_num, pred_succ, gt_covered = None,None,None,None,None,None
      try:
        pred_num, pred_succ_num, gt_num, gt_cover_num, pred_succ, gt_covered = self._eval_grasps_with_gt(
          predict_obj_single_grasp_poses,
          predict_obj_single_widths,
          predict_obj_single_types,
          label_obj_single_grasp_poses,
          label_obj_single_widths,
          label_obj_single_types,
          dist_th=dist_threshold,
          angle_th=angle_th
        )
      except: #沒有預測出grasp的時候
        #print("except")
        pred_num, pred_succ_num, gt_num, gt_cover_num, pred_succ, gt_covered = 0,0,len(label_obj.single_grasp_list),0,False,False
      #print(pred_num, pred_succ_num, gt_num, gt_cover_num, pred_succ, gt_covered)

      self.box_matching_s_g_grasp_num += pred_num
      self.box_matching_s_g_grasp_success_num += pred_succ_num

      self.box_matching_s_g_obj_success_num[predict_obj.cls]+= pred_succ_num
      self.box_matching_s_g_obj_grasp_num[predict_obj.cls]+= pred_num

      #self.box_matching_s_g_gt_coveraged_rate = 0.0
      self.box_matching_s_g_gt_coveraged_num += gt_cover_num #pred
      self.box_matching_s_g_gt_num += gt_num

      self.box_matching_s_g_obj_gt_num[label_obj.cls]+= gt_num
      self.box_matching_s_g_obj_gt_coveraged_num[label_obj.cls]+= gt_cover_num
    
      #self.box_matching_s_g_obj_success_rate
      obj_idx+=1

    #TODO: evaluate two-hands grasp 
    #=======================================#
    # Calculate Paired Grasp Accuracy (GSR) #
    #=======================================#
    """
    obj_idx = 0
    for predict_obj in predict_obj_list:
      label_obj = None
      try:
        label_obj  = label_obj_list[matching_list[obj_idx]]
      except:
        #這個predict沒有找到對應的人 那他的predicted grasp就算錯
        self.box_matching_p_g_grasp_num += len(predict_obj.paired_grasp_list)
        continue
    """
    obj_idx = 0
    for label_obj in label_obj_list: #對所有的label 
      predict_obj = None
      try:
        predict_obj  = predict_obj_list[matching_list[obj_idx]] #找出他對應的predict物件
      except:
        #self.box_matching_p_g_gt_num += len(predict_obj.paired_grasp_list)
        #如果沒有找到predict object 直接跳過換下一個  因為這個是算GSR
        continue
      #在這個物件中 每一個grasp 一個一個下去比 

      for pred_p_g in predict_obj.paired_grasp_list:

        self.box_matching_p_g_grasp_num += 1
        self.box_matching_p_g_obj_grasp_num[predict_obj.cls]+=1
        #每一個預測的雙手抓取 和所有的雙手gt比

        pred_p_g_info = pred_p_g.get_twohands_grasps_info(type = "Mat")#把這個predict的雙手抓取的轉換矩陣 寬度跟抓取type拿出來
        
        pred_succ = False
        for label_p_g in label_obj.paired_grasp_list: #把label的雙手抓取一個一個抓出來 跟這個predict的雙手抓取比 一旦有一個比對成功 就跳出迴圈 
          label_p_g_info = label_p_g.get_twohands_grasps_info(type = "Mat")
          
          #這裡一定 是只有一個對一個 只要對了 就換下一個
          try:
            pred_num, pred_succ_num, gt_num, gt_cover_num, pred_succ, gt_covered = \
            self._eval_two_grasps_with_gt(
              grasp_poses_pred0 = np.expand_dims(pred_p_g_info['grasp0']['T'], axis=0), 
              grasp_widths_pred0 = np.expand_dims(pred_p_g_info['grasp0']['width'], axis=0), 
              grasp_types_pred0 = np.expand_dims(pred_p_g_info['grasp0']['grasp_type'], axis=0), 
              
              grasp_poses_pred1 = np.expand_dims(pred_p_g_info['grasp1']['T'], axis=0), 
              grasp_widths_pred1 = np.expand_dims(pred_p_g_info['grasp1']['width'], axis=0), 
              grasp_types_pred1 = np.expand_dims(pred_p_g_info['grasp1']['grasp_type'], axis=0), 

              grasp_poses_gt_obj0 = np.expand_dims(label_p_g_info['grasp0']['T'], axis=0), 
              grasp_widths_gt_obj0 = np.expand_dims(label_p_g_info['grasp0']['width'], axis=0),
              grasp_types_gt_obj0 = np.expand_dims(label_p_g_info['grasp0']['grasp_type'], axis=0),
              
              grasp_poses_gt_obj1 = np.expand_dims(label_p_g_info['grasp1']['T'], axis=0), 
              grasp_widths_gt_obj1 = np.expand_dims(label_p_g_info['grasp1']['width'], axis=0),
              grasp_types_gt_obj1 = np.expand_dims(label_p_g_info['grasp1']['grasp_type'], axis=0),
              
              dist_th=dist_threshold,
              angle_th=angle_th
            )
            #
            # 
            # self.box_matching_p_g_obj_grasp_success_num = [0.0 for i in range(opt.num_classes)]
            # self.box_matching_p_g_obj_grasp_num = [0.0 for i in range(opt.num_classes)]
            # self.box_matching_p_g_obj_grasp_success_rate = [0.0 for i in range(opt.num_classes)]

            
            
            if pred_succ: #只要抓取點有對應到一個label 就該break
              self.box_matching_p_g_grasp_success_num += 1
              #self.box_matching_p_g_obj_success_num = [0.0 for i in range(opt.num_classes)]
              self.box_matching_p_g_obj_grasp_success_num[predict_obj.cls]+=1
              
              #self.box_matching_p_g_obj_success_rate = [0.0 for i in range(opt.num_classes)]
              break
          
          except:
            #不做任何事情 會出事是因為預測抓取有甚麼問題 那他就一定跟所有的label都無法配對 最後迴圈跑完就會算一個失敗抓取
            continue
        #print("pred_succ = ",pred_succ)

      obj_idx += 1


    #=======================================#
    # Calculate Paired Grasp Accuracy (GCR) #
    #=======================================#
    """
    這是以每一個pred為主 和每一個gt一個個比對 我們只在意這個pred有沒有成功 不在意gt 被cover多少
    """

    #self.box_matching_p_g_grasp_num = 0 #pred
    #self.box_matching_p_g_grasp_success_num = 0
    obj_idx = 0
    #for predict_obj in predict_obj_list:
    #  label_obj  = label_obj_list[matching_list[obj_idx]]

    for label_obj in label_obj_list: #對每一個label 找出一個對應的 predict obj
      predict_obj = None
      try:
        predict_obj  = predict_obj_list[matching_list[obj_idx]]
      except:
        self.box_matching_p_g_gt_num += len(predict_obj.paired_grasp_list) #如果沒有找到對應的 predict obj 那代表這個物件的label全部沒有被cover
        self.box_matching_p_g_obj_gt_num[label_obj.cls] += len(predict_obj.paired_grasp_list)
        continue
      #在這個物件中 每一個grasp 一個一個下去比 

      #for pred_p_g in predict_obj.paired_grasp_list:
      for label_p_g in label_obj.paired_grasp_list:
        # 拿出這個object裡面的每一個gt grasp

        self.box_matching_p_g_gt_num += 1
        self.box_matching_p_g_obj_gt_num[label_obj.cls]+=1
         #每一個預測的雙手抓取 和所有的雙手gt比
        label_p_g_info = label_p_g.get_twohands_grasps_info(type = "Mat")
        label_succ = False

        #for label_p_g in label_obj.paired_grasp_list:
        for pred_p_g in predict_obj.paired_grasp_list:
        # 把這個gt grasp 和所有的 predict grasp 比較

          pred_p_g_info = pred_p_g.get_twohands_grasps_info(type = "Mat")
          
          #這裡一定 是只有一個對一個 只要對了 就換下一個
          try:
            #NOTE: =======================================================#
            #  只要把predict 的地方放label 把label的地方放predict 就可以測GCR了
            #=============================================================#
            pred_num, pred_succ_num, gt_num, gt_cover_num, label_succ, gt_covered = \
            self._eval_two_grasps_with_gt(
              grasp_poses_gt_obj0 = np.expand_dims(pred_p_g_info['grasp0']['T'], axis=0), 
              grasp_widths_gt_obj0 = np.expand_dims(pred_p_g_info['grasp0']['width'], axis=0), 
              grasp_types_gt_obj0 = np.expand_dims(pred_p_g_info['grasp0']['grasp_type'], axis=0), 
              
              grasp_poses_gt_obj1 = np.expand_dims(pred_p_g_info['grasp1']['T'], axis=0), 
              grasp_widths_gt_obj1 = np.expand_dims(pred_p_g_info['grasp1']['width'], axis=0), 
              grasp_types_gt_obj1 = np.expand_dims(pred_p_g_info['grasp1']['grasp_type'], axis=0), 


              grasp_poses_pred0 = np.expand_dims(label_p_g_info['grasp0']['T'], axis=0), 
              grasp_widths_pred0 = np.expand_dims(label_p_g_info['grasp0']['width'], axis=0),
              grasp_types_pred0 = np.expand_dims(label_p_g_info['grasp0']['grasp_type'], axis=0),
              
              grasp_poses_pred1 = np.expand_dims(label_p_g_info['grasp1']['T'], axis=0), 
              grasp_widths_pred1 = np.expand_dims(label_p_g_info['grasp1']['width'], axis=0),
              grasp_types_pred1 = np.expand_dims(label_p_g_info['grasp1']['grasp_type'], axis=0),
              
              dist_th=dist_threshold,
              angle_th=angle_th
            )
            #
            # 
            # self.box_matching_p_g_obj_grasp_success_num = [0.0 for i in range(opt.num_classes)]
            # self.box_matching_p_g_obj_grasp_num = [0.0 for i in range(opt.num_classes)]
            # self.box_matching_p_g_obj_grasp_success_rate = [0.0 for i in range(opt.num_classes)]

            
            
            if label_succ: #只要label抓取點有對應到一個pred 就該break
              self.box_matching_p_g_gt_coveraged_num += 1
              #self.box_matching_p_g_obj_success_num = [0.0 for i in range(opt.num_classes)]
              self.box_matching_p_g_obj_gt_coveraged_num[label_obj.cls]+=1
              
              #self.box_matching_p_g_obj_success_rate = [0.0 for i in range(opt.num_classes)]
              break
          
          except:
            continue
          
      obj_idx += 1

    return matching_list
  

  def get_acc(self):
    self.box_matching_s_g_grasp_acc = self._to_float(
       self.box_matching_s_g_grasp_success_num / self.box_matching_s_g_grasp_num 
    )

    self.box_matching_s_g_obj_success_rate = [
      self._to_float(x/y) for x,y in zip(self.box_matching_s_g_obj_success_num, self.box_matching_s_g_obj_grasp_num) if y>0
    ]
    
    self.box_matching_s_g_gt_coveraged_rate = self._to_float(
       self.box_matching_s_g_gt_coveraged_num / self.box_matching_s_g_gt_num
    )

    self.box_matching_s_g_obj_gt_coveraged_rate = [
      self._to_float(x/y) for x,y in zip(self.box_matching_s_g_obj_gt_coveraged_num, self.box_matching_s_g_obj_gt_num) if y>0
    ]



    self.box_matching_p_g_grasp_acc = self._to_float(
       self.box_matching_p_g_grasp_success_num / self.box_matching_p_g_grasp_num 
    )

    self.box_matching_p_g_obj_grasp_success_rate = [
       self._to_float(x/y) for x,y in zip(self.box_matching_p_g_obj_grasp_success_num, self.box_matching_p_g_obj_grasp_num) if y>0
       
    ]

    self.box_matching_p_g_gt_coveraged_rate = self._to_float(
       self.box_matching_p_g_gt_coveraged_num / self.box_matching_p_g_gt_num
    )

    self.box_matching_p_g_obj_gt_coveraged_rate = [
      self._to_float(x/y) for x,y in zip(self.box_matching_p_g_obj_gt_coveraged_num, self.box_matching_p_g_obj_gt_num) if y>0
    ]
    
    print("Accurcy:")
    print("Single-Hand Grasp GSR:", self.box_matching_s_g_grasp_acc)
    print("Single-Hand Grasp GCR:", self.box_matching_s_g_gt_coveraged_rate)
    
    print("Two-Hands Grasp GSR:", self.box_matching_p_g_grasp_acc)
    print("Two-Hand Grasp GCR:", self.box_matching_p_g_gt_coveraged_rate)
    #print("box_matching_s_g_obj_acc", self.box_matching_s_g_obj_success_rate)
    """
    print("Single-Hand Grasp Object GSR:")
    for i in range(len(self.box_matching_s_g_obj_success_rate)):
      print(f"|{self.opt.class_name[i]}:{self.box_matching_s_g_obj_success_rate[i]}|success_num = {self.box_matching_s_g_obj_success_num[i]}|grasp_num = {self.box_matching_s_g_obj_grasp_num[i]}|")  
    
    print("Single-Hand Grasp Object GCR:")
    for i in range(len(self.box_matching_s_g_obj_gt_coveraged_rate)):
      print(f"|{self.opt.class_name[i]}:{self.box_matching_s_g_obj_gt_coveraged_rate[i]}|success_num = {self.box_matching_s_g_obj_gt_coveraged_num[i]}|grasp_num = {self.box_matching_s_g_obj_gt_num[i]}|")  
    
    
    
    print("Two-Hand Grasp Object GSR:")
    for i in range(len(self.box_matching_p_g_obj_grasp_success_rate)):
      print(f"|{self.opt.class_name[i]}:{self.box_matching_p_g_obj_grasp_success_rate[i]}|success_num = {self.box_matching_p_g_obj_grasp_success_num[i]}|grasp_num = {self.box_matching_p_g_obj_grasp_num[i]}|")  
    
    print("Two-Hand Grasp Object GCR:")
    for i in range(len(self.box_matching_p_g_obj_gt_coveraged_rate)):
      print(f"|{self.opt.class_name[i]}:{self.box_matching_p_g_obj_gt_coveraged_rate[i]}|success_num = {self.box_matching_p_g_obj_gt_coveraged_num[i]}|grasp_num = {self.box_matching_p_g_obj_gt_num[i]}|")  
    """
    
    




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

  
  evaluator = Evaluator(opt)
  if opt.demo == 'Dataset':
    data_dir_pth = "./lib/data/foup_grasp_dataset"
    Dataset = grasp_test_dataset(data_dir_pth, opt)
    
    dist_th_list = [0.02,0.03,0.04]
    angle_th_list = [np.pi/12, np.pi/8, np.pi/4]

    id = 0
    for dist_th in dist_th_list:
      angle_th = angle_th_list[id]
      print("-----------------------------------------")
      print("angle threshold:",angle_th/np.pi, "pi")
      print("distance threshold:", dist_th)
      print("-----------------------------------------")
      for iter in range(len(Dataset)):
        batch = Dataset[iter]
        meta, labelscene = batch
        #labelscene.scene_imshow(rotate=True)
        image = meta['image']
      
        
        imagescene = detector.run(image,meta)

        evaluator.eval_box_matching(labelscene, imagescene, dist_threshold = dist_th, angle_th = angle_th)
        #Scene.scene_imshow()
        #print("+++++++++++++++++++++++++++++++++++++++")
        #img = imagescene.scene_imshow()
        #cv2.imwrite(os.path.join(PATH, f"ALL_{iter}.png"), img)
        #print("+++++++++++++++++++++++++++++++++++++++")
      evaluator.get_acc()
      id+=1

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
