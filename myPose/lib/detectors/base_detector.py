from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
from scipy.special import softmax
#from centerpose================

#===========================


#====================================
from pose_solver.pose_recover_foup import PoseSolverFoup
from pose_solver.cubic_bbox_foup import Axis
from pose_solver.foup_pnp_shell import pnp_shell
#====================================

#====================================
from PoseGrasp.scene import ImageScene 
from PoseGrasp.Obj_Grasp import SceneObj,SingleHandGrasp,TwoHandsGrasp
#====================================

#mycode
import json
#
class BaseDetector(object):
  def __init__(self, opt):
    self.opt = opt
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')

    inp_channel = 3
    if opt.depth:
      print("detector input RGBD data")
      inp_channel = 4
    else:
      print("detector input RGB data")
      
    self.model = create_model(opt.arch, opt.heads, opt.head_conv, inp_channel, opt = opt)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    if not opt.depth:
      self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
      self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    else:
      self.mean = np.array(opt.mean_rgbd, dtype=np.float32).reshape(1, 1, 4)
      self.std = np.array(opt.std_rgbd, dtype=np.float32).reshape(1, 1, 4)

    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale, meta=None):

    # #scale = 1
    #print("input image.shape =", image.shape) 

    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      #程式會進來這裡
      inp_height, inp_width = self.opt.input_h, self.opt.input_w #512 512

      #因為scale是1 s就是原本輸入圖片的shape c就是原本輸入圖片的中心點
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    #和dataset 一樣 把輸入image做彷射變換
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    
    #彷射變換
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    
    #做一些regulazation
    if not self.opt.depth:
      inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
    else:
      inp_image[:, :, :3] = inp_image[:, :, :3]  / 255.
      inp_image[:, :, 3] = inp_image[:, :, 3]  / 65535.
      #print("color max: ", inp_image[:, :, :3].max())
      #print("depth max: ", inp_image[:, :, 3].max())
      inp_image = ((inp_image-self.mean)/self.std).astype(np.float32)

      
    

    if not self.opt.depth:
      images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    else:
      images = inp_image.transpose(2, 0, 1).reshape(1, 4, inp_height, inp_width)


    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)

    
    meta = {'c': c, 's': s, 
          'out_height': inp_height // self.opt.down_ratio, 
          'out_width': inp_width // self.opt.down_ratio}
    #print("prepross meta = ", meta)
    return images, meta
   
    
    
    #prepross meta =  {'c': [300., 400.], 's': 800.0, 'out_height': 128, 'out_width': 128}

    

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1, ct_coor_results = None):
    raise NotImplementedError

  def merge_outputs(self, detections, ct_coor_results):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results, ct_coor_results):
   raise NotImplementedError
  #mycode=====================
  def show_groudtruth_box(self,image, anns):
    num_objs = len(anns['objects'])
    for obj_idx in range(num_objs):
      ann = anns['objects'][obj_idx]
      nine_keypoint_anno = np.array(ann['projected_cuboid'])# 包含中心點在內的9個keypoints

      key_point = np.array(nine_keypoint_anno)
    

      for i in range(key_point.shape[0]):
        pt = key_point[i]
        cv2.circle(image, pt, 3, [128,127,35], 3)
        if(i>0):cv2.putText(image, str(i), pt, 2 , 2 , [0, 200 , 250], 2)
        
      #cv2.circle(image, pt, 2, c, 2)
      
      cv2.line(image, key_point[1], key_point[2], [255,0,0],2)
      cv2.line(image, key_point[1], key_point[5], [255,0,0],2)
      cv2.line(image, key_point[6], key_point[2], [255,0,0],2)
      cv2.line(image, key_point[5], key_point[6], [255,0,0],2)

      cv2.line(image, key_point[3], key_point[7], [0,255,0],2)
      cv2.line(image, key_point[4], key_point[8], [0,255,0],2)
      cv2.line(image, key_point[3], key_point[4], [0,255,0],2)
      cv2.line(image, key_point[7], key_point[8], [0,255,0],2)

      cv2.line(image, key_point[1], key_point[3], [0,0,255],2)
      cv2.line(image, key_point[4], key_point[2], [0,0,255],2)
      cv2.line(image, key_point[8], key_point[6], [0,0,255],2)
      cv2.line(image, key_point[7], key_point[5], [0,0,255],2)
    
    cv2.imshow("gt", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
  #===========================
  def run(self, image_or_path_or_tensor, meta=None):
    #這邊開始做inference meta沒東西

    """
    [因為懶得改名]
    若網路有使用深度資訊的話，那麼這裡的relative_scale, re_scale之類的東西 通通都是absolute_scale
    
    """
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False

    #把圖片讀出來=========================================================
    anno_pth = ""
    anns = None

    #因為pre_process會把meta裡的資料洗掉所以這邊些讀出來 之後再存回去
    #有點笨的作法
    color_img = meta['color_img']
    camera_instrinic = meta['intrinsic']
    width = meta['width']
    height = meta['height'] 


    if isinstance(image_or_path_or_tensor, np.ndarray): #image_or_path_or_tensor 是不是 np.ndarray這個型別
      image = image_or_path_or_tensor
      image_rgb = image[:,:,:3].astype(np.uint8)
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
      #mycode------------------------------------------------------
      anno_pth = image_or_path_or_tensor.replace('.png','.json')
      with open(anno_pth) as f:
          anns = json.load(f)
      #camera_instrinic = anns['camera_data']["intrinsics"]

      #self.show_groudtruth_box(image, anns)

      
      #------------------------------------------------------------
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    #======================================================================



    #計算時間====================================
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    #===========================================


    detections = []
    #mycode=
    ct_and_kps_coor = []
    relative_scale_dets = []

    single_grasp_ct_kps_dsp_dets = []
    single_grasp_type_dets = []
    single_grasp_width_dets = []
    single_grasp_objct_ct_dets = []
    
    paired_grasp_ct_kps_dsp_dets = []
    paired_grasp_type_dets = []
    paired_grasp_width_dets = []
    paired_grasp_objct_ct_dets = []
    #=
    for scale in self.scales:
      scale_start_time = time.time()
      
      #把圖片做一些縮放並用meta記錄他的性質
      if not pre_processed:
        #把圖片做一些縮放而已
        images, meta = self.pre_process(image, scale, meta) #scale = 1
        meta['intrinsic'] = camera_instrinic
        meta['width'] = width
        meta['height'] = height
        meta['color_img'] = color_img
        #meta =  {'c': [300., 400.], 's': 800.0, 'out_height': 128, 'out_width': 128}
        #meta紀錄了原本圖片的中心，原圖寬高比較大的那個，feature輸出的寬高
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}

      #丟入網路並開始計時
      images = images.to(self.opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      #丟入網路預測==================================================
      #output, dets, forward_time = self.process(images, return_time=True) #原本的
      output, dets, forward_time, center_and_kps_coor, relative_scale,\
      single_grasp_ct_kps_dsp, single_grasp_type, single_grasp_width,single_grasp_objct_ct,\
      paired_grasp_ct_kps_dsp, paired_grasp_type, paired_grasp_width,paired_grasp_objct_ct  = self.process(images, return_time=True) #我的
      #=============================================================
      #print("center_and_coor:", center_and_kps_coor.shape) #[batch, k, 18]
      #output是網路的output # dets = [batch. K, 6] 6分別是: bbox的左上x 左上y 右下x 右下y score classes



      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)
      
      #應該是把圖片轉回原本的大小 所以點點也要轉回去=========
      #dets = self.post_process(dets, meta, scale) #原本的
      #mycode
      
      dets, center_and_kps_coor, relative_scale,\
      single_grasp_ct_kps_dsp, single_grasp_type, single_grasp_width,single_grasp_objct_ct,\
      paired_grasp_ct_kps_dsp, paired_grasp_type, paired_grasp_width,paired_grasp_objct_ct \
      = self.post_process(dets, meta, scale, center_and_kps_coor, relative_scale, 
                          single_grasp_ct_kps_dsp, single_grasp_type, single_grasp_width,single_grasp_objct_ct,\
                          paired_grasp_ct_kps_dsp, paired_grasp_type, paired_grasp_width,paired_grasp_objct_ct 
                          )
      #==================================================
      

      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)
      #mycode=
      ct_and_kps_coor.append(center_and_kps_coor)
      relative_scale_dets.append(relative_scale)

      single_grasp_ct_kps_dsp_dets.append(single_grasp_ct_kps_dsp) 
      single_grasp_type_dets.append( single_grasp_type)
      single_grasp_width_dets.append( single_grasp_width)
      single_grasp_objct_ct_dets.append( single_grasp_objct_ct)
      
      paired_grasp_ct_kps_dsp_dets.append(paired_grasp_ct_kps_dsp)
      paired_grasp_type_dets.append(paired_grasp_type)
      paired_grasp_width_dets.append( paired_grasp_width)
      paired_grasp_objct_ct_dets.append(paired_grasp_objct_ct)

      #=

    #原本的:
    #results = self.merge_outputs(detections)
    #mycode===
      results , ct_kps_coor_results, relative_scale_results,\
      single_grasp_ct_kps_dsp_results,single_grasp_type_results,single_grasp_width_results,single_grasp_objct_ct_results,\
      paired_grasp_ct_kps_dsp_results,paired_grasp_type_results,paired_grasp_width_results,paired_grasp_objct_ct_results\
      = self.merge_outputs(
        detections, ct_and_kps_coor, relative_scale_dets,
        single_grasp_ct_kps_dsp_dets,
        single_grasp_type_dets,
        single_grasp_width_dets,
        single_grasp_objct_ct_dets,
        
        paired_grasp_ct_kps_dsp_dets ,
        paired_grasp_type_dets,
        paired_grasp_width_dets ,
        paired_grasp_objct_ct_dets 
        )
    #把所有scale的輸出 按照class聚集起來  並且將輸出限制在一定範圍內=>這句話好像沒用
    
    # 2024/8/25 凌晨改道這邊
    
    #======

    #print("relative_scale_result", relative_scale_results[1].shape) #[K,3]
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time
    


    """
    results裡面的東西：對這一張圖片的預測

    results[class][boxes]
    {
      "1":  list of boxes
      "2":  list of boxes
    
    
    }
    """ 

    if self.opt.debug == 1:
      #原本的
      #self.show_results(debugger, image, results)
      #mycode==
      print("==========================Solving PnP==========================")
      self.show_results(debugger, image_rgb, results, ct_kps_coor_results, relative_scale_results, camera_instrinic,
                        #grasp
                        single_grasp_ct_kps_dsp_results,single_grasp_type_results,
                        single_grasp_width_results,single_grasp_objct_ct_results,\
                        paired_grasp_ct_kps_dsp_results,paired_grasp_type_results,
                        paired_grasp_width_results,paired_grasp_objct_ct_results
                        
                        )
      #========

      return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time, 'ct_kps_coor_results':ct_kps_coor_results}
    
    elif self.opt.debug == 2:#3d iou
      
      cls_list = []
      results_eval = []
      visibility = []
      #scale_list = []
      box_scale_list = []
      """
      一再提醒 若有使用depth 那麼這邊的relative_scale都是absolute_scale
      """
      bboxs = [] #foup_test_dataset的

      #for j in range(1, self.num_classes + 1):
      for j in range(1, self.num_classes + 1):
        #每一個類別的
        for bbox_idx in range(len(results[j])):
          #每一個箱子

          predict_bbox = results[j][bbox_idx]
          pts = ct_kps_coor_results[j][bbox_idx]
          re_scale = relative_scale_results[j][bbox_idx]
          #print("pts shape = ", pts.shape)
          #print("re_scale=", re_scale)
          #仿造foup_test_dataset======
          bbox = {}
          bbox['kps'] = pts
          bbox['abs_scale'] = re_scale #不要懷疑 這只是為了讓程式可以過 #TODO 以後要改掉
          
          box_scale = None
          if not self.opt.depth:
            box_scale = re_scale/re_scale[1]
          else:
            box_scale = re_scale #其實是absolute scale
            
          #===========================
          if predict_bbox[4] > self.opt.vis_thresh: #把輸出設了一個thresh
            try:
              projected_points, point_3d_cam_pred, _, _, bbox, tvec, rvec  = \
                      pnp_shell(meta, bbox, pts, box_scale) 
              nine_keypoint_2d_pred = np.array(pts, dtype = int).reshape(-1,2).tolist()

              cls_list.append(j-1)
              results_eval.append(   
                                  (np.array(projected_points), point_3d_cam_pred)
                                  )
            
              #scale_list.append(abs_scale)
              box_scale_list.append(box_scale)
              bboxs.append(bbox)
            except:
              continue
              #print("yabai   ")
              #print(pts[2:]) 
              
              #raise ValueError("pause")
            

          else:
            pass



      time_result = {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time, 'ct_kps_coor_results':ct_kps_coor_results       
                
              }
    
      ret = {
        'image':image,
        'cls':cls_list,
        'results':results_eval,
        'bboxs': bboxs
      }
      #print(cls_list)
      #print("detect results:", results_eval[0])
      return time_result, ret
    
    else: #這次的
      #print("這次的")
      image_scene = ImageScene(meta = meta)
      image_scene.say_hello()
      
      obj_list = []
      box_center_list = [] 
      box_id = 0
      for j in range(1, self.num_classes + 1):
        #每一個類別的
        for bbox_idx in range(len(results[j])):
          #每一個箱子
          predict_bbox = results[j][bbox_idx]
          pts = ct_kps_coor_results[j][bbox_idx]
          re_scale = relative_scale_results[j][bbox_idx]
          bbox = {}
          bbox['kps'] = pts
          bbox['abs_scale'] = re_scale #不要懷疑 這只是為了讓程式可以過 #TODO 以後要改掉
          box_scale = None
          if not self.opt.depth:
            box_scale = re_scale/re_scale[1]
          else:
            box_scale = re_scale #其實是absolute scale
          #===========================
          if predict_bbox[4] > self.opt.vis_thresh: #把輸出設了一個thresh
            
              #print("pts.shape:", pts.shape)#torch.Size([1, 2, 18])
            pts = np.array(pts).reshape((9,2))
            #print("pts.shape:", pts.shape)
            center_point_2d = pts[0]
           

            image_scene.add_obj(
              SceneObj(
                center_point_2d = pts[0],
                keypoint8_2d= pts[1:],
                size = box_scale,
                cls = j-1,
                meta = meta,
                bbox = bbox
              )
            )
              
            

      
      for j in range(1,2):

        for sg_idx in range(len(single_grasp_ct_kps_dsp_results[j])):
          pts = single_grasp_ct_kps_dsp_results[j][sg_idx] 
          pts = np.array(pts).reshape((5,2))
          cls = np.argmax(softmax(single_grasp_type_results[j][sg_idx]))
          obj_ct_pred = single_grasp_objct_ct_results[j][sg_idx] 
          width = single_grasp_width_results[j][sg_idx] 

          image_scene.add_scene_singlehand_grasp(
            SingleHandGrasp(
              center_point_2d=pts[0],
              keypoint4_2d=pts[1:],
              cls = cls,
              meta = meta,
              obj_ct_pred = obj_ct_pred,
              width = width
            )
          )
          
      for j in range(1,2):
        for pg_idx in range(len(paired_grasp_ct_kps_dsp_results[j])):
          pts = paired_grasp_ct_kps_dsp_results[j][pg_idx]
          pts = np.array(pts).reshape((9,2))
          cls = np.argmax(softmax(paired_grasp_type_results[j][pg_idx]))
          obj_ct_pred = paired_grasp_objct_ct_results[j][pg_idx]
          width0, width1 = paired_grasp_width_results[j][pg_idx]
          image_scene.add_scene_twohands_grasp(
            TwoHandsGrasp(
              center_point_2d=pts[0],
              keypoint8_2d=pts[1:],
              cls = cls,
              meta = meta,
              obj_ct_pred = obj_ct_pred,
              width0 = width0,
              width1 = width1
            )
          )

        #after collecting all objs and grasps appear in scene
        #apply obj grasp matching

      image_scene.scene_obj_grasp_match()
        #print("+++++++++++++++++++++++++++++++++++++++")
        #image_scene.scene_info_show()
        #print("+++++++++++++++++++++++++++++++++++++++")
      return image_scene