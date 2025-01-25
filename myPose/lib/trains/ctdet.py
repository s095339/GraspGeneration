from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss,CrossEntropyLoss,GraspSmoothL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

#mycode
from models.losses import SmoothL1Loss
#
class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

    #Mycode: it is difference from CenterPose
    self.crit_kps = SmoothL1Loss()
    self.crit_scale = SmoothL1Loss()

    self.crit_grasp_hm = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_grasp_kps = GraspSmoothL1Loss()#抓取中心點和各個keypoint的位移
    self.crit_grasp_width =  SmoothL1Loss()
    self.crit_grasp_type = CrossEntropyLoss()
    self.crit_grasp_ctdsp=  SmoothL1Loss() # 抓取中心到物件中心的位移
    #========================================
  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss, kps_loss, scale_loss= 0, 0, 0, 0, 0
    grasp_hm_loss, grasp_kps_loss, grasp_type_loss, grasp_width_loss,grasp_ctdsp_loss = 0, 0, 0, 0, 0
    grasp_center_off_loss = 0
    for s in range(opt.num_stacks): # opt.num_stacks = 2 if opt.arch == 'hourglass' else 1
      output = outputs[s]
      #print(type(output))
      #for k,v in output.items():
      #  print(k, "=", v.shape)
      """
      hm = torch.Size([8, 1, 128, 128])
      wh = torch.Size([8, 2, 128, 128])
      reg = torch.Size([8, 2, 128, 128])
      """
      
    #=====================================
    # 1. Centerpoint                     #
    #=====================================
      if not opt.mse_loss: #focal loss
        output['hm'] = _sigmoid(output['hm']) #1. center heatmap loss


      #oracle : 這邊是只ground truth 的意思
      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks


    #=====================================
    # 2. 2d bbox wh                      #
    #=====================================
      

      if opt.wh_weight > 0:
        #會進來
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) / 
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          #進來這裡
          #2. bounding box dimension loss
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / opt.num_stacks
    
    #=====================================
    # 3. centerpoint subpixel offset     #
    #=====================================
          
      if opt.reg_offset and opt.off_weight > 0:
        #3. center point subpixel offset loss
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks
    
    #=========================================
    # 4. center-keypoint displacement(mycode)#
    #=========================================
      kps_loss += self.crit_kps(output['kps'], batch['kps_mask'],
                              batch['ind'], batch['kps']) / opt.num_stacks

    

    #=============================================
    # 5. Relative size of object (from centerpose)
    #=============================================
      if self.opt.depth:
        #如果有深度訊息，就讓網路預測實際size
        #print("scale:")
        #print(batch['absolute_scale'].shape)
        scale_loss += self.crit_scale(output['scale'], batch['scale_mask'],
                                  batch['ind'], batch['absolute_scale'])/ opt.num_stacks
      else:
        scale_loss += self.crit_scale(output['scale'], batch['scale_mask'],
                                  batch['ind'], batch['relative_scale'])/ opt.num_stacks
    
    #grasp
    #===============================================
    # 6. Grasp Center heatmap
    #===============================================
      if not opt.mse_loss: #focal loss
        output['single_grasp_center_hm'] = _sigmoid(output['single_grasp_center_hm']) 
        output['paired_grasp_center_hm'] = _sigmoid(output['paired_grasp_center_hm']) 

      #oracle : 這邊是只ground truth 的意思
      if opt.eval_oracle_hm:
        output['single_grasp_center_hm'] = batch['single_grasp_center_hm']
        output['paired_grasp_center_hm'] = batch['paired_grasp_center_hm']
      

      grasp_hm_loss += self.crit_grasp_hm(output['single_grasp_center_hm'], batch['single_grasp_center_hm']) / opt.num_stacks
      grasp_hm_loss += self.crit_grasp_hm(output['paired_grasp_center_hm'], batch['paired_grasp_center_hm']) / opt.num_stacks
    
    #===============================================
    # 7. Grasp kpt dsp
    #===============================================
      grasp_kps_loss += self.crit_grasp_kps(output['single_grasp_ct_kps_dsp'], batch['single_grasp_kpt_mask'],
                              batch['single_grasp_ind'], batch['single_grasp_ct_kps_dsp']) / opt.num_stacks
      # 兩隻手互換位置也OK的意思
      grasp_kps_loss += torch.minimum( 
                              (self.crit_grasp_kps(output['paired_grasp_ct_kps_dsp'], batch['paired_grasp_kpt_mask'],
                              batch['paired_grasp_ind'], batch['paired_grasp_ct_kps_dsp']) / opt.num_stacks)
                              ,
                              (self.crit_grasp_kps(output['paired_grasp_ct_kps_dsp'], batch['paired_grasp_kpt_mask'],
                              batch['paired_grasp_ind'], batch['paired_grasp_ct_kps_dsp_reverse']) / opt.num_stacks)
                              )
    
    #===============================================
    # 8. grasp type
    #===============================================
      grasp_type_loss += self.crit_grasp_type(output['single_grasp_type'], batch['single_grasp_kpt_mask'],
                              batch['single_grasp_ind'], batch['single_grasp_type']) / opt.num_stacks
      grasp_type_loss += self.crit_grasp_type(output['paired_grasp_type'], batch['paired_grasp_kpt_mask'],
                              batch['paired_grasp_ind'], batch['paired_grasp_type']) / opt.num_stacks
    
    #===============================================
    # 9. grasp width
    #===============================================
      grasp_width_loss += self.crit_grasp_width(output['single_grasp_width'], batch['single_grasp_kpt_mask'],
                              batch['single_grasp_ind'], batch['single_grasp_width']) / opt.num_stacks
      grasp_width_loss += self.crit_grasp_width(output['paired_grasp_width'], batch['paired_grasp_kpt_mask'],
                              batch['paired_grasp_ind'], batch['paired_grasp_width']) / opt.num_stacks
      
    #===============================================
    # 10. grasp ct to object ct dsp
    #===============================================
      grasp_ctdsp_loss += self.crit_grasp_ctdsp(output['single_graspct_objct_dsp'], batch['single_grasp_kpt_mask'],
                              batch['single_grasp_ind'], batch['single_graspct_objct_dsp']) / opt.num_stacks
      grasp_ctdsp_loss += self.crit_grasp_ctdsp(output['paired_graspct_objct_dsp'], batch['paired_grasp_kpt_mask'],
                              batch['paired_grasp_ind'], batch['paired_graspct_objct_dsp']) / opt.num_stacks

    #===============================================
    # 11. grasp ct offset
    #===============================================
      grasp_center_off_loss += self.crit_reg(output['single_grasp_center_offset'], batch['single_grasp_kpt_mask'],
                              batch['single_grasp_ind'], batch['single_grasp_center_offset']) / opt.num_stacks
      grasp_center_off_loss += self.crit_reg(output['paired_grasp_center_offset'], batch['paired_grasp_kpt_mask'],
                              batch['paired_grasp_ind'], batch['paired_grasp_center_offset']) / opt.num_stacks
    scale_weight = 0.8
    kps_weight = 1

    
    loss = opt.hm_weight * hm_loss  + \
           opt.off_weight * off_loss + \
           opt.hm_weight * grasp_hm_loss + grasp_kps_loss + grasp_type_loss + grasp_width_loss + grasp_ctdsp_loss+\
           opt.off_weight * grasp_center_off_loss
    
    if opt.use_3d_bounding_box_loss == True:
      loss += opt.wh_weight * wh_loss + kps_weight * kps_loss + scale_weight * scale_loss 
    
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,'off_loss': off_loss,
                  #'wh_loss': wh_loss,  'kps_loss' : kps_loss,
                  #'scale_loss': scale_loss,
                  
                  'grasp_hm_loss':grasp_hm_loss,
                  'grasp_kps_loss':grasp_kps_loss, 
                  'grasp_type_loss':grasp_type_loss, 
                  'grasp_width_loss':grasp_width_loss,
                  'grasp_ctdsp_loss':grasp_ctdsp_loss,
                  'grasp_ct_off_loss':grasp_center_off_loss
                  }
    if opt.use_3d_bounding_box_loss == True:
      #'wh_loss': wh_loss,  'kps_loss' : kps_loss,
                  #'scale_loss': scale_loss,
      loss_stats['wh_loss'] = wh_loss
      loss_stats['kps_loss'] = kps_loss
      loss_stats['scale_loss']= scale_loss
    
    return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    if opt.use_3d_bounding_box_loss == True:
      loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'kps_loss', 'scale_loss',\
                   'grasp_hm_loss', 'grasp_kps_loss', 'grasp_type_loss', 'grasp_width_loss','grasp_ctdsp_loss',\
                   'grasp_ct_off_loss'
                   ]
    else:
      loss_states = ['loss', 'hm_loss', 'off_loss',\
                   'grasp_hm_loss', 'grasp_kps_loss', 'grasp_type_loss', 'grasp_width_loss','grasp_ctdsp_loss',\
                   'grasp_ct_off_loss'
                   ]
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]