from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    #for 偵測center point subpixel offset============
    """
    feat => reg = torch.Size([8, 128 x 128, 2]) 
    ind = torch.Size(batch = 8, max_objs = 10)
    """
    #=============================
    #print("feat.size()",feat.size())
    dim  = feat.size(2) #dim = 2 也就是offset的x跟y
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    #print("ind expand", ind.shape)
    #ind.unsqueeze => torch.Size([8,10,1]) 
    #expand => torch.Size([8, 10, 2])

    #hint: torch.tensor[dim=0][dim=1][dim=2]
    feat = feat.gather(1, ind)#torch.Size([8, 128 x 128, 2]) 
                              #torch.Size([8, 10, 2])    沿著dim=1做gather

    #print("after gather")
    #print(feat.shape)

    #feat: torch.Size([8, 10, 2])

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    #for 偵測center point subpixel offset============
    """
    output(feat)
    feat = output[reg]
    reg = torch.Size([8, 2, 128, 128])
    
    ind = torch.Size(batch = 8, max_objs = 10)
    """
    #=============================
    feat = feat.permute(0, 2, 3, 1).contiguous() #reg = torch.Size([8, 128, 128, 2])
    feat = feat.view(feat.size(0), -1, feat.size(3))#reg = torch.Size([8, 128 x 128, 2]) 把她拉長 
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)