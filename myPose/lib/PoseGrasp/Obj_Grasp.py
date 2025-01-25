import numpy as np
#====================================
from pose_solver.pose_recover_foup import PoseSolverFoup
from pose_solver.cubic_bbox_foup import Axis
from pose_solver.foup_pnp_shell import pnp_shell

from pose_solver.pose_recover_grasp import PoseSolverGrasp
from pose_solver.grasp_frame import Grasp3d as Grasp_Frame
#====================================

import cv2

class SceneObj:
    def __init__(self, 
        center_point_2d,
        keypoint8_2d,     
        size,           
        cls,
        meta,
        bbox,
        # optional
        center_point_3d = None,
        keypoint8_3d = None,
        tvec = None,
        rvec = None
    ):
        #print("Obj created")  
        self.center_point_2d = center_point_2d
        self.keypoint8_2d = keypoint8_2d
        self.cls = cls
        self.size = size
        self.meta = meta
        self.bbox = bbox
        self.single_grasp_list = []
        self.paired_grasp_list = []
        if(center_point_3d != None):
            self.center_point_3d = center_point_3d,
            self.keypoint8_3d = keypoint8_3d,
            self.tvec = tvec,
            self.rvec = rvec
        self.calculate_pose()
    def get_2d_center_point(self):
        return self.center_point_2d
    def calculate_pose(self):
        box_pose_solver = PoseSolverFoup(self.meta["intrinsic"])
        ret, tvec, rvec, projected_points, reprojectionError, quaternion =\
            box_pose_solver.solve(self.keypoint8_2d, self.size, use_ct = False)
        
        self.tvec = tvec
        self.rvec = rvec
    def get_position(self):
        return self.tvec
    def add_singlehand_grasp(self, single_grasp):
        self.single_grasp_list.append(single_grasp)
    def add_twohands_grasp(self, paired_grasp):
        self.paired_grasp_list.append(paired_grasp)
    def get_secene_obj_info(self):
        return {
            "class": self.cls,
            "size": self.size,
            "meta": self.meta,
            "bbox":self.bbox,

            "rvec":self.rvec,
            "tvec":self.tvec,

            "single_hand_grasp_list": [s_g.get_grasp_info() for s_g in self.single_grasp_list],
            "two_hands_grasp_list": [p_g.get_twohands_grasps_info() for p_g in self.paired_grasp_list]
        }
    def show_secene_obj_info(self):
        print("Obj:**************************")
        print("class:", self.cls)
        print("size:", self.size)
        print("rvec:",self.rvec)
        print("tvec:",self.tvec)

        print("single_hand_grasp_list:") 
        for s_g in self.single_grasp_list:
            s_g.show_grasp_info()
        print("two_hands_grasp_list:") 
        for p_g in self.paired_grasp_list:
            p_g.show_two_hands_grasps_info()
        print("******************************")
class SingleHandGrasp:
    def __init__(self, 
        center_point_2d,
        keypoint4_2d,                
        cls,
        meta,
        obj_ct_pred,
        width,
        
    ):
        
        self.center_point_2d = center_point_2d
        self.keypoint4_2d = keypoint4_2d
        self.cls = cls
        self.obj_ct_pred = obj_ct_pred
        self.meta = meta
        self.width = width
        
        self.calculate_pose()
    def get_obj_2d_center_point(self):
        return self.obj_ct_pred
    def calculate_pose(self):
        grasp_pose_solver = PoseSolverGrasp(self.meta["intrinsic"])
        ret, tvec, rvec, projected_points, reprojectionError, quaternion =\
            grasp_pose_solver.solve(self.keypoint4_2d)
        
        self.tvec = tvec
        self.rvec = rvec
    def get_pose(self):
        return self.rvec, self.tvec
    def get_grasp_info(self, type = "vec"):
        if type == "rotMat":
            return {
                "grasp_type": self.cls,
                "width":self.width,
                "tvec":self.tvec,
                "rvec":cv2.Rodrigues(self.rvec)[0]
            }
        elif type == "Mat":
            R, _ = cv2.Rodrigues(self.rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = self.tvec.flatten()
            return {
                "grasp_type": self.cls,
                "width":self.width,
                "T":T,
            }

        return {
            "grasp_type": self.cls,
            "width":self.width,
            "tvec":self.tvec,
            "rvec":self.rvec
        }
    def show_grasp_info(self):
        print("Grasp info------------")
        print("grasp_type: ",self.cls)
        print("width: ",self.width)
        print("tvec: ",self.tvec)
        print("rvec: ",self.rvec)
        print("----------------------")

class TwoHandsGrasp: #paired grasp
    def __init__(self, 
        center_point_2d,
        keypoint8_2d, # 因為有兩個夾爪所以是八個               
        cls,
        meta,
        obj_ct_pred,
        width0,
        width1,
        # optional
        center_point_3d = None,
        keypoint8_3d = None,
        tvec = None,
        rvec = None
    ):
        
        self.center_point_2d = center_point_2d
        self.keypoint8_2d = keypoint8_2d
        self.cls = cls
        self.obj_ct_pred = obj_ct_pred
        self.meta = meta
        self.width0 = width0
        self.width1 = width1
        if(center_point_3d != None):
            self.center_point_3d = center_point_3d,
            self.keypoint8_3d = keypoint8_3d,
            self.tvec = tvec,
            self.rvec = rvec


        self.grasp0 = SingleHandGrasp(
            center_point_2d = (keypoint8_2d[0]+keypoint8_2d[1])/2.,
            keypoint4_2d = keypoint8_2d[0:4],                
            cls = self.cls,
            meta = self.meta,
            obj_ct_pred = self.obj_ct_pred,
            width = self.width0
        )
        self.rvec0, self.tvec0 = self.grasp0.get_pose()

        self.grasp1 = SingleHandGrasp(
            center_point_2d = (keypoint8_2d[4]+keypoint8_2d[5])/2.,
            keypoint4_2d = keypoint8_2d[4:8],                
            cls = self.cls,
            meta = self.meta,
            obj_ct_pred = self.obj_ct_pred,
            width = self.width1
        )
        self.rvec1, self.tvec1 = self.grasp1.get_pose()


    def get_obj_2d_center_point(self):
        return self.obj_ct_pred
    def get_pose(self):
        
        return self.rvec0, self.tvec0, self.rvec1, self.tvec1
    def get_twohands_grasps_info(self, type = 'vec'):
        return{
            "grasp0":self.grasp0.get_grasp_info(type = type),
            "grasp1":self.grasp1.get_grasp_info(type = type)
        }
    def show_two_hands_grasps_info(self):
        print("two grasps============================")
        print("grasp0:")
        print(self.grasp0.show_grasp_info())
        print("grasp1")
        print(self.grasp1.show_grasp_info())
        print("======================================")