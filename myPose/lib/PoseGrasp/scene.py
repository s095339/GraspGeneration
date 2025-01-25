import numpy as np
import cv2
from .Obj_Grasp import SceneObj,SingleHandGrasp,TwoHandsGrasp

from pose_solver.cubic_bbox_foup import Cubic_bbox_foup
from pose_solver.grasp_frame import Grasp3d as Grasp_Frame
from utils.transform import create_homog_matrix, create_rot_mat_axisAlign
color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)

color_list = color_list.reshape((-1, 3)) * 255


def rotate_poses_180_by_x(rvec, tvec):
        # Convert rvec to a rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()

        T_new = np.copy(T)
        # correct the pose correspondingly. Rotate along the x axis by 180 degrees
        M_rot = create_homog_matrix(
            R_mat=create_rot_mat_axisAlign([1, -2, -3]),
            T_vec=np.zeros((3, )) 
        )
        T_new = T_new @ M_rot 
        
        R_new = T_new[:3, :3]

        rvec_new, _ = cv2.Rodrigues(R_new)
        tvec_new = T_new[:3, 3].reshape(3, 1)
        # The translation vector remains the same since the rotation is about the origin
        


        return rvec_new, tvec_new

class Debugger:
    def __init__(self, img):
        self.img = img.copy()
        self.img_ori = img
        self.theme = ""
        colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    def add_axis(self,axis_proj,cat, conf = 1):
        #print("axis_pts", pts)
        cv2.arrowedLine(self.img, axis_proj[0], axis_proj[1], [0,0,255],2)
        cv2.arrowedLine(self.img, axis_proj[0], axis_proj[2], [0,255,0],2)
        cv2.arrowedLine(self.img, axis_proj[0], axis_proj[3], [255,0,0],2)
    def imshow(self, windowsname = "debugger"):
        cv2.imshow(windowsname, self.img)
        cv2.waitKey(0)
        return self.img
    def add_3d_bbox(self, pts, cat, conf=1 ):
        #print("3d bbox shape = ", pts.shape)
        pts = np.array(pts, dtype = np.int32)
        cat = int(cat)
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()

        ct =  pts[:2]
        
        key_point = np.array(pts).reshape((9,2))
        #color = np.array(color_list[cat+1], dtype = int).tolist()

        color = np.array([255,0,255], dtype = int).tolist()
        #print("color:",np.array(color, dtype = int).tolist())
        #print("key", key_point)
        for i in range(key_point.shape[0]):
            pt = key_point[i]
        cv2.circle(self.img, pt, 2, color, 1)
        #if(i>0):cv2.putText(self.img, str(i-1), pt, 1 , 1 , [0, 0 , 0], 1)
        #key_point[i] = pt
        #cv2.circle(self.img, pt, 2, c, 2)
        #key_point[0,:] = ct
        cv2.circle(self.img, key_point[0], 2, color, 2)
        

        cv2.line(self.img, key_point[1], key_point[2], color,2)
        cv2.line(self.img, key_point[1], key_point[5], color,2)
        cv2.line(self.img, key_point[6], key_point[2], color,2)
        cv2.line(self.img, key_point[5], key_point[6], color,2)

        cv2.line(self.img, key_point[3], key_point[7], color,2)
        cv2.line(self.img, key_point[4], key_point[8], color,2)
        cv2.line(self.img, key_point[3], key_point[4], color,2)
        cv2.line(self.img, key_point[7], key_point[8], color,2)

        cv2.line(self.img, key_point[1], key_point[3], color,2)
        cv2.line(self.img, key_point[4], key_point[2], color,2)
        cv2.line(self.img, key_point[8], key_point[6], color,2)
        cv2.line(self.img, key_point[7], key_point[5], color,2)
        return
    def add_single_grasp(self, pts, cat, conf=1 , show_obj_ct = False, obj_ct = None, rotate = False):
        pts = np.array(pts, dtype = np.int32)
        cat = int(cat)
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()

        ct =  pts[:2]
        
        key_point = np.array(pts).reshape((4,2)).astype(int)
        #print("keypoint:",key_point)
        #color = np.array(color_list[cat+1], dtype = int).tolist()
        color = np.array([0,255,255], dtype = int).tolist()
        #print("color:",np.array(color, dtype = int).tolist())
        #print("key", key_point)
        
        #key_point[i] = pt
        #cv2.circle(self.img, pt, 2, c, 2)
        #key_point[0,:] = ct
        top_center = ((key_point[0]+key_point[2])/2).astype(int)
        button_center = ((key_point[1]+key_point[3])/2).astype(int)
        rect_center = ((key_point[1]+key_point[3]+key_point[0]+key_point[2])/4).astype(int)
        right_center = ((key_point[2]+key_point[3])/2).astype(int)
        
        for i in range(4):
            cv2.circle(self.img, key_point[i].astype(int), 1, c, 1)
            if(rotate):
                cv2.putText(self.img, str(i+1), key_point[i].astype(int),1 ,1 ,[0,0,225])
            else:
                cv2.putText(self.img, str(i+1), key_point[i].astype(int),1 ,1 ,c)

        cv2.line(self.img, key_point[0], top_center, color,2)
        cv2.line(self.img, key_point[1], button_center, color,2)
        cv2.line(self.img, top_center, button_center, color,2)
        cv2.line(self.img, rect_center, right_center, color,2)
        
        if show_obj_ct and obj_ct.any():
        
            obj_ct = np.array(obj_ct, dtype = np.int32).reshape(2).astype(int)
            
            cv2.arrowedLine(self.img, key_point[0], obj_ct, color, 1)

  
        return

    def add_paired_grasp(self, pts, cat, conf=1, show_obj_ct = False, obj_ct = None):
        pts = np.array(pts, dtype = np.int32)
        cat = int(cat)
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()

        ct =  pts[:2]
        
        key_point = np.array(pts).reshape((9,2)).astype(int)
        #print("keypoint:",key_point)
        color = np.array(color_list[cat+1], dtype = int).tolist()
        color = np.array([255,255,0], dtype = int).tolist()
        #print("color:",np.array(color, dtype = int).tolist())
        #print("key", key_point)
        
        #key_point[i] = pt
        #cv2.circle(self.img, pt, 2, c, 2)
        #key_point[0,:] = ct
        print(key_point)
        top_center = ((key_point[1]+key_point[3])/2).astype(int)
        button_center = ((key_point[2]+key_point[4])/2).astype(int)
        rect_center = ((key_point[2]+key_point[4]+key_point[1]+key_point[3])/4).astype(int)
        right_center = ((key_point[3]+key_point[4])/2).astype(int)
        
        cv2.line(self.img, key_point[1], top_center, color,2)
        cv2.line(self.img, key_point[2], button_center, color,2)
        cv2.line(self.img, top_center, button_center, color,2)
        cv2.line(self.img, rect_center, right_center, color,2)

        top_center = ((key_point[5]+key_point[7])/2).astype(int)
        button_center = ((key_point[6]+key_point[8])/2).astype(int)
        rect_center = ((key_point[6]+key_point[8]+key_point[5]+key_point[7])/4).astype(int)
        right_center = ((key_point[7]+key_point[8])/2).astype(int)
        
        #cv2.line(self.img, key_point[5], top_center, color,2)
        #cv2.line(self.img, key_point[6], button_center, color,2)
        #v2.line(self.img, top_center, button_center, color,2)
        #cv2.line(self.img, rect_center, right_center, color,2)
        cv2.line(self.img, key_point[5], key_point[6], color,2)
        cv2.line(self.img, key_point[6], key_point[8], color,2)
        cv2.line(self.img, key_point[8], key_point[7], color,2)
        cv2.line(self.img, key_point[7], key_point[5], color,2)

        grasp0_center = ((key_point[1]+key_point[2])/2).astype(int)
        grasp1_center = ((key_point[5]+key_point[6])/2).astype(int)
        if show_obj_ct and obj_ct.any():
        
            obj_ct = np.array(obj_ct, dtype = np.int32).reshape(2).astype(int)
        
        cv2.arrowedLine(self.img, key_point[0], obj_ct, color, 1)
        cv2.arrowedLine(self.img, key_point[0], grasp0_center, color, 1)
        cv2.arrowedLine(self.img, key_point[0], grasp1_center, color, 1)
        return

class ImageScene: #每一張圖片的情景
    def __init__(self, meta):
        self.meta = meta
        #print(meta)

        self.scene_obj_list = [] #每一張圖片出現的物件 晶圓盒
        self.scene_obj_centerpoint_list = []

        self.scene_singlehand_grasp_list = []

        self.scene_twohand_grasp_list = []
        self._dist_coeffs = np.zeros((4, 1))

        self.threshold = 0.0

    def get_scene_obj_list(self):
        return self.scene_obj_list
    def get_obj_position_list(self, type = "3d"):
        if type == "3d": return [obj.get_position() for obj in self.scene_obj_list ]
        else : return self.scene_obj_centerpoint_list
    def _project_2d_box(self, size, rvec, tvec):
        #print("tvec:", tvec)
        #print("rvec:", rvec)
        cubic_bbox = Cubic_bbox_foup(relative_scale = size)
        cuboid3d_points = np.array(cubic_bbox.get_vertices()) #包含中心點
        projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, self.meta['intrinsic'],
                                                        self._dist_coeffs)
        projected_points = np.squeeze(projected_points)
        return projected_points
    def _project_2d_grasp(self, rvec, tvec):
        #print("tvec:", tvec)
        #print("rvec:", rvec)
        grasp3d = Grasp_Frame()
        grasp3d_points = np.array(grasp3d.get_vertices()) #! 不包含中心點
        projected_points, _ = cv2.projectPoints(grasp3d_points, rvec, tvec, self.meta['intrinsic'],
                                                        self._dist_coeffs)
        projected_points = np.squeeze(projected_points)
        return projected_points
    
    def say_hello(self):
        pass
        #print("hello!!!")
    def add_obj(self, scene_obj: SceneObj):
        #print("add_scence")
        self.scene_obj_list.append(scene_obj)
        self.scene_obj_centerpoint_list.append(scene_obj.get_2d_center_point())

    def add_scene_singlehand_grasp(self, single_hand_grasp:SingleHandGrasp):
        #print("add single grasp")
        self.scene_singlehand_grasp_list.append(single_hand_grasp)
     

    def add_scene_twohands_grasp(self, two_hands_grasp:TwoHandsGrasp):
        #print("add two grasp")
        self.scene_twohand_grasp_list.append(two_hands_grasp)
        
    def add_obj(self, scene_obj: SceneObj):
        #print("add_scence")
        self.scene_obj_list.append(scene_obj)
        self.scene_obj_centerpoint_list.append(scene_obj.get_2d_center_point())

    def scene_obj_grasp_match(self):

        #match single hand grasp candidate
        for s_g_idx in range(len(self.scene_singlehand_grasp_list)):
            s_g = self.scene_singlehand_grasp_list[s_g_idx]
            
            min = 10000
            argmin = -1
            for obj_id in range(len(self.scene_obj_centerpoint_list)):
                ct = self.scene_obj_centerpoint_list[obj_id]
                dist = np.linalg.norm(s_g.get_obj_2d_center_point()-ct)
                if(dist<min): 
                    min = dist
                    argmin = obj_id   
            if argmin>-1:
                self.scene_obj_list[argmin].add_singlehand_grasp(s_g)
            
        #match two hands grasp candidate
        for t_g_idx in range(len(self.scene_twohand_grasp_list)):
            t_g = self.scene_twohand_grasp_list[t_g_idx]
            
            min = 10000
            argmin = -1
            for obj_id in range(len(self.scene_obj_centerpoint_list)):
                ct = self.scene_obj_centerpoint_list[obj_id]
                dist = np.linalg.norm(t_g.get_obj_2d_center_point()-ct)
                if(dist<min): 
                    min = dist
                    argmin = obj_id   
            #match
            if argmin>-1:
                self.scene_obj_list[argmin].add_twohands_grasp(t_g)
        
         
    def get_scene_info(self):
        return {
            "meta", self.meta,
            "scene_objs_list", [scence_obj.get_secene_obj_info() for scence_obj in self.scene_obj_list]
        }
    def scene_info_show(self):
        print("scene info:")
        for s_obj in self.scene_obj_list:
            print("#############################")
            s_obj.show_secene_obj_info()
            print("#############################")
    
    def scene_imshow(self):
        #print("scence show")
        img = self.meta['color_img']
        debugger = Debugger(img = img)
        box_id = 0
        for obj in self.scene_obj_list:  
            info = obj.get_secene_obj_info()
            #print("info = ", info)
            box_2d_pt = self._project_2d_box(
                info['size'],info['rvec'],info['tvec']
            )
            #print("box_2d_pt:", box_2d_pt)
            #print("info['size']:",info['size'])
            
            debugger.add_3d_bbox(pts = box_2d_pt, cat = box_id)

    
            for s_g_info in obj.get_secene_obj_info()['single_hand_grasp_list']:
                #print("sginfo = ", s_g_info)
                grasp_2d_pt = self._project_2d_grasp(
                    s_g_info['rvec'], s_g_info['tvec']
                )
                debugger.add_single_grasp(pts = grasp_2d_pt, cat = box_id)
            
            for p_g_info in obj.get_secene_obj_info()['two_hands_grasp_list']:
                #print("sginfo = ", s_g_info)

                grasp_2d_pt = self._project_2d_grasp(
                    p_g_info['grasp0']['rvec'], p_g_info['grasp0']['tvec']
                )
                debugger.add_single_grasp(pts = grasp_2d_pt, cat = box_id)

                grasp_2d_pt = self._project_2d_grasp(
                    p_g_info['grasp1']['rvec'], p_g_info['grasp1']['tvec']
                )
                debugger.add_single_grasp(pts = grasp_2d_pt, cat = box_id)
            
            box_id+=1
        

        return debugger.imshow()



class LabelScene: #每一張Label的情景
    def __init__(self, meta):
        self.meta = meta
        #print(meta)

        self.scene_obj_list = [] #每一張圖片出現的物件 晶圓盒
        self.scene_obj_centerpoint_list = []
        self._dist_coeffs = np.zeros((4, 1))
    def get_scene_obj_list(self):
        return self.scene_obj_list
    def add_obj(self, scene_obj: SceneObj):
        #print("add_scence")
        self.scene_obj_list.append(scene_obj)
        self.scene_obj_centerpoint_list.append(scene_obj.get_2d_center_point())
    def get_obj_position_list(self, type = "3d"):

        if type == "3d" : return [obj.get_position() for obj in self.scene_obj_list ]
        else: return self.scene_obj_centerpoint_list 
    def get_scene_info(self):
        return {
            "meta", self.meta,
            "scene_objs_list", [scence_obj.get_secene_obj_info() for scence_obj in self.scene_obj_list]
        }
    def scene_info_show(self):
        print("scene info:")
        for s_obj in self.scene_obj_list:
            print("#############################")
            s_obj.show_secene_obj_info()
            print("#############################")
    def _project_2d_box(self, size, rvec, tvec):
        #print("tvec:", tvec)
        #print("rvec:", rvec)
        cubic_bbox = Cubic_bbox_foup(relative_scale = size)
        cuboid3d_points = np.array(cubic_bbox.get_vertices()) #包含中心點
        projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, self.meta['intrinsic'],
                                                        self._dist_coeffs)
        projected_points = np.squeeze(projected_points)
        return projected_points

    def _project_2d_grasp(self, rvec, tvec, rotate = False):
        #print("tvec:", tvec)
        #print("rvec:", rvec)
        grasp3d = Grasp_Frame()
        grasp3d_points = np.array(grasp3d.get_vertices()) #! 不包含中心點

        if(rotate):
            rvec,tvec = rotate_poses_180_by_x(rvec,tvec)
        
        projected_points, _ = cv2.projectPoints(grasp3d_points, rvec, tvec, self.meta['intrinsic'],
                                                        self._dist_coeffs)
        projected_points = np.squeeze(projected_points)
        return projected_points
    def scene_imshow(self, rotate = False):
        #print("scence show")
        img = self.meta['color_img']
        debugger = Debugger(img = img)
        box_id = 0
        for obj in self.scene_obj_list:  
            info = obj.get_secene_obj_info()
            #print("info = ", info)
            box_2d_pt = self._project_2d_box(
                info['size'],info['rvec'],info['tvec']
            )
            #print("box_2d_pt:", box_2d_pt)
            #print("info['size']:",info['size'])
            debugger.add_3d_bbox(pts = box_2d_pt, cat = box_id)
            for s_g_info in obj.get_secene_obj_info()['single_hand_grasp_list']:
                #print("sginfo = ", s_g_info)
                grasp_2d_pt = self._project_2d_grasp(
                    s_g_info['rvec'], s_g_info['tvec'], rotate=rotate
                )
                debugger.add_single_grasp(pts = grasp_2d_pt, cat = box_id)

            for p_g_info in obj.get_secene_obj_info()['two_hands_grasp_list']:
                #print("sginfo = ", s_g_info)

                grasp_2d_pt = self._project_2d_grasp(
                    p_g_info['grasp0']['rvec'], p_g_info['grasp0']['tvec'], rotate=rotate
                )
                debugger.add_single_grasp(pts = grasp_2d_pt, cat = box_id)

                grasp_2d_pt = self._project_2d_grasp(
                    p_g_info['grasp1']['rvec'], p_g_info['grasp1']['tvec']
                )
                debugger.add_single_grasp(pts = grasp_2d_pt, cat = box_id)

            box_id+=1
        

        debugger.imshow(windowsname = "label")