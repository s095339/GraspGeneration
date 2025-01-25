
import cv2
import numpy as np 

grasp_rect_h = 0.1;
grasp_rect_w = 0.08;

class Grasp3d:
    # 這是因為foup的標註是基於right-hand rule，和objectron所使用的
    def __init__(self):
        
        self._vertices = [0, 0, 0] * 4
        self.center_location = [0,0,0]
        
        self._build_3dbox()
    def get_vertices(self):
        return self._vertices
    def _build_3dbox(self):
        #print("build vertice")
        cx = 0.0
        cy = 0.0
        cz = 0.0

        
        #note that it is right hand rule
        
        #x axis point to the left
        left = cx; # grasp center is on the left
        right = cx - grasp_rect_w;
        #y point upperward
        top = cy + grasp_rect_h / 2;
        bottom = cy - grasp_rect_h / 2;
     
        
        self._vertices = np.array(
            [
                # For PnP algorithm
                #cx, cy, cz, #grasp center point
                [left, top, cz], # top left
                [left, bottom, cz], # bottom left
                [right, top, cz], # top right
                [right, bottom, cz], # bottom right
            ], dtype = np.float32
        );

        #print(self.vertices)
        
