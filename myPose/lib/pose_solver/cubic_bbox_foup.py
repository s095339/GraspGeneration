
import cv2
import numpy as np 

class Cubic_bbox_foup:
    # 這是因為foup的標註是基於right-hand rule，和objectron所使用的
    # left-hand rule不一樣，所以需要另外寫一個
    Center = 8
    FrontTopRight = 0
    FrontTopLeft = 1
    FrontBottomLeft = 2
    FrontBottomRight = 3
    RearTopRight = 4
    RearTopLeft = 5
    RearBottomLeft = 6
    RearBottomRight = 7
    
    def __init__(self, relative_scale = [1.0,1.0,1.0], coord_sys = None):
        
        self.ct_location = [0, 0, 0]
        self.coord_sys = coord_sys
        self.relative_scale = relative_scale
        self._vertices = [0, 0, 0] * 8
        self.center_location = [0,0,0]
        self._build_3dbox()

    def get_vertices(self):
        return self._vertices
    def _build_3dbox(self):

        #reference from CenterPose
        width, height, depth = self.relative_scale
        cx, cy, cz = self.center_location
        # X axis point to the left
        left = cx + width / 2.0
        right = cx - width / 2.0
        # Y axis point upward
        top = cy + height / 2.0
        bottom = cy - height / 2.0
        # Z axis point forward
        front = cz + depth / 2.0
        rear = cz - depth / 2.0
        
        self._vertices = [
                self.center_location,   # Center
                [left, bottom, rear],  # Rear Bottom Left
                [left, bottom, front],  # Front Bottom Left
                [left, top, rear],  # Rear Top Left
                [left, top, front],  # Front Top Left

                [right, bottom, rear],  # Rear Bottom Right
                [right, bottom, front],  # Front Bottom Right
                [right, top, rear],  # Rear Top Right
                [right, top, front],  # Front Top Right
            ]
        

class Axis:
    
    def __init__(self):
        self._vertices = [
               [0,0,0],
               [0.05,0,0],
               [0,0.05,0],
               [0,0,0.05]
            ]
        
        pass

    def get_vertices(self):
        return self._vertices
    