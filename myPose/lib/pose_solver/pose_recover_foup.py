import cv2
import numpy as np

from .cubic_bbox_foup import Cubic_bbox_foup
from pyrr import Quaternion
from scipy.spatial.transform import Rotation as R
import sklearn

class PoseSolverFoup:

    def __init__(self, camera_intrinsic):
        
        self.camera_intrinsic = camera_intrinsic
        self._dist_coeffs = np.zeros((4, 1))
        self.cubic_bbox = None
    def set_camera_instrinsic(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
    def solve(self,  bbox_point_2d, relative_scale, use_ct = True):
        #================================================#
        #1. 決定 3D bounding box的 3d座標 (以中心點為原點) #
        #================================================#

        cubic_bbox = Cubic_bbox_foup(relative_scale = relative_scale)
        self.cubic_bbox = cubic_bbox

        #================================================#
        #2. 決定 PNP 的演算法                             #
        #================================================#

        pnp = cv2.SOLVEPNP_ITERATIVE

        #================================================#
        #3. 蒐集可用的點                                  #
        #================================================#

        bbox_point_2d = np.array(bbox_point_2d).reshape(-1, 2).tolist()
        #如果bbox_point_2d的形狀本來就是(-1,2) 這一段code則不會造成啥變化
        
        #print(bbox_point_2d)
        #borrow from centerpose
        location = None
        quaternion = None

        cuboid3d_points = np.array(cubic_bbox.get_vertices()) #包含中心點
        if(not use_ct):cuboid3d_points = cuboid3d_points[1:]
        #print("box3d = ",cuboid3d_points)
        bbox_2d_points = []
        bbox_3d_points = []

        for i in range(len(bbox_point_2d)):
            pt_2d = bbox_point_2d[i]
            if (pt_2d is None or pt_2d[0] < -5000 or pt_2d[1] < -5000):
                continue

            bbox_2d_points.append(pt_2d)
            bbox_3d_points.append(cuboid3d_points[i])

        bbox_2d_points = np.array(bbox_2d_points, dtype=float)
        bbox_3d_points = np.array(bbox_3d_points, dtype=float)

        if len(bbox_2d_points) > 3:
            #需要至少四個點才能做pnp來還原姿態
            #print(self.camera_intrinsic)
            ret, rvec, tvec, reprojectionError = cv2.solvePnPGeneric(
                bbox_3d_points,
                bbox_2d_points,
                self.camera_intrinsic,
                self._dist_coeffs,
                flags=pnp
            )

            #mycode =======================================================================#
            #print("rvec:", rvec)
            #print("tvec:", tvec)
            #print("ret", ret)

            #convert rotation vector to rotation matrix
        
            #=======================================================================#    
            
            #borrow from CenterPose
            if ret:
                rvec = np.array(rvec[0])
                tvec = np.array(tvec[0])
                reprojectionError = reprojectionError.flatten()[0]

                location = list(x[0] for x in tvec)
                quaternion = self.convert_rvec_to_quaternion(rvec)

                # Still use OpenCV way to project 3D points
                projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, self.camera_intrinsic,
                                                        self._dist_coeffs)
                projected_points = np.squeeze(projected_points)

                #print("rvec:", rvec)
                #print("tvec:", tvec)
                # Todo: currently, we assume pnp fails if z<0
                x, y, z = location
                if z < 0:
                    # # Get the opposite location
                    # location = [-x, -y, -z]
                    #
                    # # Change the rotation by 180 degree
                    # rotate_angle = np.pi
                    # rotate_quaternion = Quaternion.from_axis_rotation(location, rotate_angle)
                    # quaternion = rotate_quaternion.cross(quaternion)
                    location = None
                    quaternion = None
                    location_new = None
                    quaternion_new = None
                    
                    #print("PNP solution is behind the camera (Z < 0) => Fail")
                else:
                    pass
                    #print("solvePNP found good results (opencv)- location: {} - rotation: {} !!!".format(location, quaternion))

            #print("tvec.shape = ", tvec.shape, rvec.shape)
            return ret, tvec, rvec, projected_points, reprojectionError, quaternion
        else:
            print("2d point < 4")
                
    def convert_rvec_to_quaternion(self, rvec):
        '''Convert rvec (which is log quaternion) to quaternion'''
        theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
        raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

        # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
        return Quaternion.from_axis_rotation(raxis, theta)
    #mycode
    