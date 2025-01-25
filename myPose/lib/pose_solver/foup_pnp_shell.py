# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import numpy as np
from .cubic_bbox_foup import Cubic_bbox_foup
from .pose_recover_foup import PoseSolverFoup
from scipy.spatial.transform import Rotation as R


def pnp_shell(meta, bbox, points_filtered, scale):
    #cuboid3d = Cubic_bbox_foup(1 * np.array(scale) / scale[1])
    intrinsic = meta["intrinsic"]
    pnp_solver = PoseSolverFoup(intrinsic)
    
    #print("solver: points_filtered shape: ", len(points_filtered))
    ret, location, rvec, projected_points, reprojectionError,quaternion = pnp_solver.solve(
        points_filtered, scale
        )  # N * 2
    
    if ret == 0:
        print("ret=0")
        return
    if location is not None :
        #print("quaternion:",quaternion)
        # Save to results
        bbox['location'] = location
        bbox['quaternion'] = quaternion
        bbox['projected_cuboid'] = projected_points  # Just for debug # not normalized 16
        try:
            ori = R.from_quat(quaternion).as_matrix()
        except:
            #print("quaternoion error")
            ori = R.from_rotvec(rvec.reshape(3,)).as_matrix()
        pose_pred = np.identity(4)
        pose_pred[:3, :3] = ori
        pose_pred[:3, 3] = location.reshape(3,)
        point_3d_obj = pnp_solver.cubic_bbox.get_vertices()

        point_3d_cam = pose_pred @ np.hstack(
            (np.array(point_3d_obj), np.ones((np.array(point_3d_obj).shape[0], 1)))).T
        point_3d_cam = point_3d_cam[:3, :].T  # 8 * 3
        
        #print("point_3d_cam.shape=", point_3d_cam.shape)
        # Add the centroid 我把它槓掉因為我自己把center加進去了
        #point_3d_cam = np.insert(point_3d_cam, 0, np.mean(point_3d_cam, axis=0), axis=0)

        bbox['kps_3d_cam'] = point_3d_cam  # Just for debug

        # Add the center 我把它槓掉因為我自己把center加進去了
        #projected_points = np.insert(projected_points, 0, np.mean(projected_points, axis=0), axis=0)

        # Normalization
        projected_points[:, 0] = projected_points[:, 0] / meta['width']
        projected_points[:, 1] = projected_points[:, 1] / meta['height']


        def is_visible(point):
            """Determines if a 2D point is visible."""
            return point[0] > 0 and point[0] < 1 and point[1] > 0 and point[1] < 1

        if not is_visible(projected_points[0]):
            return

        points = [(x[0], x[1]) for x in np.array(bbox['kps']).reshape(-1, 2)]

        # Add the center
        points_ori = np.insert(points, 0, np.mean(points, axis=0), axis=0)

        # Normalization
        points_ori[:, 0] = points_ori[:, 0] / meta['width']
        points_ori[:, 1] = points_ori[:, 1] / meta['height']

        # keypoint_2d_pnp, keypoint_3d, predicted_scale, keypoint_2d_ori, result_ori for debug

        #result_ori 是原本網路predict出來的九個keypoint 然後除以長寬
        #projected_points 是pnp根據原本網路predict出來的九個keypoint推算出的rvec tvec 將3D box投影到2d平面時產生的新的9個keypoint
        #
        #print("pnp_shell good result")
        return projected_points, point_3d_cam, np.array(bbox['abs_scale']), points_ori, bbox

    print("location is NONE")
    return #if location is None
