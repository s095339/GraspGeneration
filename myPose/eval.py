"""Example Evaluation script for Objectron dataset.

It reads a tfrecord, runs evaluation, and outputs a summary report with name
specified in report_file argument. When adopting this for your own model, you
have to implement the Evaluator.predict() function, which takes an image and produces 
a 3D bounding box.

Example: 
  python3 -m objectron.dataset.eval --eval_data=.../chair_test* --report_file=.../report.txt
"""

import math
import os
import time
import warnings

from absl import app
from absl import flags

import glob
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as rotation_util
import tensorflow as tf
import tqdm

import tools.objectron.dataset.iou as IoU3D
import tools.objectron.dataset.box as Box
import tools.objectron.dataset.metrics as metrics
#import tools.objectron.dataset.parser as parser

#my dastaset
from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
from lib.dataset.foup_test_dataset import foup_test_dataset

#torch
import torch
from pytorch3d.ops import box3d_overlap


FLAGS = flags.FLAGS

flags.DEFINE_string('eval_data', None, 'File pattern for evaluation data.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('max_num', -1, 'Max number of examples to evaluate.')
flags.DEFINE_string('report_file', None, 'Path of the report file to write.')

_MAX_PIXEL_ERROR = 20.
_MAX_AZIMUTH_ERROR = 30.
_MAX_POLAR_ERROR = 20.
_MAX_DISTANCE = 1.0   # In meters
_NUM_BINS = 21

def safe_divide(i1, i2):
  divisor = float(i2) if i2 > 0 else 1e-6
  return i1 / divisor

class Evaluator(object):
  """Class for evaluating the Objectron's model."""

  def __init__(self, height = 1280, width = 720):
    self.height, self.width = height, width
    #self.encoder = parser.ObjectronParser(self.height, self.width)
    self._vis_thresh = 0.1
    self._error_2d = 0.
    self._matched = 0

    self._iou_3d = 0.
    self._azimuth_error = 0.
    self._polar_error = 0.

    self._iou_thresholds = np.linspace(0.0, 1., num=_NUM_BINS)
    self._pixel_thresholds = np.linspace(0.0, _MAX_PIXEL_ERROR, num=_NUM_BINS)
    self._azimuth_thresholds = np.linspace(
        0.0, _MAX_AZIMUTH_ERROR, num=_NUM_BINS)
    self._polar_thresholds = np.linspace(0.0, _MAX_POLAR_ERROR, num=_NUM_BINS)
    self._add_thresholds = np.linspace(0.0, _MAX_DISTANCE, num=_NUM_BINS)
    self._adds_thresholds = np.linspace(0.0, _MAX_DISTANCE, num=_NUM_BINS)

    self._iou_ap = metrics.AveragePrecision(_NUM_BINS)
    self._pixel_ap = metrics.AveragePrecision(_NUM_BINS)
    self._azimuth_ap = metrics.AveragePrecision(_NUM_BINS)
    self._polar_ap = metrics.AveragePrecision(_NUM_BINS)
    self._add_ap = metrics.AveragePrecision(_NUM_BINS)
    self._adds_ap = metrics.AveragePrecision(_NUM_BINS)

  #
  #
  # TODO: implement this function for your own model.
    # #local param
    opt.depth = True
    #

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 0)
    Detector = detector_factory[opt.task]
    self.detector = Detector(opt)

    if not os.path.exists("./eval"):
      os.mkdir("./eval")
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    self.report_file_dir = f"./eval/{time_str}"
    os.mkdir(self.report_file_dir)
    self.report_file  = os.path.join(self.report_file_dir, "report.txt")
    
  #def predict(self, images, batch_size):
  def predict(self, meta, ret):
    """
      Implement your own function/model to predict the box's 2D and 3D 
      keypoint from the input images. 
      Note that the predicted 3D bounding boxes are correct upto an scale. 
      You can use the ground planes to re-scale your boxes if necessary. 

      Returns:
        A list of list of boxes for objects in images in the batch. Each box is 
        a tuple of (point_2d, point_3d) that includes the predicted 2D and 3D vertices.
    """
    image = meta['image']
    r = self.detector.run(image, meta)
    
    return r 


  def evaluate(self, meta, ret):
    """Evaluates a batch of serialized tf.Example protos."""
    images, labels, projs, planes = [], [], [], []  #暫且先不管planes好了

    """ my command
    for serialized in batch:#把一個batch裡面的每個資料拿出來
      example = tf.train.Example.FromString(serialized)
      image, label = self.encoder.parse_example(example)
      images.append(image)
      labels.append(label)
      proj, _ = self.encoder.parse_camera(example)
      projs.append(proj)
      plane = self.encoder.parse_plane(example)
      planes.append(plane)
    """

    #mycode ========================
    w,h = meta["width"], meta["height"]
    labels = [ret]
    #print("label = ", ret)
    #pred = self.model.predict(np.asarray(images), batch_size=len(batch))

    #把所有的image丟進去預測 並得到所有結果
    #results = self.predict(np.asarray(images), batch_size=len(batch))
    
    _, r = self.predict(meta, ret)

    results = [r['results']]
    #mycode ========================




    #print("label.len", len(labels), "result.len", len(results))
    # Creating some fake results for testing as well as example of what the 
    # the results should look like.
    # results = []
    # for label in labels:
    #  instances = label['2d_instance']
    #  instances_3d = label['3d_instance']
    #  boxes = []
    #  for i in range(len(instances)):
    #    point_2d = np.copy(instances[i])
    #    point_3d = np.copy(instances_3d[i])
    #    for j in range(9):
    #      # Translating the box in 3D, this will have a large impact on 3D IoU.
    #      point_3d[j] += np.array([0.01, 0.02, 0.5])
    #    boxes.append((point_2d, point_3d))
    #  results.append(boxes)


    #總而言之，results應該要長得像這樣：[(pt_2d,pt_3d),(pt_2d,pt_3d),...]
    # labels的話 就是一個batch裡每一張照片的label的list
    # [label1, label2 ....]
    # 而每一個label則是
    #   {
    #     "2d_instance" : 
    #     "3d_instance" :
    #     "visibility"  : 不知道這三小
    #   }
    #
    #for boxes, label, plane in zip(results, labels, planes):  
    #print("about to eval")
    for boxes, label in zip(results, labels):
      #print("???")
      #取出這一張圖片的預測box 跟label     
      instances = label['2d_instance']  #4個箱子的2d點
      instances_3d = label['3d_instance'] #4個箱子的3d點
      visibilities = label['visibility']  #4個箱子的可見姓

      #print(instances)
      #print(instances_3d)
      #print(visibilities)
      num_instances = 0
      for instance, instance_3d, visibility in zip(
          instances, instances_3d, visibilities):

        #print("instance_3d[0, 2]:", instance_3d[0, 2])
        #if (visibility > self._vis_thresh and
        #    self._is_visible(instance[0]) and instance_3d[0, 2] < 0):
        #print(self._is_visible(instance[0]))
        #print(visibility > self._vis_thresh)
        #print(instance_3d[0, 2] < 0)
        if (visibility > self._vis_thresh and
          self._is_visible(instance[0]) and instance_3d[0, 2] > 0): #原本是instance_3d[0, 2] < 0
          
          num_instances += 1  #總共幾個label這樣

      # We don't have negative examples in evaluation.
      if num_instances == 0:
        print("no instance")
        continue

      iou_hit_miss = metrics.HitMiss(self._iou_thresholds)
      azimuth_hit_miss = metrics.HitMiss(self._azimuth_thresholds)
      polar_hit_miss = metrics.HitMiss(self._polar_thresholds)
      pixel_hit_miss = metrics.HitMiss(self._pixel_thresholds)
      add_hit_miss = metrics.HitMiss(self._add_thresholds)
      adds_hit_miss = metrics.HitMiss(self._adds_thresholds)

      num_matched = 0
      for box in boxes: #預測結果的每一種box
        #print("here?")
        #print("box = ", box)
        box_point_2d, box_point_3d = box #每一個box的2d跟3d
        index = self.match_box(box_point_2d, instances, visibilities)
        if index >= 0:
          #print("match")
          num_matched += 1
          pixel_error = self.evaluate_2d(box_point_2d, instances[index])
          # If you only compute the 3D bounding boxes from RGB images, 
          # your 3D keypoints may be upto scale. However the ground truth
          # is at metric scale. There is a hack to re-scale your box using 
          # the ground planes (assuming your box is sitting on the ground).
          # However many models learn to predict depths and scale correctly.
          #scale = self.compute_scale(box_point_3d, plane)
          #box_point_3d = box_point_3d * scale
          azimuth_error, polar_error, iou, add, adds= self.evaluate_3d(box_point_3d, instances_3d[index])
        else:
          print("iou=0")
          pixel_error = _MAX_PIXEL_ERROR
          azimuth_error = _MAX_AZIMUTH_ERROR
          polar_error = _MAX_POLAR_ERROR
          iou = 0.
          add = _MAX_DISTANCE
          adds = _MAX_DISTANCE
  
        iou_hit_miss.record_hit_miss(iou)
        add_hit_miss.record_hit_miss(add, greater=False)
        adds_hit_miss.record_hit_miss(adds, greater=False)
        pixel_hit_miss.record_hit_miss(pixel_error, greater=False)
        azimuth_hit_miss.record_hit_miss(azimuth_error, greater=False)
        polar_hit_miss.record_hit_miss(polar_error, greater=False)
      
      self._iou_ap.append(iou_hit_miss, len(instances))
      self._pixel_ap.append(pixel_hit_miss, len(instances))
      self._azimuth_ap.append(azimuth_hit_miss, len(instances))
      self._polar_ap.append(polar_hit_miss, len(instances))
      self._add_ap.append(add_hit_miss, len(instances))
      self._adds_ap.append(adds_hit_miss, len(instances))
      self._matched += num_matched

  def evaluate_2d(self, box, instance):
    """Evaluates a pair of 2D projections of 3D boxes.

    It computes the mean normalized distances of eight vertices of a box.

    Args:
      box: A 9*2 array of a predicted box.
      instance: A 9*2 array of an annotated box.

    Returns:
      Pixel error
    """
    error = np.mean(np.linalg.norm(box[1:] - instance[1:], axis=1))
    self._error_2d += error
    return error

  def evaluate_3d(self, box_point_3d, instance):
    """Evaluates a box in 3D.

    It computes metrics of view angle and 3D IoU.

    Args:
      box: A predicted box.
      instance: A 9*3 array of an annotated box, in metric level.

    Returns:
      A tuple containing the azimuth error, polar error, 3D IoU (float), 
      average distance error, and average symmetric distance error.
    """
    azimuth_error, polar_error = self.evaluate_viewpoint(box_point_3d, instance)
    avg_distance, avg_sym_distance = self.compute_average_distance(box_point_3d,
                                                                   instance)
    #print("how many box?")
    #print(box_point_3d)
    #print(instance)
    
    
    #try:
    iou = self.evaluate_iou(box_point_3d, instance)
    print("iou:", iou)
    #except:
      
    return azimuth_error, polar_error, iou, avg_distance, avg_sym_distance 

  def compute_scale(self, box, plane):
    """Computes scale of the given box sitting on the plane."""
    center, normal = plane
    vertex_dots = [np.dot(vertex, normal) for vertex in box[1:]]
    vertex_dots = np.sort(vertex_dots)
    center_dot = np.dot(center, normal)
    scales = center_dot / vertex_dots[:4]
    return np.mean(scales)

  def compute_ray(self, box):
    """Computes a ray from camera to box centroid in box frame.

    For vertex in camera frame V^c, and object unit frame V^o, we have
      R * Vc + T = S * Vo,
    where S is a 3*3 diagonal matrix, which scales the unit box to its real size.

    In fact, the camera coordinates we get have scale ambiguity. That is, we have
      Vc' = 1/beta * Vc, and S' = 1/beta * S
    where beta is unknown. Since all box vertices should have negative Z values,
    we can assume beta is always positive.

    To update the equation,
      R * beta * Vc' + T = beta * S' * Vo.

    To simplify,
      R * Vc' + T' = S' * Vo,
    where Vc', S', and Vo are known. The problem is to compute
      T' = 1/beta * T,
    which is a point with scale ambiguity. It forms a ray from camera to the
    centroid of the box.

    By using homogeneous coordinates, we have
      M * Vc'_h = (S' * Vo)_h,
    where M = [R|T'] is a 4*4 transformation matrix.

    To solve M, we have
      M = ((S' * Vo)_h * Vc'_h^T) * (Vc'_h * Vc'_h^T)_inv.
    And T' = M[:3, 3:].

    Args:
      box: A 9*3 array of a 3D bounding box.

    Returns:
      A ray represented as [x, y, z].
    """
    if box[0, -1] > 0:
      warnings.warn('Box should have negative Z values.')

    size_x = np.linalg.norm(box[5] - box[1])
    size_y = np.linalg.norm(box[3] - box[1])
    size_z = np.linalg.norm(box[2] - box[1])
    size = np.asarray([size_x, size_y, size_z])
    box_o = Box.UNIT_BOX * size
    box_oh = np.ones((4, 9))
    box_oh[:3] = np.transpose(box_o)

    box_ch = np.ones((4, 9))
    box_ch[:3] = np.transpose(box)
    box_cht = np.transpose(box_ch)

    box_oct = np.matmul(box_oh, box_cht)
    box_cct_inv = np.linalg.inv(np.matmul(box_ch, box_cht))
    transform = np.matmul(box_oct, box_cct_inv)
    return transform[:3, 3:].reshape((3))

  def compute_average_distance(self, box, instance):
    """Computes Average Distance (ADD) metric."""
    add_distance = 0.
    for i in range(Box.NUM_KEYPOINTS):
      delta = np.linalg.norm(box[i, :] - instance[i, :])
      add_distance += delta
    add_distance /= Box.NUM_KEYPOINTS


    # Computes the symmetric version of the average distance metric.
    # From PoseCNN https://arxiv.org/abs/1711.00199
    # For each keypoint in predicttion, search for the point in ground truth
    # that minimizes the distance between the two.
    add_sym_distance = 0.
    for i in range(Box.NUM_KEYPOINTS):
      # Find nearest vertex in instance
      distance = np.linalg.norm(box[i, :] - instance[0, :])
      for j in range(Box.NUM_KEYPOINTS):
        d = np.linalg.norm(box[i, :] - instance[j, :])
        if d < distance:
          distance = d
      add_sym_distance += distance
    add_sym_distance /= Box.NUM_KEYPOINTS

    return add_distance, add_sym_distance

  def compute_viewpoint(self, box):
    """Computes viewpoint of a 3D bounding box.

    We use the definition of polar angles in spherical coordinates
    (http://mathworld.wolfram.com/PolarAngle.html), expect that the
    frame is rotated such that Y-axis is up, and Z-axis is out of screen.

    Args:
      box: A 9*3 array of a 3D bounding box.

    Returns:
      Two polar angles (azimuth and elevation) in degrees. The range is between
      -180 and 180.
    """
    x, y, z = self.compute_ray(box)
    theta = math.degrees(math.atan2(z, x))
    phi = math.degrees(math.atan2(y, math.hypot(x, z)))
    return theta, phi

  def evaluate_viewpoint(self, box, instance):
    """Evaluates a 3D box by viewpoint.

    Args:
      box: A 9*3 array of a predicted box.
      instance: A 9*3 array of an annotated box, in metric level.

    Returns:
      Two viewpoint angle errors.
    """
    predicted_azimuth, predicted_polar = self.compute_viewpoint(box)
    gt_azimuth, gt_polar = self.compute_viewpoint(instance)

    polar_error = abs(predicted_polar - gt_polar)
    # Azimuth is from (-180,180) and a spherical angle so angles -180 and 180
    # are equal. E.g. the azimuth error for -179 and 180 degrees is 1'.
    azimuth_error = abs(predicted_azimuth - gt_azimuth)
    if azimuth_error > 180:
      azimuth_error = 360 - azimuth_error

    self._azimuth_error += azimuth_error
    self._polar_error += polar_error
    return azimuth_error, polar_error

  def evaluate_rotation(self, box, instance):
    """Evaluates rotation of a 3D box.

    1. The L2 norm of rotation angles
    2. The rotation angle computed from rotation matrices
          trace(R_1^T R_2) = 1 + 2 cos(theta)
          theta = arccos((trace(R_1^T R_2) - 1) / 2)

    3. The rotation angle computed from quaternions. Similar to the above,
       except instead of computing the trace, we compute the dot product of two
       quaternion.
         theta = 2 * arccos(| p.q |)
       Note the distance between quaternions is not the same as distance between
       rotations.

    4. Rotation distance from "3D Bounding box estimation using deep learning
       and geometry""
           d(R1, R2) = || log(R_1^T R_2) ||_F / sqrt(2)

    Args:
      box: A 9*3 array of a predicted box.
      instance: A 9*3 array of an annotated box, in metric level.

    Returns:
      Magnitude of the rotation angle difference between the box and instance.
    """
    prediction = Box.Box(box)
    annotation = Box.Box(instance)
    gt_rotation_inverse = np.linalg.inv(annotation.rotation)
    rotation_error = np.matmul(prediction.rotation, gt_rotation_inverse)

    error_angles = np.array(
        rotation_util.from_dcm(rotation_error).as_euler('zxy'))
    abs_error_angles = np.absolute(error_angles)
    abs_error_angles = np.minimum(
        abs_error_angles, np.absolute(math.pi * np.ones(3) - abs_error_angles))
    error = np.linalg.norm(abs_error_angles)

    # Compute the error as the angle between the two rotation
    rotation_error_trace = abs(np.matrix.trace(rotation_error))
    angular_distance = math.acos((rotation_error_trace - 1.) / 2.)

    # angle = 2 * acos(|q1.q2|)
    box_quat = np.array(rotation_util.from_dcm(prediction.rotation).as_quat())
    gt_quat = np.array(rotation_util.from_dcm(annotation.rotation).as_quat())
    quat_distance = 2 * math.acos(np.dot(box_quat, gt_quat))

    # The rotation measure from "3D Bounding box estimation using deep learning
    # and geometry"
    rotation_error_log = scipy.linalg.logm(rotation_error)
    rotation_error_frob_norm = np.linalg.norm(rotation_error_log, ord='fro')
    rotation_distance = rotation_error_frob_norm / 1.4142

    return (error, quat_distance, angular_distance, rotation_distance)

  def evaluate_iou(self, box, instance):
    """Evaluates a 3D box by 3D IoU.

    It computes 3D IoU of predicted and annotated boxes.

    Args:
      box: A 9*3 array of a predicted box.
      instance: A 9*3 array of an annotated box, in metric level.

    Returns:
      3D Intersection over Union (float)
    """
    # Computes 3D IoU of the two boxes.
    prediction = Box.Box(box)
    annotation = Box.Box(instance)
    iou = IoU3D.IoU(prediction, annotation)
    iou_result = iou.iou()
    self._iou_3d += iou_result
    return iou_result

  def match_box(self, box, instances, visibilities):
    """Matches a detected box with annotated instances.

    For a predicted box, finds the nearest annotation in instances. This means
    we always assume a match for a prediction. If the nearest annotation is
    below the visibility threshold, the match can be skipped.

    Args:
      box: A 9*2 array of a predicted box.
      instances: A ?*9*2 array of annotated instances. Each instance is a 9*2
        array.
      visibilities: An array of the visibilities of the instances.

    Returns:
      Index of the matched instance; otherwise -1.
    """
    #mycode
    instances = np.array(instances)

    #print("yoyo:")
    #print(box.shape)
    #print(instances.shape)
    norms = np.linalg.norm(instances[:, 1:, :] - box[1:, :], axis=(1, 2))
    #print("norms:",norms)
    i_min = np.argmin(norms)
    if visibilities[i_min] < self._vis_thresh:
      return -1
    return i_min

  def write_report(self):
    """Writes a report of the evaluation."""

    def report_array(f, label, array):
      f.write(label)
      for val in array:
        f.write('{:.4f},\t'.format(val))
      f.write('\n')

    #report_file = FLAGS.report_file
    report_file = self.report_file
    with open(report_file, 'w') as f:
      f.write('Mean Error 2D: {}\n'.format(
          safe_divide(self._error_2d, self._matched)))
      f.write('Mean 3D IoU: {}\n'.format(
          safe_divide(self._iou_3d, self._matched)))
      f.write('Mean Azimuth Error: {}\n'.format(
          safe_divide(self._azimuth_error, self._matched)))
      f.write('Mean Polar Error: {}\n'.format(
          safe_divide(self._polar_error, self._matched)))

      f.write('\n')
      f.write('IoU Thresholds: ')
      for threshold in self._iou_thresholds:
        f.write('{:.4f},\t'.format(threshold))
      f.write('\n')
      report_array(f, 'AP @3D IoU    : ', self._iou_ap.aps)

      f.write('\n')
      f.write('2D Thresholds : ')
      for threshold in self._pixel_thresholds:
        f.write('{:.4f},\t'.format(threshold * 0.1))
      f.write('\n')
      report_array(f, 'AP @2D Pixel  : ', self._pixel_ap.aps)
      f.write('\n')

      f.write('Azimuth Thresh: ')
      for threshold in self._azimuth_thresholds:
        f.write('{:.4f},\t'.format(threshold * 0.1))
      f.write('\n')
      report_array(f, 'AP @Azimuth   : ', self._azimuth_ap.aps)
      f.write('\n')

      f.write('Polar Thresh  : ')
      for threshold in self._polar_thresholds:
        f.write('{:.4f},\t'.format(threshold * 0.1))
      f.write('\n')
      report_array(f, 'AP @Polar     : ', self._polar_ap.aps)
      f.write('\n')

      f.write('ADD Thresh    : ')
      for threshold in self._add_thresholds:
        f.write('{:.4f},\t'.format(threshold))
      f.write('\n')
      report_array(f, 'AP @ADD       : ', self._add_ap.aps)
      f.write('\n')

      f.write('ADDS Thresh   : ')
      for threshold in self._adds_thresholds:
        f.write('{:.4f},\t'.format(threshold))
      f.write('\n')
      report_array(f, 'AP @ADDS      : ', self._adds_ap.aps)

  def finalize(self):
    """Computes average precision curves."""
    self._iou_ap.compute_ap_curve()
    self._pixel_ap.compute_ap_curve()
    self._azimuth_ap.compute_ap_curve()
    self._polar_ap.compute_ap_curve()
    self._add_ap.compute_ap_curve()
    self._adds_ap.compute_ap_curve()

  #def _is_visible(self, point):
  #my modification
  def _is_visible(self, point):
    """Determines if a 2D point is visible."""
    "箱子的中心點"
    #print(point)
    
    return point[0] > 0 and point[0] < 1 and point[1]> 0 and point[1] < 1


#def main(argv):



class Evaluator_torch:
  def __init__(self):
    print("using torch iou")

    # TODO: implement this function for your own model.
    # #local param
    opt.depth = True
    #

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 0)
    Detector = detector_factory[opt.task]
    self.detector = Detector(opt)

    self.dataset_class_list = [0,1,2,2]
    self.class_num = 3


    if not os.path.exists("./eval_torch3d"):
      os.mkdir("./eval_torch3d")
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    self.report_file_dir = f"./eval_torch3d/{time_str}"
    os.mkdir(self.report_file_dir)
    self.report_file  = os.path.join(self.report_file_dir, "report.txt")


    self._iou_3d = [0. for i in range(self.class_num)]
    self._iou_thresholds = np.linspace(0.0, 1., num=_NUM_BINS)
    self._iou_ap = [metrics.AveragePrecision(_NUM_BINS) for i in range(self.class_num)]
    self._matched = 0

  def calculate_box_iou(self, boxes1, boxes2 ):
    
      #去掉中心點
      boxes1 = boxes1[:,1:,:]
      boxes2 = boxes2[:,1:,:]
      
      

      # reorder the box vertice number
      my_box_order =  [0,1,2,3,4,5,6,7]
      iou_box_order = [6,2,5,1,7,3,4,0]
      for i in range(boxes1.shape[0]):
        box = boxes1[i,:,:].clone() #去掉中心點
        for j in range(boxes1.shape[1]):
          ori_idx = my_box_order[j]
          iou_idx = iou_box_order[j]
          boxes1[i][iou_idx] = box[ori_idx]

      for i in range(boxes2.shape[0]):
        box = boxes2[i,:,:].clone() #去掉中心點
        for j in range(boxes2.shape[1]):
          ori_idx = my_box_order[j]
          iou_idx = iou_box_order[j]
          boxes2[i][iou_idx] = box[ori_idx]



      # Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)
      try:
        intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)
        return intersection_vol, iou_3d
      except:
        return None
      

      box3d_overlap()
  def predict(self, meta, ret):
    image = meta['image']
    r = self.detector.run(image, meta)
    return r 
  
  def evaluate(self, meta, ret):
    images, labels = [], [] 

    w,h = meta["width"], meta["height"]
    labels = [ret]
    #print("label = ", ret)
    #pred = self.model.predict(np.asarray(images), batch_size=len(batch))

    #把所有的image丟進去預測 並得到所有結果
    #results = self.predict(np.asarray(images), batch_size=len(batch))
    
    _, r = self.predict(meta, ret)

    results = [r['results']]

    for boxes, label in zip(results, labels):
      #print("???")
      #取出這一張圖片的預測box 跟label     
      instances = label['2d_instance']  #4個箱子的2d點
      instances_3d = label['3d_instance'] #4個箱子的3d點
      visibilities = label['visibility']  #4個箱子的可見姓

      boxes1 = np.zeros((len(boxes), 9, 3))
      for idx in range(len(boxes)):
        boxes1[idx,:,:] = boxes[idx][1] 
      
      boxes1 = torch.tensor(boxes1, dtype = torch.float32)
      boxes2 = torch.tensor(instances_3d, dtype = torch.float32)
        
        
        #print('test')
        #print(boxes1.shape)
        #print(boxes2.shape)

      #calculate iou
      iou_hit_miss = [metrics.HitMiss(self._iou_thresholds) for i in range(self.class_num)]
      

      ret = self.calculate_box_iou(boxes2, boxes1)
      if ret == None: return
      intersection_vol, iou_3d  = ret

      # Calculate the max value of every row
      max_values, _ = torch.max(iou_3d, dim=1)

      # Reshape to a 4x1 tensor
      max_values = max_values.view(-1, 1)

      #print(max_values)
      iou_class = [0.0 for i in range(self.class_num) ] #每一個類別的iou
      #sel]f.dataset_class_list = [0,1,2,2
      for i in range(len(max_values)):
        val = max_values[i]
        cls_id = self.dataset_class_list[i]
        iou_class[cls_id] += val
      

      for cls_id in range(self.class_num):
        self._iou_3d[cls_id] += iou_class[cls_id]
        iou_hit_miss[cls_id].record_hit_miss(iou_class[cls_id])
        self._iou_ap[cls_id].append(iou_hit_miss[cls_id], len(instances))
    self._matched += 1

  def finalize(self):
    for cls_id in range(self.class_num):
      """Computes average precision curves."""
      self._iou_ap[cls_id].compute_ap_curve()
    
  def write_report(self):
    def report_array(f, label, array):
      f.write(label)
      for val in array:
        f.write('{:.4f},\t'.format(val))
      f.write('\n')

    #report_file = FLAGS.report_file
    report_file = self.report_file
    with open(report_file, 'w') as f:
      for cls_idx in range(self.class_num):
        f.write("class number: {}\n".format(cls_idx))
        f.write('Mean 3D IoU: {}\n'.format(
            safe_divide(self._iou_3d[cls_idx], self._matched)))
      
      


      



def main(opt):
  #if len(argv) > 1:
  #  raise app.UsageError('Too many command-line arguments.')

  #evaluator = Evaluator_torch()
  opt.depth = True
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 0)
  

  evaluator = Evaluator()
  """
  ds = tf.data.TFRecordDataset(glob.glob(FLAGS.eval_data)).take(FLAGS.max_num)
  batch = []
  for serialized in tqdm.tqdm(ds):
    batch.append(serialized.numpy())
    if len(batch) == FLAGS.batch_size:
      evaluator.evaluate(batch)# 蒐集一個batch之後 丟進去evaluate
      batch.clear()
  if batch:#應該是指如果資料不滿一個batch時 就把剩下的丟進去
    evaluator.evaluate(batch)
  """
  #用自己的資料集

  data_dir_pth = "./lib/data/foup_dataset_depth"
  Dataset = foup_test_dataset(data_dir_pth, opt)
  
  for iter in range(len(Dataset)):
      batch = Dataset[iter]
      meta, ret = batch
      #image = ret['image']

      evaluator.evaluate(meta = meta, ret = ret)
      #image = torch.squeeze(image, dim = 0)
      #print("squeeze:",image.shape)
      #cv2.imshow("test", image[:,:,3])
      #cv2.waitKey(0)
      #r = detector.run(image,meta)

  

  
  evaluator.finalize()
  evaluator.write_report()
 

if __name__ == '__main__':
  #flags.mark_flag_as_required('report_file')
  #flags.mark_flag_as_required('eval_data')
  #app.run(main)
  opt = opts().init()
  print(opt)
  main(opt)
