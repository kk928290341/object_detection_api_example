# -*- coding: utf-8 -*-

import os
import time

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if tf.__version__ != '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

# 模型配置
PATH_TO_CKPT = 'training/export_result/frozen_inference_graph.pb'
PATH_TO_LABELS = 'data/classes_map.pbtxt'
NUM_CLASSES = 10

# 加载模型
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')

# 加载类别
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 读取图片
def read_image(path):
    image_np = cv2.imread(path)
    # image_np = cv2.resize(image_np, (780, 540), interpolation=cv2.INTER_CUBIC)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    WIDTH = image_np.shape[1]
    HEIGHT = image_np.shape[0]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    return image_np, image_np_expanded, WIDTH, HEIGHT

img_path = 'data/images/G5_47.jpg'

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        image_np, image_np_expanded, WIDTH, HEIGHT = read_image(img_path)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes = np.reshape(boxes, (-1, boxes.shape[-1]))
        scores = np.reshape(scores, (-1))
        classes = np.reshape(classes, (-1)).astype(np.int32)

        vis_util.visualize_boxes_and_labels_on_image_array(image_np, boxes, classes, scores, category_index,
                                                           use_normalized_coordinates=True, line_thickness=8)
        # cv2.imwrite('detection.png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        img_detection = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("detection", img_detection)
        cv2.waitKey(0)