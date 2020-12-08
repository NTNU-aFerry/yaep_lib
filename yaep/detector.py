#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge

import numpy as np

import rospy
import ros_af_msgs.msg as aferry_msg
from ros_af_msgs.msg import ImageDetection, ObjectBoundingBox, ObjectClass

import sensor_msgs.msg as sensor_msg

from .infer_image import obj_list, infer_images, load_model
from .infer_image import plot_one_box, get_index_label, color_list, set_cuda_device

def image_msg_to_cv2_image(image_msg, sensor_mode):
    """
    Converts the byte array from the image_msg to
    an image with width, height and color.

    :type image_msg: sensor_msgs.Image
    :param image_msg: ROS message containing information and data for an image.

    :type sensor_mode: str
    :param sensor_mode: The type of sensor that is active infrared vs eo

    :return: Cv2 Image

    """

    image_array_flat = np.fromstring(image_msg.data, np.uint8)
    if sensor_mode == "infrared":
        image_array_grayscale = image_array_flat.reshape(image_msg.height, image_msg.width, 1)
        image_bgr = cv2.cvtColor(image_array_grayscale, cv2.COLOR_GRAY2BGR)
        return image_bgr

    # TODO: figure out which color format that is most suited for rgb inference
    elif sensor_mode == "optical":
        image_array_grayscale = image_array_flat.reshape(image_msg.height, image_msg.width, 1)
        image_rgb = cv2.cvtColor(image_array_grayscale, cv2.COLOR_BGR2RGB)
        return image_rgb


class Detector(object):
    """
    Detector class.
    :type detection_pub: Detection publisher.
    :param detection_pub: ROS Publisher object,
    publishing on ImageDetection topic.

    :type model_path: str
    :param model_path: Path to the model weights.

    :type detection_threshold: float
    :param detection_threshold: Float value between 0 and 1
    determining whether a detection should be passed along or not.

    :type model_depth: int
    :param model_depth: The model depth coefficiant, determines
    which pre-trained weights to be used.

    :type gpu_device: int
    :param gpu_device: The gpu device number that the detector should
    run on

    :type input_topics: list
    :param input_topics: List of Image topics that should be subsrcibed to

    :type detection_topics: list
    :param detection_topics: List of detection topics that should be published to

    :type output_image_topics: list
    :param output_image_topics: List of Image topics with detections added on
    that should be published to

    :type sensor_mode: str
    :param sensor_mode: String containing information on whether this is a infrared
    or optical detector.

    """

    def __init__(self,
                 model_path,
                 detection_threshold,
                 model_depth,
                 gpu_device,
                 input_topics,
                 detection_topics,
                 output_image_topics,
                 sensor_mode,
                 input_decimation_factor=1 # Decimate the input, e.g. only use every N image
                 ):

        self.detection_threshold = detection_threshold

        self.model_depth = model_depth

        self.bridge = CvBridge()

        self.model = load_model(model_path, model_depth)

        self.decimation_factor = input_decimation_factor
        self.decimation_counter = dict()

        # Setting which gpu device to be used for running detection
        self.gpu_device = gpu_device
        set_cuda_device(gpu_device)

        # Creating dictionary containing the subscribers for the
        # different input image topics.
        input_subs = dict()
        for topic in input_topics:
            input_subs[topic] = rospy.Subscriber(
                topic,
                sensor_msg.Image,
                callback=self.callback,
                queue_size=10)

        self.input_subs = input_subs

        # Creating dictionary containing the publishers for the different
        # detections topics.
        detection_pubs = dict()
        for topic in detection_topics:
            detection_pubs[topic] = rospy.Publisher(
                topic,
                ImageDetection,
                queue_size=10)


        self.detection_pubs = detection_pubs

        # Creating dictionary containing the publishers for the
        # different output image topics
        image_pubs = dict()
        for topic in output_image_topics:
            image_pubs[topic] = rospy.Publisher(
                topic,
                sensor_msg.Image,
                queue_size=10)


        self.image_pubs = image_pubs

        self.sensor_mode = sensor_mode

    def callback(self, image_msg):
        camera_label = image_msg.header.frame_id
        if camera_label not in self.decimation_counter:
            self.decimation_counter[camera_label] = 0
        self.decimation_counter[camera_label] += 1
        if np.mod(self.decimation_counter[camera_label], self.decimation_factor) == 0:
            self.decimation_counter[camera_label] = 0
        else:
            return # Do not infer
        # Do inferring
        image = image_msg_to_cv2_image(image_msg, self.sensor_mode)
        images = list()
        images.append(image)
        preds, ori_imgs = infer_images(images, self.model, self.detection_threshold)

        # In this case the length of ori_imgs will
        # always be 1.
        bounding_boxes = list()
        for i in range(len(ori_imgs)):
            bounding_boxes.clear()

            for j in range(len(preds[i]['rois'])):
                x_min, y_min, x_max, y_max = preds[i]['rois'][j].astype(np.int)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])

                # TODO: Remove this when the new model are trained and
                # all classes should be published
                if obj == 'person' or obj == 'boat':

                    if obj == 'person':
                        obj_class = 1
                    elif obj == 'boat':
                        obj_class = 5

                    bounding_boxes.append(
                        ObjectBoundingBox(
                            coordinates=ObjectBoundingBox.PIXEL_COORDINATES,
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max,
                            detection_class=ObjectClass(
                                detection_class=obj_class
                            ),
                            score=score
                        )
                    )

                    image = plot_one_box(ori_imgs[i], [x_min, y_min, x_max, y_max],
                                         label=obj, score=score,
                                         color=color_list[get_index_label(obj, obj_list)])

            detection_msg = aferry_msg.ImageDetection(
                header=image_msg.header,
                detections=bounding_boxes
            )

            if self.sensor_mode == "infrared":
                image_detection_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
                image_detection_msg.header = image_msg.header
                if camera_label == 'ir_f':
                    self.image_pubs["/infrared/detector/f/image"].publish(image_detection_msg)
                    self.detection_pubs["/infrared/detector/f/boundingbox"].publish(detection_msg)
                elif camera_label == 'ir_fl':
                    self.image_pubs["/infrared/detector/fl/image"].publish(image_detection_msg)
                    self.detection_pubs["/infrared/detector/fl/boundingbox"].publish(detection_msg)
                elif camera_label == 'ir_fr':
                    self.image_pubs["/infrared/detector/fr/image"].publish(image_detection_msg)
                    self.detection_pubs["/infrared/detector/fr/boundingbox"].publish(detection_msg)
                elif camera_label == 'ir_rl':
                    self.image_pubs["/infrared/detector/rl/image"].publish(image_detection_msg)
                    self.detection_pubs["/infrared/detector/rl/boundingbox"].publish(detection_msg)
                elif camera_label == 'ir_rr':
                    self.image_pubs["/infrared/detector/rr/image"].publish(image_detection_msg)
                    self.detection_pubs["/infrared/detector/rr/boundingbox"].publish(detection_msg)

            elif self.sensor_mode == "optical":
                image_detection_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")

                image_detection_msg.header = image_msg.header
                if camera_label == 'EO_F':
                    self.image_pubs["/optical/detector/f/image"].publish(image_detection_msg)
                    self.detection_pubs["/optical/detector/f/boundingbox"].publish(detection_msg)
                elif camera_label == 'EO_FL':
                    self.image_pubs["/optical/detector/fl/image"].publish(image_detection_msg)
                    self.detection_pubs["/optical/detector/fl/boundingbox"].publish(detection_msg)
                elif camera_label == 'EO_FR':
                    self.image_pubs["/optical/detector/fr/image"].publish(image_detection_msg)
                    self.detection_pubs["/optical/detector/fr/boundingbox"].publish(detection_msg)
                elif camera_label == 'EO_RL':
                    self.image_pubs["/optical/detector/rl/image"].publish(image_detection_msg)
                    self.detection_pubs["/optical/detector/rl/boundingbox"].publish(detection_msg)
                elif camera_label == 'EO_RR':
                    self.image_pubs["/optical/detector/rr/image"].publish(image_detection_msg)
                    self.detection_pubs["/optical/detector/rr/boundingbox"].publish(detection_msg)

