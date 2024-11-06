"""Implements an operator that detects obstacles."""
import time

import erdos

import numpy as np

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    OBSTACLE_LABELS, load_coco_bbox_colors, load_coco_labels

import tensorflow as tf



class DetectionHandler:
    """Detects obstacles using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages.
        model_path(:obj:`str`): Path to the model pb file.
        flags (absl.flags): Object to be used to access absl flags.
    """

    def __init__(self, model_path: str,path_coco_labels, flags):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(
            [physical_devices[flags.obstacle_detection_gpu_index]],
            'GPU')
        tf.config.experimental.set_memory_growth(
            physical_devices[flags.obstacle_detection_gpu_index], True)
        tf.config.threading.set_intra_op_parallelism_threads(8)


        # pylot.utils.set_tf_loglevel(logging.ERROR)
        # Only sets memory growth for flagged GPU
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_visible_devices(
        #     [physical_devices[self._flags.obstacle_detection_gpu_index]],
        #     'GPU')

       
        # tf.config.experimental.set_memory_growth(
        #     physical_devices[self._flags.obstacle_detection_gpu_index], True)
        # tf.config.threading.set_intra_op_parallelism_threads(8)
        # Load the model from the saved_model format file.

        self._model = tf.saved_model.load(model_path)

        # logical_devices = tf.config.list_logical_devices()
        # for device in logical_devices:
        #     self._logger.info(f'Logical device {device.name} is on physical device {device.device_type}')


        self._coco_labels = load_coco_labels(path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)
        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0
        self.__run_model(np.zeros((108, 192, 3), dtype='uint8'))


    # def on_msg_camera_stream(self, msg: erdos.Message,
    #                          obstacles_stream: erdos.WriteStream):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.FrameMessage`): Message
                received.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.ObstaclesMessage`
                messages.
        """
    def detect(self, frame):
        num_detections, res_boxes, res_scores, res_classes = self.__run_model(frame)
        obstacles = []
        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                if (res_scores[i] >= 0.3):
                    if (self._coco_labels[res_classes[i]] in OBSTACLE_LABELS):
                        obstacles.append(
                            Obstacle(BoundingBox2D(
                                int(res_boxes[i][1] *
                                    1920),
                                int(res_boxes[i][3] *
                                    1920),
                                int(res_boxes[i][0] *
                                    1080),
                                int(res_boxes[i][2] *
                                    1080)),
                                res_scores[i],
                                self._coco_labels[res_classes[i]],
                                id=self._unique_id))
                        self._unique_id += 1
        return obstacles

    def __run_model(self, image_np):
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        infer = self._model.signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        # for k,v in result.items():
        #     print(v.device)


        boxes = result['boxes'].numpy()
        scores = result['scores'].numpy()
        classes = result['classes'].numpy()
        num_detections = result['detections'].numpy()

        num_detections = int(num_detections[0])
        res_classes = [int(cls) for cls in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        return num_detections, res_boxes, res_scores, res_classes
