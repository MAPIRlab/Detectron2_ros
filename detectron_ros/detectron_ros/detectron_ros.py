import rclpy
import rclpy.node
import sensor_msgs.msg

import detectron2
import detectron2.config
import detectron2.data
import detectron2.model_zoo
import detectron2.utils
import detectron2.engine
from detectron2.utils.visualizer import Visualizer
from detectron_msgs.srv import SegmentImage

import numpy as np
import torch
from cv_bridge import CvBridge
import cv2

'''
Params:
interest_classes: ([int]) list of COCO class indices that we want to output. Defaults to all
visualization: (bool) whether to publish the segmented image on a topic
visualization_topic: (str) pretty self-explanatory
model_file: (str) path to the model yaml relative to detectron2/configs. You do not need to download the weights separately, detectron handles that itself
'''

class Detectron_ros (rclpy.node.Node):

    def __init__(self):
        super().__init__('Detectron_ros')
        
        # list of COCO class indices that we want to output. Defaults to all
        self.interest_classes = self.get_parameter_or("interest_classes", [*range(80)])  
        
        self.publish_visualization = self.get_parameter_or("publish_visualization", True) 
        visualization_topic = self.get_parameter_or("visualization_topic", "/detectron/segmentedImage")
        self.visualization_pub = self.create_publisher(sensor_msgs.msg.Image, visualization_topic, 1)

        self.cv_bridge = CvBridge()
        
        self.segment_image_srv =  self.create_service(SegmentImage, "/detectron/segment", self.segment_image)
        self.set_up_detectron()
        self._logger.info("Done setting up!")
        self._logger.info(f"Advertising service: {self.segment_image_srv.srv_name}")

    def set_up_detectron(self):
        MODEL_FILE=self.get_parameter_or("model_file", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self._logger.info(f"Detectron model file: {MODEL_FILE}")
        
        self.cfg = detectron2.config.get_cfg()
        self.cfg.merge_from_file(detectron2.model_zoo.get_config_file(MODEL_FILE))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(MODEL_FILE)
        self.predictor = detectron2.engine.DefaultPredictor(self.cfg)

        self._class_names = np.array( detectron2.data.MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None) )
        self._class_names = self._class_names[self.interest_classes]
        self._logger.info(f"Classes of interest: {self._class_names}")

    def segment_image(self, request, response):
        np_image = self.cv_bridge.imgmsg_to_cv2(request.image) #self.convert_to_cv_image(request.image)
        outputs = self.predictor( np_image )
        results = outputs["instances"].to("cpu")

        if results.has("pred_masks"):
            masks = np.asarray(results.pred_masks)
        else:
            return response
        
        boxes = results.pred_boxes if results.has("pred_boxes") else []
        
        response.class_ids = results.pred_classes.tolist() if results.has("pred_classes") else []
        response.class_names = self._class_names.tolist()
        response.scores =  torch.flatten( self.normalize( results.all_scores[:, self.interest_classes] )  ).tolist() if results.has("all_scores") else []
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self.cv_bridge.cv2_to_imgmsg(mask)
            response.masks.append(mask)

            box = sensor_msgs.msg.RegionOfInterest()
            box.x_offset = int(np.uint32(x1))
            box.y_offset = int(np.uint32(y1))
            box.height = int(np.uint32(y2 - y1))
            box.width = int(np.uint32(x2 - x1))
            response.boxes.append(box)
        
        if self.publish_visualization:
            visualizer = Visualizer(np_image[:, :, ::-1], detectron2.data.MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            visualizer = visualizer.draw_instance_predictions(results)
            img = visualizer.get_image()[:, :, ::-1]

            image_msg_a = self.cv_bridge.cv2_to_imgmsg(img)
            self.visualization_pub.publish(image_msg_a)

        return response

    def normalize(self, scores):
        for i in range( list(scores.shape)[0] ):
            scores[i] /= sum(scores[i])
        return scores

    def convert_to_cv_image(self, image_msg):
        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        return cv_img
    

def main(args=None):
    rclpy.init(args=args)
    node = Detectron_ros()

    rclpy.spin(node)

    node.destroy_node()


if __name__ == '__main__':
    main()
