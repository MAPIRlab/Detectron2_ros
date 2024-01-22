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
from segmentation_msgs.srv import SegmentImage
from segmentation_msgs.msg import SemanticInstance2D
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose

import numpy as np
from cv_bridge import CvBridge

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
        
        interest_class_names = self._class_names[self.interest_classes]
        self._logger.info(f"Classes of interest: {interest_class_names}")

    def segment_image(self, request, response):
        numpy_image = self.cv_bridge.imgmsg_to_cv2(request.image)
        outputs = self.predictor( numpy_image )
        results = outputs["instances"].to("cpu")

        if results.has("pred_masks"):
            masks = np.asarray(results.pred_masks)
        else:
            return response
        
        boxes = results.pred_boxes if results.has("pred_boxes") else []
        scores = results.scores
        
        #This field requires a modified version of detectron2. The official version does not output the scores of "losing" classes 
        scores_all_classes = self.normalize( results.all_scores[:, self.interest_classes] ) if results.has("all_scores") else None

        
        for i, bbox in enumerate(boxes):

            semantic_instance = SemanticInstance2D()

            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]] = 255

            semantic_instance.mask = self.cv_bridge.cv2_to_imgmsg(mask)

            if scores_all_classes is not None:
                semantic_instance.detection = self.set_multiclass_detection(scores_all_classes[i,:], bbox)
            else:
                semantic_instance.detection = self.set_singleclass_detection(self._class_names[results.pred_classes[i]], float(scores[i]), bbox)

            response.instances.append(semantic_instance)

        if self.publish_visualization:
            visualizer = Visualizer(numpy_image[:, :, ::-1], detectron2.data.MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            visualizer = visualizer.draw_instance_predictions(results)
            img = visualizer.get_image()[:, :, ::-1]

            image_msg_a = self.cv_bridge.cv2_to_imgmsg(img)
            self.visualization_pub.publish(image_msg_a)

        return response

    def normalize(self, scores):
        for i in range( list(scores.shape)[0] ):
            scores[i] /= sum(scores[i])
        return scores

    def set_singleclass_detection(self, class_name, score, bbox):

        detection = Detection2D()
        detection.bbox = BoundingBox2D()
        detection.bbox.center.position.x = float(bbox[0]) 
        detection.bbox.center.position.y = float(bbox[1])
        detection.bbox.size_x = float(bbox[2] - bbox[0])
        detection.bbox.size_y = float(bbox[3] - bbox[1])
        detection.results = []
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = class_name
        hypothesis.hypothesis.score = score

        detection.results.append(hypothesis)

        return detection

    def set_multiclass_detection(self, scores, bbox):

        detection = Detection2D()
        detection.bbox = BoundingBox2D()
        detection.bbox.center.position.x = float(bbox[0]) 
        detection.bbox.center.position.y = float(bbox[1])
        detection.bbox.size_x = float(bbox[2] - bbox[0])
        detection.bbox.size_y = float(bbox[3] - bbox[1])
        detection.results = []

        for i in range(scores):

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = self._class_names[self.interest_classes[i]]
            hypothesis.hypothesis.score = scores[i]

            detection.results.append(hypothesis)

        return detection

    def make_classification(self, class_name, score):
        c = Classification()
        c.class_name = class_name
        c.score = score
        return c
    
    def make_all_classifications(self, scores):
        classifications = []
        for i in range(scores):
            class_id = self.interest_classes[i]
            class_name = self._class_names[class_id]
            c = Classification()
            c.class_name = class_name
            c.score = scores[i]
            classifications.append(c)
        return classifications

def main(args=None):
    rclpy.init(args=args)
    node = Detectron_ros()

    rclpy.spin(node)

    node.destroy_node()


if __name__ == '__main__':
    main()
