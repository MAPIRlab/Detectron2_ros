import detectron2
import detectron2.config
import detectron2.model_zoo
import detectron2.engine
import cv2


## Minimal example, just to check that detectron is propely installed in your system and added to the PYTHONPATH

def __main__():

	MODEL_FILE="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

	cfg = detectron2.config.get_cfg()
	cfg.merge_from_file(detectron2.model_zoo.get_config_file(MODEL_FILE))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(MODEL_FILE)
	predictor = detectron2.engine.DefaultPredictor(cfg)

	im = cv2.imread("detectron_ros/resources/Untitled.png")
	outputs = predictor(im)
	print(outputs)

__main__()