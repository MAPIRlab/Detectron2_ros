import detectron2
import detectron2.config
import detectron2.model_zoo
import detectron2.engine
import cv2

def __main__():

	MODEL_FILE="COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"

	cfg = detectron2.config.get_cfg()
	cfg.merge_from_file(detectron2.model_zoo.get_config_file(MODEL_FILE))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(MODEL_FILE)
	predictor = detectron2.engine.DefaultPredictor(cfg)

	im = cv2.imread("src/Untitled.png")
	outputs = predictor(im)
	print(outputs)

__main__()