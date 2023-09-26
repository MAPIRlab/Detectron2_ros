# Bounding boxes in pixels
sensor_msgs/RegionOfInterest box
# Instance masks as Image
sensor_msgs/Image mask

int32 class_id

# Float probability scores of the class_id
float32 score
float32[] scores_all_classes