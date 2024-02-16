# Header
std_msgs/Header header

# Point cloud, where each point is annotated with the instance ID.
sensor_msgs/PointCloud2 cloud

# Object instances, annotated with their ID and probability distribution over the set of object categories
vision_msgs/Detection2D[] instances

# Object categories from the object detector
string[] categories

# Sensor pose
geometry_msgs/PoseWithCovariance pose