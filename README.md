# Detectron_ros

A ROS2 node for performing instance segmentation with [detectron2](https://github.com/facebookresearch/detectron2/tree/main). Requires detectron2 to be installed on your system.

## Usage
You can select a specific model using the `model_file` ros parameter, which specifies a given `.yaml` file under `detectron2/configs`. Detectron2 is then initialized with that model, and you can send it images to be segmented through the `/detectron/segment` service. See the [service message specification](detectron_msgs/srv/SegmentImage.srv) for details about the output format. 

If the `publish_visualization` parameter is set to true, the node also publishes a version of image annotated with the masks and scores, on a topic specified through the `visualization_topic` parameter.