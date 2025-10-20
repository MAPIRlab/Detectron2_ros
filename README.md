# Detectron_ros

A ROS2 node for performing instance segmentation with [detectron2](https://github.com/facebookresearch/detectron2/tree/main). Requires detectron2 to be installed on your system.

## Dependencies
This project depends on ROS [vision_msgs](http://wiki.ros.org/vision_msgs), which can be installed as follows:

```sudo apt install ros-humble-vision-msgs```

## Usage
You can select a specific model using the `model_file` ros parameter, which specifies a given `.yaml` file under `detectron2/configs`. Detectron2 is then initialized with that model, and you can send it images to be segmented through the `/detectron/segment` service. See the [service message specification](segmentation_msgs/srv/SegmentImage.srv) for details about the output format. 

If the `publish_visualization` parameter is set to true, the node also publishes a version of image annotated with the masks and scores, on a topic specified through the `visualization_topic` parameter.

## Usage with [Voxeland](https://github.com/MAPIRlab/Voxeland)

### 1. Build the Workspace

Clean previous build artifacts and execute `colcon build` to compile the workspace. Please adapt the paths if necessary:

```bash
cd ~/ros2_ws
rm -rf build/ install/ log/
colcon build --symlink-install --cmake-clean-cache
```

### 2. Launch the Detectron2 ROS 2 Node

Please adapt the paths if necessary:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run detectron_ros detectron_ros_node
```

### 3. Run Voxeland and Play a ScanNet ROS Bag

Create and execute a bash script that contains the following commands. Please adapt the paths if necessary:

```bash
cd ~/ros2_ws

# Init voxeland_robot_perception with Detectron2 detector  
gnome-terminal -- bash -c "source ~/.bashrc; source /home/ubuntu/ros2_ws/venvs/voxenv/bin/activate; ros2 launch voxeland_robot_perception semantic_mapping.launch.py object_detector:=detectron2; exec bash"

# Init voxeland server
gnome-terminal -- bash -c "ros2 launch voxeland voxeland_server.launch.xml; exec bash"

# Open bag folder and play ros2 bag
gnome-terminal -- bash -c "cd /home/ubuntu/ros2_ws/bag/ScanNet/to_ros/ROS2_bags/scene0000_01/; ros2 bag play scene0000_01.db3; exec bash"
```

### Service Interface

**Service:** `/yoloe/segment`  
**Type:** `segmentation_msgs/srv/SegmentImage`

**Request:**
- `sensor_msgs/Image image` - Input RGB image

**Response:**
- `segmentation_msgs/SemanticInstance2D[] instances` - Detected objects with masks and bounding boxes

**Visualization Topic:** `/segmentedImage` (if enabled)