cmake_minimum_required(VERSION 3.8)
project(segmentation_msgs)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)


# services and actions
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
	
	msg/SemanticInstance2D.msg
	msg/SemanticPointCloud.msg
	srv/SegmentImage.srv

    DEPENDENCIES std_msgs sensor_msgs vision_msgs
)
ament_export_dependencies(rosidl_default_runtime)



ament_package()
