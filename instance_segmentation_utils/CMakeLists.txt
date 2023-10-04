cmake_minimum_required(VERSION 3.8)
project(instance_segmentation_utils)

## Dependencies
find_package(ament_cmake REQUIRED)
set(AMENT_DEPS
    rclcpp
    tf2
	tf2_ros
	tf2_geometry_msgs
	sensor_msgs
	segmentation_msgs
	cv_bridge
    )
foreach(Dependency IN ITEMS ${AMENT_DEPS})
  find_package(${Dependency} REQUIRED)
endforeach()

find_package(OpenCV REQUIRED)

## TARGETS
include_directories(include)

add_executable(projectTo3D src/projectTo3D.cpp)
ament_target_dependencies(projectTo3D ${AMENT_DEPS})
target_link_libraries(projectTo3D ${OpenCV_LIBS})

## INSTALL
install(
	TARGETS projectTo3D
	DESTINATION lib/${PROJECT_NAME}
)
ament_package()
