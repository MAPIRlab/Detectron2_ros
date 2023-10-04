#include <cv_bridge/cv_bridge.h>
#include <instance_segmentation_utils/projectTo3D.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<ProjectTo3D>();

	// we have a polling loop rather than just spinning and letting the callback handle everything to avoid the headache
	// of calling a service inside a callback
	// technically doable, I think, but recursive spinning is not allowed, so not trivial
	rclcpp::Rate rate(30);
	while (rclcpp::ok())
	{
		rclcpp::spin_some(node);
		node->processImage();
		rate.sleep();
	}
	return 0;
}

ProjectTo3D::ProjectTo3D() : Node("ProjectInstancesTo3D"), m_tfBuffer(get_clock())
{
	using namespace std::placeholders;

	std::string camera_topic = declare_parameter<std::string>("camera_topic", "camera");
	m_cameraSub = create_subscription<Image>(camera_topic, 5, [this](const Image::SharedPtr msg) { m_lastImage = msg; });

	m_detectronClient = create_client<segmentation_msgs::srv::SegmentImage>("/detectron/segment");
	m_pub3D = create_publisher<ObjectWithBoundingBox3DArray>("semantic_instances_3D", 10);

	m_cameraCalibration.centerX = declare_parameter<float>("imageCenter_x", 320);
	m_cameraCalibration.centerY = declare_parameter<float>("imageCenter_y", 240);

	m_cameraCalibration.focalY = declare_parameter<float>("focalDistance_y", 457.14);
	m_cameraCalibration.focalY = declare_parameter<float>("focalDistance_y", 470.58);

	m_cameraCalibration.depthLimit = declare_parameter<float>("depthLimit", 15.0);

	while (!m_detectronClient->wait_for_service(std::chrono::seconds(5)))
		RCLCPP_INFO(get_logger(), "Waiting for service to become available...");
}

void ProjectTo3D::processImage()
{
	if (!m_lastImage)
		return;

	cv::Mat rgb;
	cv::Mat depth;
	// split the image into rgb and d
	{
		cv::Mat channels[4];
		cv_bridge::CvImagePtr receivedImageCV = cv_bridge::toCvCopy(*m_lastImage, m_lastImage->encoding);
		cv::split(receivedImageCV->image, channels);

		cv::merge(channels, 3, rgb);
		depth = channels[3];
	}

	// send the rgb image to detectron
	auto request = std::make_shared<segmentation_msgs::srv::SegmentImage::Request>();
	cv_bridge::CvImage(m_lastImage->header, "rgb8", rgb).toImageMsg(request->image);
	auto header = m_lastImage->header;
	m_lastImage = nullptr;

	auto future = m_detectronClient->async_send_request(request);
	if (rclcpp::spin_until_future_complete(shared_from_this(), future) == rclcpp::FutureReturnCode::SUCCESS)
		projectInstancesAndPublish(future.get(), depth, header);

	else
		RCLCPP_ERROR(get_logger(), "Could not access the detectron service!");
}

void ProjectTo3D::projectInstancesAndPublish(segmentation_msgs::srv::SegmentImage::Response::SharedPtr response, const cv::Mat& depth,
											 std_msgs::msg::Header image_header)
{
	cv::Mat x_coord(cv::Size(depth.rows, depth.cols), CV_32FC1);
	cv::Mat y_coord(cv::Size(depth.rows, depth.cols), CV_32FC1);
	cv::Mat z_coord(cv::Size(depth.rows, depth.cols), CV_32FC1);
	{
#pragma omp parallel for collapse(2)
		for (int column = 0; column < x_coord.cols; column++)
		{
			for (int row = 0; row < x_coord.rows; row++)
			{
				float z = depth.at<uint8_t>(row, column) / 255.0 * m_cameraCalibration.depthLimit;
				x_coord.at<float>(column, row) = (m_cameraCalibration.centerX - column) * (z / m_cameraCalibration.focalX);
				y_coord.at<float>(column, row) = (m_cameraCalibration.centerY - row) * (z / m_cameraCalibration.focalY);
				z_coord.at<float>(column, row) = z;
			}
		}
	}

	// process the instances

	ObjectWithBoundingBox3DArray objectsArray;
	for (const segmentation_msgs::msg::SemanticInstance2D& instance : response->instances)
	{
		objectsArray.objects.emplace_back();
		auto& objectBB = objectsArray.objects.back();

		objectBB.classifications = instance.classifications;

		cv_bridge::CvImagePtr mask = cv_bridge::toCvCopy(instance.mask);

		// bandpass filter depth data. Lifted straight from the other implementation, don't ask me why this is a thing
		{
			double min_z, max_z;
			cv::minMaxIdx(z_coord, &min_z, &max_z, nullptr, nullptr, mask->image);
			float top_margin = (max_z - min_z) * 0.9 + min_z;
			float bottom_margin = (max_z - min_z) * 0.1 + min_z;

			for (int column = 0; column < mask->image.cols; column++)
			{
				for (int row = 0; row < mask->image.rows; row++)
				{
					float z = z_coord.at<float>(column, row);
					if (z < bottom_margin || z > top_margin)
						mask->image.at<float>(column, row) = 0;
				}
			}
		}

		double x_min, x_max, y_min, y_max, z_min, z_max;
		cv::minMaxIdx(x_coord, &x_min, &x_max, nullptr, nullptr, mask->image);
		cv::minMaxIdx(y_coord, &y_min, &y_max, nullptr, nullptr, mask->image);
		cv::minMaxIdx(z_coord, &z_min, &z_max, nullptr, nullptr, mask->image);

		geometry_msgs::msg::TransformStamped transform =
			m_tfBuffer.buffer.lookupTransform(image_header.frame_id, "map", tf2_ros::fromMsg(image_header.stamp));

		objectBB.size.x = x_max - x_min;
		objectBB.size.y = y_max - y_min;
		objectBB.size.z = z_max - z_min; // originally, this was the stdDev of the z values in the mask
		tf2::doTransform(objectBB.size, objectBB.size, transform);

		objectBB.center.header.frame_id = "map";
		objectBB.center.header.stamp = image_header.stamp;
		objectBB.center.point.x = x_min + objectBB.size.x * 0.5;
		objectBB.center.point.y = y_min + objectBB.size.y * 0.5;
		objectBB.center.point.z = z_min + objectBB.size.z * 0.5;
		tf2::doTransform(objectBB.center, objectBB.center, transform);
	}

	m_pub3D->publish(objectsArray);
}
