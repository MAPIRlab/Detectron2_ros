import rclpy
import ament_index_python
import rclpy.node
from cv_bridge import CvBridge
import cv2
import segmentation_msgs.srv
import os.path

class TestDetectron(rclpy.node.Node):
    def __init__(self):
        super().__init__("Test_detectron")
        self.client = self.create_client(segmentation_msgs.srv.SegmentImage, "/detectron/segment")

    def sendRequest(self):
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service to become available...')
            
        image_cv = cv2.imread(os.path.join(ament_index_python.get_package_share_directory("detectron_ros"), "resources", "Untitled.png") )
        image_msg = CvBridge().cv2_to_imgmsg(image_cv)
        request = segmentation_msgs.srv.SegmentImage.Request()
        request.image = image_msg
        
        self.get_logger().info("Sending request")
        future = self.client.call_async(request=request)
        rclpy.spin_until_future_complete(self, future)
        self._logger.info(f"Found {len(future.result().instances)} masks in the image")
        self._logger.info(f"First one classfied as: {future.result().instances[0].classifications}")

def main(args=None):
    rclpy.init(args=args)
    node = TestDetectron()
    node.sendRequest()
    rclpy.spin(node)


if __name__ == '__main__':
    main()