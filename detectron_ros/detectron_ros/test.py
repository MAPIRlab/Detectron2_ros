import rclpy
import ament_index_python
import rclpy.node
from cv_bridge import CvBridge
import cv2
import detectron_msgs.srv
import os.path

class TestDetectron(rclpy.node.Node):
    def __init__(self):
        super().__init__("Test_detectron")
        self.client = self.create_client(detectron_msgs.srv.SegmentImage, "/detectron/segment")

    def sendRequest(self):
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
            
        image_cv = cv2.imread(os.path.join(ament_index_python.get_package_share_directory("detectron_ros"), "resources", "Untitled.png") )
        image_msg = CvBridge().cv2_to_imgmsg(image_cv)
        request = detectron_msgs.srv.SegmentImage.Request()
        request.image = image_msg
        self.client.call(request=request)

def main(args=None):
    rclpy.init(args=args)
    node = TestDetectron()
    node.sendRequest()


if __name__ == '__main__':
    main()