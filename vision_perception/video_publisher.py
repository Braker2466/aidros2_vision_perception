import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import aidcv as cv2
from ament_index_python.packages import get_package_share_directory

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        resource_dir = get_package_share_directory('vision_perception')
        video_name = 'testvideo2.mp4'
        video_path = os.path.join(resource_dir, video_name)
        # video_path = '/home/aidlux/aidcode/test_ws/src/vision_perception/vision_perception/testvideo1.mp4'  # 修改为你的视频绝对路径
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.get_logger().error('无法打开视频文件！')
            rclpy.shutdown()
            return

        timer_period = 1/10  # 约 10 帧每秒
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('视频发布节点启动 ✅')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info('📽️ 视频播放完毕，停止发布。')
            self.cap.release()
            self.timer.cancel()
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()