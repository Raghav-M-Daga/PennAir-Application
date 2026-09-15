"""
Node 1: read the video file and publish its frames on /camera/image_raw.

This stands in for the camera. It reads one frame at a time and puts it on a
topic at the video's own frame rate, and nothing downstream knows or cares
that the frames came from a file instead of a real camera. Swapping in a real
camera driver that publishes the same topic would change nothing else.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")
        self.declare_parameter("video", "")
        path = self.get_parameter("video").value

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open {path}")

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)

        # publish on a timer at the video's own frame rate, so it comes out
        # in real time instead of as fast as the disk can read
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.create_timer(1.0 / fps, self.publish_frame)
        self.get_logger().info(f"publishing {path} at {fps:.1f} fps")

    def publish_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the video
            return

        message = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        # stamp it with the time and the frame it came from. anything that
        # lines messages up later (rviz, tf) needs these to be filled in.
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera"
        self.publisher.publish(message)


def main():
    rclpy.init()
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
