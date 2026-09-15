"""
Node 3: save the annotated stream to a video file.

Just a subscriber. It takes whatever shows up on /shapes/image and writes it
to disk, which is how the ROS side produces an output video like the other
parts do, except here the frames travelled over a topic instead of through a
function call.

It is a separate node on purpose. The detector's job is finding shapes, not
knowing about files, and anything else that wants to record a stream can
reuse this node by pointing it at another topic.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoRecorder(Node):
    def __init__(self):
        super().__init__("video_recorder")
        self.declare_parameter("output", "output_ros.mp4")
        self.declare_parameter("fps", 30.0)
        self.path = self.get_parameter("output").value
        self.fps = self.get_parameter("fps").value

        self.bridge = CvBridge()
        self.writer = None      # opened on the first frame, once I know
        self.frames = 0         # how big the pictures are
        self.create_subscription(Image, "/shapes/image", self.on_image, 10)
        self.get_logger().info(f"recording /shapes/image to {self.path}")

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        if self.writer is None:
            height, width = frame.shape[:2]
            # avc1 is H.264, which normal video players can actually open
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"avc1"),
                self.fps, (width, height))
            if not self.writer.isOpened():
                raise SystemExit(f"could not open {self.path} for writing")

        self.writer.write(frame)
        self.frames += 1
        if self.frames % 100 == 0:
            self.get_logger().info(f"{self.frames} frames recorded")

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(f"{self.frames} frames -> {self.path}")


def main():
    rclpy.init()
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
