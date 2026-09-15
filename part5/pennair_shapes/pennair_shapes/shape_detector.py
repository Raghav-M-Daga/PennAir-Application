"""
node 2: take frames off the camera topic, find the shapes, publish results.

All the vision is in detection.py, the same file the plain scripts use, so
this node is running the part 3 pipeline rather than a copy of it that could
drift. Everything here is just plumbing.

  /shapes/image    the annotated picture, for rqt_image_view
  /shapes/centers  the numbers, a PoseArray of 3D centers in inches
"""

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import Image

from pennair_shapes import detection


class ShapeDetector(Node):
    def __init__(self):
        super().__init__("shape_detector")
        self.bridge = CvBridge()
        self.depth = None      # last known distance to the surface

        self.image_pub = self.create_publisher(Image, "/shapes/image", 10)
        self.pose_pub = self.create_publisher(PoseArray, "/shapes/centers", 10)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.get_logger().info("waiting for /camera/image_raw")

    def on_image(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        shapes = detection.find_shapes(image)

        # keep the old depth when the circle is off screen, the surface is
        # flat so it has not moved
        new_depth = detection.find_depth(shapes)
        if new_depth is not None:
            self.depth = new_depth

        self.publish_centers(shapes, msg.header)

        detection.draw_shapes(image, shapes, self.depth)
        annotated = self.bridge.cv2_to_imgmsg(image, "bgr8")
        annotated.header = msg.header
        self.image_pub.publish(annotated)

    def publish_centers(self, shapes, header):
        """send every center out as a 3D position in inches"""
        poses = PoseArray()
        poses.header = header

        if self.depth is None:
            self.pose_pub.publish(poses)
            return

        for shape in shapes:
            cx, cy = detection.find_center(shape)
            x, y = detection.pixel_to_world(cx, cy, self.depth)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = self.depth
            poses.poses.append(pose)

        self.pose_pub.publish(poses)


def main():
    rclpy.init()
    node = ShapeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
