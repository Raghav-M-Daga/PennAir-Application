"""
Start both nodes together:

    ros2 launch pennair_shapes shapes.launch.py video:=/path/to/video.mp4

The publisher reads the video and puts frames on /camera/image_raw; the
detector picks them up there and publishes /shapes/image and
/shapes/centers. Watch the result with:

    ros2 run rqt_image_view rqt_image_view /shapes/image
    ros2 topic echo /shapes/centers
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    video = LaunchConfiguration("video")
    output = LaunchConfiguration("output")

    return LaunchDescription([
        DeclareLaunchArgument(
            "video",
            description="path to the input video file",
        ),
        DeclareLaunchArgument(
            "output",
            default_value="output_ros.mp4",
            description="where to save the annotated video",
        ),
        Node(
            package="pennair_shapes",
            executable="video_publisher",
            name="video_publisher",
            parameters=[{"video": video}],
            output="screen",
        ),
        Node(
            package="pennair_shapes",
            executable="shape_detector",
            name="shape_detector",
            output="screen",
        ),
        Node(
            package="pennair_shapes",
            executable="video_recorder",
            name="video_recorder",
            parameters=[{"output": output}],
            output="screen",
        ),
    ])
