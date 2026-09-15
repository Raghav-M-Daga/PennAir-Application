from setuptools import setup

package_name = "pennair_shapes"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/shapes.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Raghav",
    maintainer_email="raghavdaga2007@gmail.com",
    description="Shape detection on a video stream, published on ROS 2 topics.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "video_publisher = pennair_shapes.video_publisher:main",
            "shape_detector = pennair_shapes.shape_detector:main",
            "video_recorder = pennair_shapes.video_recorder:main",
        ],
    },
)
