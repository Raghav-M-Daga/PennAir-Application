# Part 5 — ROS 2

## What ROS is actually doing here

Parts 1–4 are one program: read a frame, detect, draw, write. That is fine for
a video file, but it is not how a drone is built. On a real aircraft the camera,
the detector, the navigation logic and the logger are separate programs, often
on separate machines and written by different people. ROS 2 is the plumbing that
lets them talk without knowing about each other.

So Part 5 takes the *same* detection code and splits it up the way a real
system would be split:

| node | role on a real drone |
|---|---|
| `video_publisher` | the camera driver, publishes frames and nothing else |
| `shape_detector` | the perception node, consumes frames and publishes what it found |
| `video_recorder` | the logger, records the stream for later review |

The vision code itself lives in `detection.py`, which the node imports. It is
the same pipeline as part 3 with part 4's depth maths on top, so the ROS
output and `part3/output_annotated_2.mp4` come from one shared file rather
than two copies that can drift apart. `shape_detector.py` is only plumbing.

What that buys you, and what to say if asked why it is worth the trouble:

- **The detector no longer cares where frames come from.** Swap
  `video_publisher` for a real camera driver publishing on the same topic and
  nothing else changes. That is the whole point of a topic.
- **The vision code is shared, not copied.** Both the script and the node
  import `detection.py`, so there is one algorithm and one place to fix it.
- **The results are available to anything else on the network.** A navigation
  node can subscribe to `/shapes/centers` and fly toward a shape without ever
  touching the vision code.
- **You can watch it live and record it separately.** `rqt_image_view`
  subscribes to the annotated stream; `video_recorder` writes it to disk. Both
  are just more subscribers.
- **Nodes fail independently.** If the detector crashes, the camera keeps
  publishing and the recorder keeps running.

## Topics

| topic | type | what |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | raw frames from the video |
| `/shapes/image` | `sensor_msgs/Image` | outlines + centers drawn on |
| `/shapes/centers` | `geometry_msgs/PoseArray` | x, y, z of each center, in inches |

`/shapes/centers` is the one that matters — it is the machine-readable answer.
The annotated image is for humans.

## Build and run (Ubuntu + ROS 2)

```bash
mkdir -p ~/ros2_ws/src
cp -r part5/pennair_shapes ~/ros2_ws/src/
cd ~/ros2_ws
rosdep install --from-paths src -y --ignore-src
colcon build --packages-select pennair_shapes
source install/setup.bash

ros2 launch pennair_shapes shapes.launch.py \
    video:=/absolute/path/to/PennAir\ 2024\ App\ Dynamic\ \(1\).mp4 \
    output:=$HOME/output_ros.mp4
```

Stop it with Ctrl-C; the recorder finalises the file on the way out.

## Look at the output

```bash
ros2 run rqt_image_view rqt_image_view /shapes/image
ros2 topic echo /shapes/centers
ros2 topic hz /shapes/image
```

`output_ros.mp4` in this folder came out of the pipeline above, running on the
hard video. The frames in it went camera node -> topic -> detector node ->
topic -> recorder node, rather than straight through a function call.

## Note on throughput

The publisher runs at roughly half the video's 30 fps. The bottleneck is not
the algorithm: a 1920×1080 `bgr8` frame is **6.2 MB**, so 30 fps means pushing
186 MB/s through the middleware. A real system publishes `CompressedImage`, or keeps the
nodes in one process so the frames never get serialised at all.
