"""
find the edges and centers of each shape but with a stream of images.

Blur + Canny does not work here since trapezoid green is too similar

The textures are more distinct: grass is rough and the shapes are flat. So for
every pixel I measure how much its neighbourhood varies. Grass scores about
20, a flat shape about 0.3, plenty to find a difference.

Frames are processed one at a time, so this should work live.
"""

import subprocess
import cv2
import imageio_ffmpeg
import numpy as np

VIDEO = "files/PennAir 2024 App Dynamic.mp4"
OUTPUT = "part2/output_annotated.mp4"

PATCH = 9         # size of the patch I measure roughness over
BLUR = 9         # blur the roughness map so the outline stops jittering
SMOOTHNESS = 0.55    # smooth means below 55% of the frame's usual roughness
MIN_AREA = 2000    # smaller than this is not a shape
COLOR_SEAM = 60          # colour jump big enough to mean two shapes are touching
CORNER_CUT = 0.005 # how much the drawn outline may cut a corner
SPECK = 9          # specks narrower than this get wiped out

FONT = cv2.FONT_HERSHEY_SIMPLEX
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SPECKLE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SPECK, SPECK))


def measure_roughness(gray):
    """how much each pixel's neighbourhood varies.

    This is the standard deviation of every 9x9 patch. Both averages are
    box filters so it costs two blurs no matter how big the window is.
    """
    mean = cv2.boxFilter(gray, -1, (PATCH, PATCH))
    mean_of_squares = cv2.boxFilter(gray * gray, -1, (PATCH, PATCH))

    # shapes can be so smooth that rounding can make them negative
    variance = np.maximum(mean_of_squares - mean * mean, 0)
    roughness = np.sqrt(variance)

    # average the roughness so its not noisy
    return cv2.blur(roughness, (BLUR, BLUR))


def find_smooth_areas(image):
    """mask that is 1 on the flat areas and 0 on the textured background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    roughness = measure_roughness(gray)

    # compare against this frame's own roughness so the threshold adapts to
    # each scene. Every 8th pixel is averaged for the median
    usual = np.median(roughness[::8, ::8])
    mask = (roughness < SMOOTHNESS * usual).astype(np.uint8)

    return remove_specks(mask)


def remove_specks(mask):
    """shrinks, removes specks so the outline is not jagged, then scales back up"""
    height, width = mask.shape
    small = cv2.resize(mask, None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_NEAREST)
    small = cv2.morphologyEx(small, cv2.MORPH_OPEN, SPECKLE)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def find_seams(image, mask):
    """find the edges between touching shapes. 
    Shapes have similar smoothness so they come out as one blob. 
    However, in color it's more noticable, so change in color can differentiate
    shapes from each other. 
    """
    small = cv2.resize(image, None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)
    change = cv2.morphologyEx(small, cv2.MORPH_GRADIENT,
                              np.ones((3, 3), np.uint8))
    change = change.max(axis=2)    # strongest change of the three channels

    height, width = mask.shape
    seams = cv2.resize(change, (width, height),
                       interpolation=cv2.INTER_NEAREST)
    return seams > COLOR_SEAM


def tidy_outline(contour):
    """straightens the outline a bit so it has smoother straight edges"""
    hull = cv2.convexHull(contour)
    limit = CORNER_CUT * cv2.arcLength(hull, True)
    return cv2.approxPolyDP(hull, limit, True)


def find_shapes(image):
    mask = find_smooth_areas(image)
    mask[find_seams(image, mask)] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        if cv2.contourArea(contour) > MIN_AREA:
            shapes.append(tidy_outline(contour))
    return shapes


def find_center(contour):
    m = cv2.moments(contour)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def draw_label(image, text, cx, cy):
    spot = (cx - 60, cy - 25)
    cv2.putText(image, text, spot, FONT, 0.8, BLACK, 5, cv2.LINE_AA)
    cv2.putText(image, text, spot, FONT, 0.8, WHITE, 2, cv2.LINE_AA)


def draw_shapes(image, shapes):
    for shape in shapes:
        cx, cy = find_center(shape)
        cv2.drawContours(image, [shape], -1, (0, 255, 255), 3)
        cv2.drawMarker(image, (cx, cy), BLACK, cv2.MARKER_CROSS, 30, 3)
        draw_label(image, f"({cx}, {cy})", cx, cy)


def start_writer(path, width, height, fps):
    """Send frames to ffmpeg to create a H.264 video from the BGR frames.
    Makes the file size a bit smaller
    """
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-",
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-crf", "20", "-movflags", "+faststart", path,
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def run():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"could not open {VIDEO}, run this from the root")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = start_writer(OUTPUT, width, height, fps)

    frames = 0
    while True:
        ok, frame = cap.read()      # one frame, as it arrives
        if not ok:
            break                   # video finished

        shapes = find_shapes(frame)
        draw_shapes(frame, shapes)

        writer.stdin.write(frame.tobytes())
        frames += 1

    cap.release()
    writer.stdin.close()
    writer.wait()
    print(f"{frames} frames -> {OUTPUT}")


if __name__ == "__main__":
    run()
