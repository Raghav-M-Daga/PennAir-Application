"""
Part 3: the same detector as part 2, run on the harder video.

Only two things change: the file it opens and MIN_AREA, which I dropped so
shapes that are half off the edge of the frame still get picked up. The
algorithm is untouched, which is the point, it transfers without edits.

This video is harder in two ways at once. The background is gravel instead of
grass, and the shapes are filled with colour gradients instead of flat colour.
Both squeeze the gap the roughness test works in: a flat shape used to read
0.00 against grass at 14.47, here a gradient shape reads 3.99 against gravel
at 9.40. It still separates, just with a lot less room.

Run it from the project root:  python part3/detect_video2.py
"""

import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

VIDEO = "files/PennAir 2024 App Dynamic Hard.mp4"
OUTPUT = "part3/output_annotated_2.mp4"

WINDOW = 9         # size of the patch I measure roughness over
SETTLE = 9         # blur the roughness map so the outline stops jittering
FLATNESS = 0.55    # smooth means below 55% of the frame's usual roughness
MIN_AREA = 1000    # lower than part 2 so half visible shapes count
SEAM = 60          # colour jump big enough to mean two shapes are touching
CORNER_CUT = 0.005 # how much the drawn outline may cut a corner

SPECK = 9          # specks narrower than this get wiped out
SPECKLE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SPECK, SPECK))
FONT = cv2.FONT_HERSHEY_SIMPLEX
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def measure_roughness(gray):
    """How much each pixel's neighbourhood varies.

    This is the standard deviation of every 9x9 patch, worked out with
    std = sqrt(average of squares - square of the average). Both averages are
    box filters so it costs two blurs no matter how big the window is.
    """
    mean = cv2.boxFilter(gray, -1, (WINDOW, WINDOW))
    mean_of_squares = cv2.boxFilter(gray * gray, -1, (WINDOW, WINDOW))

    # inside a perfectly flat shape the variance is 0, and rounding can push
    # it slightly negative, which would make sqrt return nan and punch holes
    # in the shapes. so clamp it first.
    variance = np.maximum(mean_of_squares - mean * mean, 0)
    roughness = np.sqrt(variance)

    # roughness is a measurement and it is noisy near a low contrast edge,
    # so average it before deciding anything
    return cv2.blur(roughness, (SETTLE, SETTLE))


def find_smooth_areas(image):
    """Mask that is 1 on the flat areas and 0 on the textured background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    roughness = measure_roughness(gray)

    # compare against this frame's own roughness so the threshold adapts to
    # the scene instead of being a fixed number. every 8th pixel is plenty
    # for a median and it saves about 18 ms a frame.
    usual = np.median(roughness[::8, ::8])
    mask = (roughness < FLATNESS * usual).astype(np.uint8)

    return remove_specks(mask)


def remove_specks(mask):
    """Wipe out the little smooth patches so only real shapes are left.

    An opening is an erode then a dilate. The erode eats a layer off
    everything, which kills the specks outright and snaps the thin bridges
    where one has stuck onto a shape. The dilate then grows whatever
    survived back to its old size, so the shapes come out unharmed.

    This does a lot of work: without it there are around 260 contours to
    trace per frame, with it there are 5.

    Shrinking to half size first is not only about speed. A 9 pixel kernel
    on a half size mask reaches as far as a 17 pixel one would on the full
    mask, so I get the wider reach and it still runs about 8x faster.
    """
    height, width = mask.shape
    small = cv2.resize(mask, None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_NEAREST)
    small = cv2.morphologyEx(small, cv2.MORPH_OPEN, SPECKLE)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def find_seams(image, mask):
    """Find where two shapes are touching, so I can split them apart.

    Two overlapping shapes come out as one blob, because the seam between
    them is smooth as well. In grey the magenta pentagon is 105 and the green
    trapezoid is 125, only 20 apart, so the roughness test cannot see it. In
    colour the same seam is obvious. The morphological gradient (dilate minus
    erode) says how fast the colour changes, and a shape's own inside reads 0.
    """
    small = cv2.resize(image, None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)
    change = cv2.morphologyEx(small, cv2.MORPH_GRADIENT,
                              np.ones((3, 3), np.uint8))
    change = change.max(axis=2)    # strongest change of the three channels

    height, width = mask.shape
    seams = cv2.resize(change, (width, height),
                       interpolation=cv2.INTER_NEAREST)
    return seams > SEAM


def tidy_outline(contour):
    """Clean up a traced outline.

    Every shape in this task is convex, so the convex hull fills back in
    whatever another shape was covering and straightens out any wobble.
    approxPolyDP then drops the leftover points down to the real corners.
    """
    hull = cv2.convexHull(contour)
    limit = CORNER_CUT * cv2.arcLength(hull, True)
    return cv2.approxPolyDP(hull, limit, True)


def find_shapes(image):
    """Return a tidy contour for each shape in the frame."""
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
    """Center of a contour, using image moments."""
    m = cv2.moments(contour)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def draw_label(image, text, cx, cy):
    """Thick black behind thin white so the text reads on any colour."""
    spot = (cx - 60, cy - 25)
    cv2.putText(image, text, spot, FONT, 0.8, BLACK, 5, cv2.LINE_AA)
    cv2.putText(image, text, spot, FONT, 0.8, WHITE, 2, cv2.LINE_AA)


def draw_shapes(image, shapes):
    """Outline every shape and mark its center."""
    for shape in shapes:
        cx, cy = find_center(shape)
        cv2.drawContours(image, [shape], -1, YELLOW, 3)
        cv2.drawMarker(image, (cx, cy), BLACK, cv2.MARKER_CROSS, 30, 3)
        draw_label(image, f"({cx}, {cy})", cx, cy)


def start_writer(path, width, height, fps):
    """Send frames to ffmpeg instead of cv2.VideoWriter.

    OpenCV has no H.264 encoder of its own on Windows, so it falls back to
    something a lot of players refuse to open. ffmpeg gives a normal H.264
    file at a sensible size.
    """
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-",
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-crf", "20", "-movflags", "+faststart", path,
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main():
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
    main()
