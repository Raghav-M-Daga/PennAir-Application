"""
The detector itself, kept in its own file so the ROS node and the plain
scripts run exactly the same code. This is a copy of part 3's pipeline with
part 4's depth maths added, nothing about it is ROS specific.
"""

import cv2
import numpy as np

WINDOW = 9         # size of the patch I measure roughness over
SETTLE = 9         # blur the roughness map so the outline stops jittering
FLATNESS = 0.55    # smooth means below 55% of the frame's usual roughness
MIN_AREA = 1000    # same as part 3, so half visible shapes still count
SEAM = 60          # colour jump big enough to mean two shapes are touching
CORNER_CUT = 0.005 # how much the drawn outline may cut a corner

# the camera matrix the task gave us
K = np.array([[2564.3186869, 0.0, 0.0],
              [0.0, 2569.70273111, 0.0],
              [0.0, 0.0, 1.0]])
FX = K[0, 0]
FY = K[1, 1]
CX = K[0, 2]
CY = K[1, 2]
CIRCLE_RADIUS = 10.0      # inches, given in the task
ROUND_ENOUGH = 0.95       # circularity above this means "that is the circle"

SPECK = 9          # specks narrower than this get wiped out
SPECKLE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SPECK, SPECK))
FONT = cv2.FONT_HERSHEY_SIMPLEX
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def measure_roughness(gray):
    """How much each pixel's neighbourhood varies.

    Standard deviation of every 9x9 patch, worked out with
    std = sqrt(average of squares - square of the average).
    """
    mean = cv2.boxFilter(gray, -1, (WINDOW, WINDOW))
    mean_of_squares = cv2.boxFilter(gray * gray, -1, (WINDOW, WINDOW))

    # a flat shape has variance 0 and rounding can push it slightly negative,
    # which makes sqrt return nan and puts holes in the shapes
    variance = np.maximum(mean_of_squares - mean * mean, 0)
    return cv2.blur(np.sqrt(variance), (SETTLE, SETTLE))


def find_smooth_areas(image):
    """Mask that is 1 on the flat areas and 0 on the textured background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    roughness = measure_roughness(gray)

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
    """Where two shapes are touching, so they can be split apart."""
    small = cv2.resize(image, None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)
    change = cv2.morphologyEx(small, cv2.MORPH_GRADIENT,
                              np.ones((3, 3), np.uint8))
    change = change.max(axis=2)

    height, width = mask.shape
    seams = cv2.resize(change, (width, height),
                       interpolation=cv2.INTER_NEAREST)
    return seams > SEAM


def tidy_outline(contour):
    """Convex hull to fill in whatever is covered, then simplify."""
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


def find_depth(shapes):
    """Distance to the surface in inches, measured off the circle."""
    for shape in shapes:
        area = cv2.contourArea(shape)
        perimeter = cv2.arcLength(shape, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > ROUND_ENOUGH:
            radius_in_pixels = np.sqrt(area / np.pi)
            focal_length = (FX + FY) / 2
            return focal_length * CIRCLE_RADIUS / radius_in_pixels
    return None


def pixel_to_world(cx, cy, depth):
    """Turn a pixel position into inches, now that the depth is known."""
    x = (cx - CX) * depth / FX
    y = (cy - CY) * depth / FY
    return x, y


def draw_label(image, text, cx, cy):
    """Thick black behind thin white so the text reads on any colour."""
    spot = (cx - 95, cy - 25)
    cv2.putText(image, text, spot, FONT, 0.8, BLACK, 5, cv2.LINE_AA)
    cv2.putText(image, text, spot, FONT, 0.8, WHITE, 2, cv2.LINE_AA)


def draw_shapes(image, shapes, depth):
    """Outline every shape and label its center."""
    for shape in shapes:
        cx, cy = find_center(shape)
        cv2.drawContours(image, [shape], -1, YELLOW, 3)
        cv2.drawMarker(image, (cx, cy), BLACK, cv2.MARKER_CROSS, 30, 3)

        if depth is None:
            continue
        x, y = pixel_to_world(cx, cy, depth)
        draw_label(image, f"x{x:.0f} y{y:.0f} z{depth:.0f} in", cx, cy)
