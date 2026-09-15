"""
the detector on its own, with no ROS in it.

This is part 3's pipeline with part 4's depth maths on top. It sits in its own
file so the ROS node can import it instead of keeping a second copy, which
means the node really is running the same code as the plain scripts.
"""

import cv2
import numpy as np

PATCH = 9         # size of the patch I measure roughness over
BLUR = 9         # blur the roughness map so the outline stops jittering
SMOOTHNESS = 0.55    # smooth means below 55% of the frame's usual roughness
MIN_AREA = 1000    # same as part 3 so half visible shapes count
COLOR_SEAM = 60          # colour jump big enough to mean two shapes are touching
CORNER_CUT = 0.005 # how much the drawn outline may cut a corner
SPECK = 9          # specks narrower than this get wiped out

# the camera matrix from the task
K = np.array([[2564.3186869, 0.0, 0.0],
              [0.0, 2569.70273111, 0.0],
              [0.0, 0.0, 1.0]])
FX = K[0, 0]              # focal length in pixels, one per axis
FY = K[1, 1]
CX = K[0, 2]              # principal point, which this matrix puts at (0, 0)
CY = K[1, 2]
CIRCLE_RADIUS = 10.0      # inches, given
ROUND_ENOUGH = 0.95       # circularity over this means it is the circle

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


def find_depth(shapes):
    """how far away the surface is, worked out from the circle.

    Circularity is 4*pi*area / perimeter^2, which is exactly 1.0 for a
    perfect circle and lower for everything else, so no shape classifier
    needed. Then Z = focal length * real radius / radius in pixels, which is
    just similar triangles.
    """
    for shape in shapes:
        area = cv2.contourArea(shape)
        perimeter = cv2.arcLength(shape, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > ROUND_ENOUGH:
            radius_in_pixels = np.sqrt(area / np.pi)
            focal_length = (FX + FY) / 2
            return focal_length * CIRCLE_RADIUS / radius_in_pixels
    return None     # circle is not on screen this frame


def pixel_to_world(cx, cy, depth):
    """turn a pixel position into inches now that the depth is known"""
    x = (cx - CX) * depth / FX      # inches right of the camera axis
    y = (cy - CY) * depth / FY      # inches below the camera axis
    return x, y


def draw_label(image, text, cx, cy):
    spot = (cx - 95, cy - 25)
    cv2.putText(image, text, spot, FONT, 0.8, BLACK, 5, cv2.LINE_AA)
    cv2.putText(image, text, spot, FONT, 0.8, WHITE, 2, cv2.LINE_AA)


def draw_shapes(image, shapes, depth):
    for shape in shapes:
        cx, cy = find_center(shape)
        cv2.drawContours(image, [shape], -1, (0, 255, 255), 3)
        cv2.drawMarker(image, (cx, cy), BLACK, cv2.MARKER_CROSS, 30, 3)

        if depth is None:
            continue
        x, y = pixel_to_world(cx, cy, depth)
        draw_label(image, f"x{x:.0f} y{y:.0f} z{depth:.0f} in", cx, cy)
