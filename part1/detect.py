"""
find the shapes in the static image, outline them and mark the centers.

The trick here is to blur wider than a blade of grass so the grass smears
into a flat green, but the shapes survive. After that the only edges left in
the picture are the shape outlines, so Canny finds the edges.
"""

import cv2

IMAGE = "files/PennAir 2024 App Static.png"
OUTPUT = "part1/output_detected.png"

BLUR = 21         # wider than the grass texture, has to odd :/
MIN_AREA = 2000   # anything smaller than this is noise, not a shape

FONT = cv2.FONT_HERSHEY_SIMPLEX


def find_shapes(image):
    """return a contour for each shape in the image"""
    # blur away the grass so the background has no edges left
    blurred = cv2.medianBlur(image, BLUR)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        if cv2.contourArea(contour) > MIN_AREA:
            shapes.append(contour)
    return shapes


def find_center(contour):
    """center of a contour, using image moments"""
    m = cv2.moments(contour)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def draw_label(image, text, cx, cy):
    """double text more visible"""
    spot = (cx - 60, cy - 25)
    cv2.putText(image, text, spot, FONT, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, spot, FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_shapes(image, shapes):
    """outline every shape, mark its center"""
    for shape in shapes:
        cx, cy = find_center(shape)
        cv2.drawContours(image, [shape], -1, (0, 255, 255), 3)
        cv2.drawMarker(image, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 30, 3)
        draw_label(image, f"({cx}, {cy})", cx, cy)


def run():
    image = cv2.imread(IMAGE)

    shapes = find_shapes(image)
    draw_shapes(image, shapes)

    cv2.imwrite(OUTPUT, image)


if __name__ == "__main__":
    run()
