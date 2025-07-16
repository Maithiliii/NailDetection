import cv2
import numpy as np

def get_nail_height_and_weight(image, box, pixel_to_mm=0.2, weight_per_mm=0.03):
    """
    Estimate height (mm) and weight (g) of a nail from its bounding box and pixel size.
    
    Args:
        image (np.ndarray): Original BGR image
        box (list or array): [x1, y1, x2, y2] bounding box
        pixel_to_mm (float): Conversion ratio from pixels to mm (e.g., 1 pixel = 0.2mm)
        weight_per_mm (float): Assumed weight per mm (e.g., 0.03g per mm)

    Returns:
        height_mm (float), weight_g (float)
    """
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]

    # Convert to grayscale & threshold to get contour
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0

    # Get largest contour (assumed to be the nail)
    largest = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(largest)

    height_mm = h * pixel_to_mm
    weight_g = height_mm * weight_per_mm

    return round(height_mm, 2), round(weight_g, 3)

