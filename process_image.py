# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import cv2
import numpy as np
from pathlib import Path

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# tools
# 0 --> blanc ; 255 --> noir
#------------------------------------------------------------------------------
def load_image(chemin_image):
    chemin_image = Path(chemin_image)
    if not chemin_image.is_file():
        raise FileNotFoundError(f"Image not found at {chemin_image}.")
    return cv2.imread(str(chemin_image), cv2.IMREAD_GRAYSCALE)

def grey_to_binary(image, threshold=127):
    _, image_binaire = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return image_binaire

def get_roi(data, location, roi="rois"):
    return data[roi][location]['x'], data[roi][location]['y'], data[roi][location]['width'], data[roi][location]['height']

def create_roi(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def create_roi_last_row(image, start=40, end=10, left_margin=10, right_margin=10):
    height, width = image.shape
    return image[height-start:height-end, left_margin:width-right_margin]

def segment_image(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        segmented_object = image[y:y+h, x:x+w]
        if is_digit_segment(segmented_object):
            border_width = 3
            segmented_object = np.pad(segmented_object, border_width, mode='constant')
            segmented_object = cv2.resize(segmented_object, (28, 28), interpolation=cv2.INTER_LANCZOS4)
            yield segmented_object

def is_digit_segment(segment):
    pixel_ratio = np.count_nonzero(segment) / segment.size
    if pixel_ratio < 0.9 and (segment.shape[0] >= 7 or segment.shape[1] >= 7):
        return True
    return False
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------