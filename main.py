# -*- coding: utf-8 -*-

# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import cv2
import numpy as np
from process_image import load_image, create_roi, segment_image
from leNet4 import load_single_model, predict, predict_boosted
import os

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Constant
PATH_MODEL = "./model/model.h5"
FILE_DATA = "path_to_your_data"
PATH_RESULT = "path_to_your_result_folder"
FILE_RESULT = "your_result_file.csv"

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_image(model, path_data=FILE_DATA, boost=False):
    image = load_image(path_data)
    if image is None:
        return 0
    
    ROI = cv2.bitwise_not(create_roi(image, x=60, y=20, width=90, height=650))
    
    try:
        image_segmented = segment(image, ROI)
        if image_segmented is None:
            return 0
        
        predictions_fn = predict_boosted if boost else predict
        predictions = predictions_fn(model, image_segmented)
        return predictions
    except ValueError as e:
        return handle_value_error(e)

def handle_value_error(e):
    return -1
       

def segment(image, roi):
    segmentation = segment_image(roi)
    if segmentation:
        numbers = np.stack(segmentation, axis=0).astype("float32") / 255.0
        return numbers
    return None


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__=="__main__" :

    model = load_single_model(PATH_MODEL)
    process_image(model, FILE_DATA)
        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
