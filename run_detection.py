import os
import csv
import sys
import typing
from PIL import Image

from detection_model.SKU110K_CVPR19.object_detector_retinanet.keras_retinanet.bin.predict import main


def run_detection(image_folder_path: str, detection_save_folder: str) -> str:
    """Master script to run SKU110K retail object detection on all images in image_folder_path
    
    Arguments:
        image_folder_path {str} -- path to the image folder
        detection_save_folder {str} -- path where to save detection results
    
    Returns:
        return detection_csv_results_file_path {str} -- path to detection results csv file
    """

    model_wights_path = os.path.join(os.getcwd(), 'detection_model/SKU110K_CVPR19/model_weights/iou_resnet50_csv_03.h5')

    detection_csv_results_file_path = main(image_folder_path, detection_save_folder, model_wights_path)

    return detection_csv_results_file_path

# Needs to be updated in relation to the changes in predict.py argparser
def run_detection_from_bash(image_folder_path: str, detection_save_folder: str):
    """Master script to run SKU110K retail object detection on all images in image_folder_path
       It creates a new bash session and runs SKU's predict.py from bash with necessary flag parameters.
        
    Arguments:
        image_folder_path {str} -- path to the image folder 
    """

    model_wights_path = os.path.join(os.getcwd(), 'detection_model/SKU110K_CVPR19/model_weights/iou_resnet50_csv_03.h5')
    
    NEW_PYTHONPATH = os.path.join(os.getcwd(), 'detection_model/SKU110K_CVPR19')

    run_predict_command = 'export PYTHONPATH={} && python -u {}/object_detector_retinanet/keras_retinanet/bin/predict.py'.format(NEW_PYTHONPATH, NEW_PYTHONPATH)

    run_predict_flags = " --model " + model_wights_path + " --base_dir " + image_folder_path + " --save-path " + detection_save_folder + " csv"

    print("running \"$ {}\" ...".format(run_predict_command + run_predict_flags))
    os.system(run_predict_command + run_predict_flags)
