import os
import csv
import sys
from PIL import Image
from typing import *

# sys.path.append(os.path.join(os.getcwd(), 'detection_model/SKU110K_CVPR19'))
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

    # run_predict_command = 'export PYTHONPATH=$(pwd) && python -u object_detector_retinanet/keras_retinanet/bin/predict.py'
    run_predict_command = 'export PYTHONPATH={} && python -u {}/object_detector_retinanet/keras_retinanet/bin/predict.py'.format(NEW_PYTHONPATH, NEW_PYTHONPATH)

    run_predict_flags = " --model " + model_wights_path + " --base_dir " + image_folder_path + " --save-path " + detection_save_folder + " csv"


    print("running \"$ {}\" ...".format(run_predict_command + run_predict_flags))
    os.system(run_predict_command + run_predict_flags)



def create_csv(image_folder_path: str, formats: list) -> str:
    """Create a CSV file to be used as input argument for the SKU110K product detector
       This function is not necessary anymore since SKU prediciton was updated not to need the csv file
    
    Arguments:
        image_folder_path {str} -- path to folder containing the images.
        formats {list[str]} -- List of strings containing the allowed image format extensions (e.g 'jpeg') 
    
    Returns:
        str -- generated csv_file_path
    """
    
    image_folder_basename = os.path.basename(os.path.normpath(image_folder_path))

    csv_file_path = os.path.join(image_folder_path, image_folder_basename + "_annotationlist_for_SKU.csv")
    
    #Create csv file and file_writer
    csv_file = open(csv_file_path, mode='w')
    file_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONE)

    # Write image paths info etc. to csv in the format needed for the SKU110K detection.
    image_info = [(image_name, Image.open(os.path.join(image_folder_path,image_name)).size[0], Image.open(os.path.join(image_folder_path,image_name)).size[1] ) for image_name in os.listdir(image_folder_path) if image_name.split('.')[-1] in formats]
    # [file_writer.writerow(['img_file','x1','y1','x2','y2','class_name', 'image_width', 'image_height'])]
    [file_writer.writerow([image[0],0,1,2,3,'object',image[1],image[2]]) for image in image_info]

    return csv_file_path
