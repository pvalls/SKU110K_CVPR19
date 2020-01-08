import os
import csv
from PIL import Image
from typing import *

def run_detection():

    # Define folders with shelf images to run prediction on
    image_folder_paths = [os.path.join("DAMM-dataset", 'CARREFOUR_warped'), 
                          os.path.join("DAMM-dataset", 'CONDIS_warped'), 
                          os.path.join("DAMM-dataset", 'SORLI_warped')]

    weight_file_path = "/Users/polvalls/repos/ai-damm/apps/detection-model/SKU110K_CVPR19/snapshot/Fri_Jan__3_10.59.16_2020/iou_resnet50_csv_01.h5"
    
    csv_file_paths = []

    # Create csv annotation files for each folder and store path to csv_file_paths
    for image_folder_path in image_folder_paths:
        csv_file_paths.append(create_csv(image_folder_path, ['jpeg']))


    print("running \"$ export PYTHONPATH=$(pwd)...\" ")
    os.system("export PYTHONPATH=$(pwd)")


    for csv_file_path in csv_file_paths:
        
        run_predict_command = 'python -u object_detector_retinanet/keras_retinanet/bin/predict.py'

        run_predict_flags = " csv --annotations " + csv_file_path + " " + weight_file_path
        
        
        print("running \"$ {}\" ...".format(run_predict_command + run_predict_flags))
        os.system(run_predict_command + run_predict_flags)


def create_csv(image_folder_path: str, formats: list):
    """Create a CSV file to be used as input argument for the SKU110K product detector
    
    Arguments:
        image_folder_path {str} -- path to folder containing the images.
        formats {list[str]} -- List of strings containing the allowed image format extensions (e.g 'jpeg') 
    """
    
    image_folder_basename = os.path.basename(os.path.normpath(image_folder_path))

    csv_file_path = os.path.join(image_folder_path, image_folder_basename + "_annotations.csv")
    
    #Create csv file and file_writer
    csv_file = open(csv_file_path, mode='w')
    file_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONE)

    # Write image paths info etc. to csv in the format needed for the SKU110K detection.
    image_info = [(image_name, Image.open(os.path.join(image_folder_path,image_name)).size[0], Image.open(os.path.join(image_folder_path,image_name)).size[1] ) for image_name in os.listdir(image_folder_path) if image_name.split('.')[-1] in formats]
    # [file_writer.writerow(['img_file','x1','y1','x2','y2','class_name', 'image_width', 'image_height'])]
    [file_writer.writerow([image[0],0,1,2,3,'object',image[1],image[2]]) for image in image_info]

    return csv_file_path

if __name__ == "__main__":

    run_detection()

