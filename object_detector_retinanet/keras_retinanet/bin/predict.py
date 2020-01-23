#!/usr/bin/env python
import argparse
import os
import sys

import keras
import numpy
import tensorflow as tf

# Equivalent to doing export PYTHONPATH=$(pwd) from bash before running predict.py
sys.path.append(os.path.join(os.getcwd(), 'detection_model/SKU110K_CVPR19'))

from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.preprocessing.csv_generator import FOLDERGenerator
from object_detector_retinanet.keras_retinanet.utils.predict_iou import predict
from object_detector_retinanet.keras_retinanet.utils.keras_version import check_keras_version
from object_detector_retinanet.utils import annotation_path, root_dir


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args, image_folder_path):
    """ Create generators for evaluation.
    """
    if args.dataset_type == 'csv':
        validation_generator = FOLDERGenerator(
            image_folder_path,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            base_dir=args.base_dir
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = False

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    
    parser.add_argument('--hard_score_rate', help='')

    parser.add_argument('--model', help='Path to RetinaNet model.')
    parser.add_argument('--base_dir', help='Path to base dir for CSV file.')
    parser.add_argument('--convert-model',
                        help='Convert the model to an inference model (ie. the input is a training model).', type=int,
                        default=1)

    parser.add_argument('--backbone', help='The backbone of the model.')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).')
    parser.add_argument('--iou-threshold', help='IoU Threshold to count for a positive detection (defaults to 0.5).')
    parser.add_argument('--save-path', help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.')
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than image_max_side.')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('--annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('--classes', help='Path to a CSV file containing class label mapping.',
                            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_mappings.csv'))
                            
    return parser.parse_args(args)


def main(image_folder_path: str, detection_save_folder: str, model_wights_path: str,
                       classes=None,
                       hard_score_rate = 0.5,
                       backbone = 'resnet50',
                       gpu = None,
                       score_threshold = 0.1,
                       iou_threshold = 0.5,
                       image_min_side = 800,
                       image_max_side = 1333,
                       args = None):

    # parse arguments
    if args is None:
        # args = sys.argv[1:]
        args = ["--model", model_wights_path, 'csv']

    args = parse_args(args)

    # fill args with function parameters initialized with defaults
    # if args.annotations is None:
    #     args.annotations = csv_file_path

    if args.hard_score_rate is None:
        args.hard_score_rate = hard_score_rate
        # hard_score_rate = float(args.hard_score_rate.lower())

    if args.model is None:
        args.model = model_wights_path

    if args.base_dir is None:
        args.base_dir = image_folder_path

    if args.backbone is None:
        args.backbone = backbone

    if args.gpu is None:
        args.gpu = gpu

    if args.score_threshold is None:
        args.score_threshold = float(score_threshold)

    if args.iou_threshold is None:
        args.iou_threshold = float(iou_threshold)

    if args.save_path is None:
        args.save_path = detection_save_folder

    if args.image_max_side is None:
        args.image_max_side = int(image_max_side)

    if args.image_min_side is None:
        args.image_min_side = int(image_min_side)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    use_cpu = False

    if args.gpu:
        gpu_num = args.gpu
    else:
        gpu_num = str(0)

    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(666)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args, args.base_dir)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(os.path.join(root_dir(), args.model), backbone_name=args.backbone, convert=args.convert_model, nms=False)

    # start prediction
    predict(
        generator,
        model,
        score_threshold=args.score_threshold,
        # save_path=os.path.join(root_dir(), 'res_images_iou'),
        save_path = os.path.join(generator.base_dir, "detection_result_images"),
        hard_score_rate=hard_score_rate
    )


if __name__ == '__main__':

    main(str(), str(), str(),str(), args = sys.argv[1:])
