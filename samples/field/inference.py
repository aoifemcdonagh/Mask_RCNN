import os
import sys
import random
import math
import re
import time
import numpy as np
import skimage.io
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.field import field

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


if __name__ == '__main__':
    image_path = str(sys.argv[1])  # path to image to perform inference

    config = field.FieldConfig()
    FIELD_DIR = os.path.join(ROOT_DIR, "datasets/field")

    # Load validation dataset
    dataset = field.FieldDataset()
    dataset.load_field(FIELD_DIR, "val")

    # Must call before using the dataset
    dataset.prepare()

    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    TEST_MODE = "inference"

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    weights_path = model.find_last()

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # Load image
    image = skimage.io.imread(image_path)

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    #ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], show_bbox=False, title="Predictions")
