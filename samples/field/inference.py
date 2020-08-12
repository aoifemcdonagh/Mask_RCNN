import os
import sys
import random
import math
import re
import time
import numpy as np
import skimage.io
from skimage.measure import find_contours
import tensorflow as tf
import matplotlib
from matplotlib.patches import Polygon
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


def get_vertices(mask):
    """
    Function which takes a single boolean mask as input
    returns polygon vertices
    :param mask: boolean mask
    :return: array of vertices
    """

    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1

    return verts


def get_polygons(masks):
    """
    Function which takes a series of boolean masks as input and returns polygon vertices
    :param masks: tuple of boolean masks
    :return: list of polygon vertices corresponding to masks
    """

    num_masks = masks.shape[2]
    polygons = [None] * num_masks

    for i in range(num_masks):
        mask = masks[:,:,i]
        polygons[i] = get_vertices(mask)

    return polygons


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
    image = skimage.io.imread(image_path, plugin='pil')

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    #ax = get_ax(1)
    r = results[0]
    masks = r['masks']

    polygon_vertices = get_polygons(masks=masks)

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, show_bbox=False, title="Predictions")

    