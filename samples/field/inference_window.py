# Testing inference on windows of larger input image

import os
import sys
import numpy as np
import skimage.io
from skimage.measure import find_contours
import tensorflow as tf
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

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


def split_image(window_len=128, slide_len=64):
    """
    Function which splits up a large image into smaller chunks for inference
    :param window_len: length of window for performing inference
    :param slide_len: length to slide window for each inference pass
    :return:
    """


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
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

    # ndarray of shape (n,2) where n is number of vertices in contour
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
        mask = masks[:, :, i]
        polygons[i] = get_vertices(mask)

    return polygons


def get_UTM_coords(polygon, origin=(0, 0)):
    """
    Function which returns the polygon vertices as UTM coordinates
    :param polygon:
    :param origin: UTM coordinate at bottom left of image
    :return:
    """

    utm_coords = [None] * len(polygon)  # make list of Nones the same length as polygon

    # for vertice in polygon:
    # convert to UTM using origin values


def display_polygons(image, polygons, title="Predictions", figsize=(16, 16)):
    """
    Function which
    :param image:
    :param polygons:
    :return:
    """

    N = len(polygons)  # number of instances
    _, ax = plt.subplots(1, figsize=figsize)
    auto_show = True

    colors = visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    plot_image = image.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]

        verts = polygons[i]  # get ndarray of polygon vertices

        p = Polygon(verts, facecolor="none", edgecolor=color)
        ax.add_patch(p)

    ax.imshow(plot_image.astype(np.uint8))
    if auto_show:
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', type=str, required=True, help="path to image to perform inference on")
    parser.add_argument('-m', '--weights', type=str, help="path to model weights")
    parser.add_argument('-w', '--window-width', type=int, help="window width to use for inference")
    parser.add_argument('-s', '--slide-width', type=int, help="slide width for inference window")
    args = parser.parse_args()

    image_path = args.image  # path to image to perform inference

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

    if args.weights is not None:  # use path to model weights if given
        weights_path = args.weights
    else:  # else find most recent trained model in logs
        weights_path = model.find_last()

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # Load image
    image = skimage.io.imread(image_path, plugin='pil')

    # may need to swap these around?
    image_x = image.shape[0]
    image_y = image.shape[1]

    window_width = 256  # width of window to use for inference pass
    slide_width = 256  # change to number < 256 to overlap inference passes

    # initialise counters
    i = 0  # width counters
    j = window_width -1

    m = 0  # height counters
    n = window_width -1

    while j < image_x:  # iterate down height
        while n < image_y:  # iterate across width
            window = image[i:j, m:n, :]  # get image window

            # Run object detection
            results = model.detect([image], verbose=1)

            # Display results
            # ax = get_ax(1)
            r = results[0]
            masks = r['masks']

            extracted_polygons = get_polygons(masks=masks)

            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        dataset.class_names, show_bbox=False, show_mask=False, title="Predictions")

            # update counters
            m += slide_width
            n += slide_width

        # update counters
        i += slide_width
        j += slide_width

        # re initialise counters
        m = 0  # width counters
        n = window_width - 1


    display_polygons(image=image, polygons=extracted_polygons)

