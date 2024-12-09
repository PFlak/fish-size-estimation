"""
Mask R-CNN
Train on the toy FishSizeEstimation dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 fishSizeEstimation.py train --dataset=/path/to/fishSizeEstimation/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 fishSizeEstimation.py train --dataset=/path/to/fishSizeEstimation/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 fishSizeEstimation.py train --dataset=/path/to/fishSizeEstimation/dataset --weights=imagenet

    # Apply color splash to an image
    python3 fishSizeEstimation.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 fishSizeEstimation.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class FishSizeEstimationConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "FishSizeEstimation"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 31  # Background + Pagrus pagrus

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class FishSizeEstimationDataset(utils.Dataset):


    def load_fishsizeestimation(self, dataset_dir, subset):
        """Load a subset of the fishSizeEstimation dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("fse", 1, "Chelidonichthys lastoviza")
        self.add_class("fse", 2, "Diplodus vulgaris")
        self.add_class("fse", 3, "Pagrus pagrus")
        self.add_class("fse", 4, "Scomber japonicus")
        self.add_class("fse", 5, "Scorpaena notata")
        self.add_class("fse", 6, "Scorpaena porcus")
        self.add_class("fse", 7, "Serranus scriba")
        self.add_class("fse", 8, "Species 10")
        self.add_class("fse", 9, "Species 11")
        self.add_class("fse", 10, "Species 13")
        self.add_class("fse", 11, "Species 15")
        self.add_class("fse", 12, "Species 18")
        self.add_class("fse", 13, "Species 20")
        self.add_class("fse", 14, "Species 22")
        self.add_class("fse", 15, "Species 3")
        self.add_class("fse", 16, "Species 32")
        self.add_class("fse", 17, "Species 36")
        self.add_class("fse", 18, "Species 38")
        self.add_class("fse", 19, "Species 4")
        self.add_class("fse", 20, "Species 46")
        self.add_class("fse", 21, "Species 6")
        self.add_class("fse", 22, "Species 68")
        self.add_class("fse", 23, "Species 7")
        self.add_class("fse", 24, "Species 81")
        self.add_class("fse", 25, "Species 84")
        self.add_class("fse", 26, "Species 85")
        self.add_class("fse", 27, "Species 86")
        self.add_class("fse", 28, "Species 87")
        self.add_class("fse", 29, "Sphyraena sphyraena")
        self.add_class("fse", 30, "Spicara smaris")
        self.add_class("fse", 31, "Trachurus spp")


        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = annotations["files"]

        for annotation in annotations:
            path = os.path.join(dataset_dir, annotation["filename"])
            height = annotation["height"]
            width = annotation["width"]
            regions = annotation["regions"]

            self.add_image(
                "fse",
                image_id=annotation["filename"],
                path=path,
                width=width,
                height=height,
                regions=regions
            )


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a fishSizeEstimation dataset image, delegate to parent class.
        # image_info = list(filter(lambda x: x['id'] == image_id, self.image_info))[0]
        image_info = self.image_info[image_id]
        # print(image_info)
        if image_info["source"] != "fse":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = image_info
        mask = np.zeros([info["height"], info["width"], len(info["regions"])],
                        dtype=np.uint8)
        classes = []
        for i, p in enumerate(info["regions"]):
            classes.append(p["class_id"])
            # Get indexes of pixels inside the polygon and set them to 1
            y = [pt[1] for pt in p["polygons"]]
            x = [pt[0] for pt in p["polygons"]]
            rr, cc = skimage.draw.polygon(y, x)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(classes).astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "fse":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = FishSizeEstimationDataset()
    dataset_train.load_fishsizeestimation(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FishSizeEstimationDataset()
    dataset_val.load_fishsizeestimation(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=65,
                layers='heads')
    
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect pagrus pagrus.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/fishSizeEstimation/dataset/",
                        help='Directory of the FishSizeEstimation dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = FishSizeEstimationConfig()
    else:
        class InferenceConfig(FishSizeEstimationConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
