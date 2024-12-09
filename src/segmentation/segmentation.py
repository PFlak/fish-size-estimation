from datetime import datetime
import os
import sys

import cv2

ROOT_DIR = '../'
MRCNN_DIR = os.path.abspath('../')
sys.path.append(MRCNN_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from fishSizeEstimation import fishSizeEstimation

config = fishSizeEstimation.FishSizeEstimationConfig()

class Segmentation:

    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    def __init__(self, model_dir = None, model_name = None):
        self.config = Segmentation.InferenceConfig()
        self.config.display()
        self.model_dir = model_dir
        self.model_name = model_name

    def load_model(self, model_dir=None, model_name=None):
        if model_dir is None:
            model_dir = self.model_dir
        if model_name is None:
            model_name = self.model_name

        self.model = modellib.MaskRCNN(mode='inference', model_dir=self.model_dir, config=self.config)
        self.model.load_weights(filepath=os.path.join(model_dir, model_name), by_name=True)

    def predict(self, image):
        img = cv2.imread(image)
        result = self.model.detect([img], verbose=1)
        return result

    def predict_from_image(self, img):
        result = self.model.detect([img], verbose=1)
        return result

    def visualize(self, image, result, class_names):
        image = cv2.imread(image)
        visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'], class_names=class_names,title='Prediction')

    def save_image(self, img, result, class_names):
        now = datetime.now()
        timestamp = now.strftime("%d_%m_%y_%M:%S")

        img = cv2.imread(img)
        save_path = os.path.join(ROOT_DIR, 'data_out', 'segmentation_' + timestamp + '.png')
        visualize.display_instances(img, result['rois'], result['masks'], result['class_ids'],
                                    class_names=class_names, title='Prediction', save=True, save_path=save_path)




if __name__ == '__main__':
    model_dir = os.path.join(ROOT_DIR, 'ai_model', 'segmentation')
    model_name = os.listdir(model_dir)[0]
    seg = Segmentation(model_dir, model_name)
    seg.load_model()

    img = os.path.join(ROOT_DIR, 'data_in', os.listdir(os.path.join(ROOT_DIR, 'data_in'))[0])
    results = seg.predict(img)
    results = results[0]
    seg.visualize(img, results, class_names=['BG', 'pagrus_pagrus'])
    seg.save_image(img, results, class_names=['BG', 'pagrus_pagrus'])