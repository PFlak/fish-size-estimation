
import os

from src.homography.homography import Homography

import math
import os
from scipy.spatial import distance

import cv2
import networkx as nx

from segmentation import segmentation
from skeleton import skeleton

DATA_IN_DIR = './data_in'
DATA_OUT_DIR = './out'
MRCNN_DIR = '../'
TRAY_MODEL_DIR = '../ai_model/tray_model/out_model.h5'
SEG_MODEL_DIR = '../ai_model/segmentation'

HOMO_OUT_DIR = './homo_out'
MODEL_DIR = '../ai_model/segmentation'
MODEL_NAME = os.listdir(MODEL_DIR)[-1]

MEAN_RATIO = 0.002
MEAN_TRACE_WIDTH = 56

BORDER = 0.08


def analize_img(img):
    result = seg.predict_from_image(img)[0]

    for i in range(len(result['class_ids'])):
        if result['class_ids'][i] != 3:
            continue

        roi = result['rois'][i]
        class_id = result['class_ids'][i]
        score = result['scores'][i]
        mask = result['masks'][:, :, i]

        ske = skeleton.Skeleton()
        mask_img = ske.create_mask(img, mask)
        sk, dist = ske.make_skeleton(mask_img)
        graph = ske.create_graph(sk, dist)
        ske.show_graph(graph, img)
        longest = 0
        nodes = list(nx.nodes(graph))
        for n in nodes:
            longest = max(longest, ske.longest_path(graph, n))

        h, w = img.shape[:2]
        h, w = h * 2, w * 2

        b_w = w * BORDER

        width = w - 2 * b_w
        height = width * MEAN_TRACE_WIDTH

        b_h = (h - height / 2)

        n_pt_c = (b_w, b_h)
        n_pt_d = (w - b_w, b_h)
        n_pt_a = (b_w, h - b_h)
        n_pt_b = (w - b_w, h - b_h)

        trace_w = distance.euclidean(n_pt_c, n_pt_d)

        fish_px_size = longest

        ratio = fish_px_size / trace_w

        estimated_size = ratio * MEAN_TRACE_WIDTH

        print('Fish {}: {}cm'.format(i, estimated_size))


if __name__ == '__main__':
    homography = Homography(TRAY_MODEL_DIR)
    homography.load_model()

    seg = segmentation.Segmentation(MODEL_DIR, MODEL_NAME)
    seg.load_model()

    for img_name in os.listdir(os.path.join('./', 'data_in')):
        img = os.path.join('./', 'data_in', img_name)
        print(img)

        pts = homography.predict(img)
        pts = homography.sort_points(pts)
        img = homography.visualize(img, pts)
        homography.save_image(img, DATA_OUT_DIR)

        analize_img(img)
