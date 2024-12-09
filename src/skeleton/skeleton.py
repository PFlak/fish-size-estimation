import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize, medial_axis
from skimage import morphology
import networkx as nx
from scipy.ndimage import convolve
import sknw

from mrcnn.visualize import apply_mask

ROOT_DIR = '../'
sys.path.append(ROOT_DIR)
from src.segmentation.segmentation import Segmentation



class Skeleton:
    def __init__(self):
        pass

    def make_skeleton(self, mask_img):
        binary_mask = (mask_img > 0).astype(np.uint8)

        return medial_axis(binary_mask, return_distance=True)

    def show_skeleton(self, mask_img):
        # Convert mask to binary: values of 0 and 1
        self.h, self.w = mask_img.shape
        binary_mask = (mask_img > 0).astype(np.uint8)

        # Apply skeletonize on the binary mask
        skeleton, dist = medial_axis(binary_mask, return_distance=True)

        # Convert skeleton to uint8 (needed for display and dilation)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        # Increase thickness using dilation
        kernel = np.ones((3, 3), np.uint8)
        # thick_skeleton = cv2.dilate(skeleton_uint8, kernel, iterations=1)

        # Plot both mask and skeleton
        plt.figure(figsize=(10, 5))

        # Show the mask
        plt.imshow(mask_img, cmap='gray', alpha=0.6)

        # Overlay the skeleton (red color)
        plt.imshow(skeleton, cmap='Reds', alpha=0.7)

        # Display the plot
        plt.title("Mask and Skeleton")
        plt.axis('off')
        plt.show()

    def create_mask(self, image, mask):
        masked_image = np.zeros_like(image)
        masked_image = apply_mask(masked_image, mask, (1.,1.,1.), alpha=1)
        masked_image = masked_image[:,:,0]
        return masked_image

    def create_graph(self, skeleton, dist):
        graph = sknw.build_sknw(skeleton)
        node_points = nx.get_node_attributes(graph, 'pts')
        attrs = {}
        for i in range(len(node_points)):
            neighbors = list(graph.neighbors(i))
            if len(neighbors) > 1:
                attrs[i] = {'distance': 0}
                continue
            pts = node_points[i][0]
            local_dist = dist[pts[0]][pts[1]]
            attrs[i] = {'distance': local_dist}
        nx.set_node_attributes(graph, attrs)
        edges = graph.edges(data=True)
        attrs = {}
        for edge in edges:
            node_1, node_2, data = edge
            weight = data['weight']
            distances = nx.get_node_attributes(graph, 'distance')
            dst_1, dst_2 = distances[node_1], distances[node_2]
            weight = dst_1 + dst_2 + weight
            attrs[(node_1, node_2)] = {'distance': weight}
        nx.set_edge_attributes(graph, attrs)
        return graph

    def longest_path(self, graph, start, visited = None, current_length = 0):
        if visited is None:
            visited = set()
        visited.add(start)
        max_length = current_length
        neighbors = nx.neighbors(graph, start)


        for neighbor in neighbors:
            if neighbor not in visited:
                distance = graph.get_edge_data(neighbor, start)['distance']
                max_length = max(max_length, self.longest_path(graph, neighbor, visited, current_length + distance))

        return max_length

    def show_graph(self, graph, img):
        plt.imshow(img, cmap='gray')

        # draw edges by pts
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')

        # draw node by o
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')
        plt.show()



if __name__ == '__main__':
    model_dir = os.path.join(ROOT_DIR, 'ai_model', 'segmentation')
    model_name = os.listdir(model_dir)[0]
    seg = Segmentation(model_dir, model_name)
    seg.load_model()

    img_path = os.path.join(ROOT_DIR, 'data_in', os.listdir(os.path.join(ROOT_DIR, 'data_in'))[0])
    img = cv2.imread(img_path)
    result = seg.predict(img_path)
    result = result[0]
    roi = result['rois'][0]
    class_id = result['class_ids'][0]
    score = result['scores'][0]
    mask = result['masks'][:,:,3]

    skeleton = Skeleton()
    mask_img = skeleton.create_mask(img, mask)
    sk, dist = skeleton.make_skeleton(mask_img)
    graph = skeleton.create_graph(sk, dist)
    longest = 0
    nodes = list(nx.nodes(graph))
    for n in nodes:
        longest = max(longest, skeleton.longest_path(graph, n))
    print(longest)

    # print(graph.nodes)
    # distances = nx.get_edge_attributes(graph, 'distance')
    # print(distances)
    # skeleton.show_skeleton(mask_img)
    # skeleton.show_graph(graph)
    # longest_path, longest_path_length = skeleton.find_longest_vector(graph)
    # print(longest_path, longest_path_length)

    # graph = skeleton.remove_negative_cycles(graph)
    # skeleton.show_graph(graph)