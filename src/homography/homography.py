import os

import cv2
import tensorflow as tf
import numpy as np
from datetime import datetime

ROOT_DIR = '../'

class Homography:
    def __init__(self, model=None):
        if model is None:
            model_path = os.path.join(ROOT_DIR, 'ai_model', 'tray_model')
            self.model_path = model_path
        else:
            self.model_path = model

        self.img_width = 256
        self.img_height = 256

        self.out_dir = os.path.join(ROOT_DIR, 'data_out')

        self.BORDER = 0.05
        self.MEAN_RATIO = 0.002

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path

        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(e)

    def load_model_v1(self, model_dir):
        """Load a SavedModel from the specified directory."""
        # Create a new TensorFlow session and load the model
        sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)

        # Get input and output tensor names from the graph
        graph = sess.graph
        input_tensor = graph.get_tensor_by_name(
            'input_tensor_name:0')  # Replace 'input_tensor_name' with the actual input name
        output_tensor = graph.get_tensor_by_name(
            'output_tensor_name:0')  # Replace 'output_tensor_name' with the actual output name

        return sess, input_tensor, output_tensor

    def predict_v1(self, sess, input_tensor, output_tensor, img, target_size=(224, 224)):
        """Preprocess the input image using OpenCV and run prediction."""

        # Convert image to RGB (if needed)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to the model's input size
        img_resized = cv2.resize(img, target_size)

        # Normalize the image (assuming the model was trained on images normalized to [0, 1])
        img_resized = img_resized / 255.0

        # Expand dimensions to match the model's input shape (e.g., [1, 224, 224, 3])
        img_expanded = np.expand_dims(img_resized, axis=0)

        # Run the session to get predictions
        predictions = sess.run(output_tensor, feed_dict={input_tensor: img_expanded})

        return predictions

    def predict(self, img):
        img = cv2.imread(img)
        (h, w) = img.shape[:2]
        input_img = cv2.resize(img, (self.img_width, self.img_height))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = cv2.Laplacian(input_img, -1)
        input_img = input_img.astype(np.float32)
        input_img /= 255.
        input_img = list(map(lambda x: list(map(lambda y: [y], x)), input_img))
        # input_tensor = tf.convert_to_tensor(input_img, dtype=tf.float32)
        # print(input_tensor.shape)
        # infer = model.signatures['serving_default'](input_tensor)
        # print(infer)
        out = list(self.model.predict(np.array([input_img])))[0]
        cvt_point = []
        i = 0
        while i < len(out):
            cvt_point.append((out[i] * w, out[i + 1] * h))
            i += 2
        cvt_point = np.array(cvt_point, dtype=np.int32)
        return cvt_point

    def sort_points(self, points):

        points = sorted(points, key=lambda p: [p[1], p[0]])  # Sort by y first, then by x

        # After sorting, we can categorize the points
        top_left = points[0]
        top_right = points[1]
        bottom_left = points[2]
        bottom_right = points[3]

        # Ensure that among the top two, the one with the smaller x is the top-left
        if top_right[0] < top_left[0]:
            top_left, top_right = top_right, top_left

        # Ensure that among the bottom two, the one with the smaller x is the bottom-left
        if bottom_right[0] < bottom_left[0]:
            bottom_left, bottom_right = bottom_right, bottom_left

        return [bottom_left, bottom_right, top_right, top_left]

    def visualize(self, img, points):
        img = cv2.imread(img)

        h, w = img.shape[:2]
        h, w = h * 2, w * 2

        b_w = w * self.BORDER

        width = w - 2 * b_w
        height = width * self.MEAN_RATIO

        b_h = (h - height / 2)

        n_pt_c = (b_w, b_h)
        n_pt_d = (w - b_w, b_h)
        n_pt_a = (b_w, h - b_h)
        n_pt_b = (w - b_w, h - b_h)

        pt_a = points[0]
        pt_b = points[1]
        pt_d = points[2]
        pt_c = points[3]

        pts_src = np.array([pt_c, pt_d, pt_b,pt_a])
        pts_dst = np.array([n_pt_a, n_pt_b, n_pt_d,n_pt_c])
        H, status = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC,5.0)

        pts_src = self.sort_points(pts_src)
        # im_dst = cv2.polylines(img, np.int32([pts_src]), 1, (0, 0, 255), 10)
        im_dst = cv2.warpPerspective(img, H, (w, h))
        return im_dst

    def save_image(self, img, out_dir):
        now = datetime.now()
        timestamp = now.strftime("%d_%m_%y_%M_%S")
        cv2.imwrite(os.path.join(out_dir, 'homography' + timestamp + '.jpg'), img)



if __name__ == '__main__':
    homography = Homography()
    homography.load_model()

    img = os.path.join(ROOT_DIR, 'data_in', os.listdir(os.path.join(ROOT_DIR, 'data_in'))[0])
    print(img)

    pts = homography.predict(img)
    pts = homography.sort_points(pts)
    img = homography.visualize(img, pts)
    homography.save_image(img)