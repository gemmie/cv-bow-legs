import logging
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from skimage import img_as_ubyte
from skimage import morphology, color
from skimage import transform, io

BK_MODEL_PATH = "models/model_bk.json"
BK_MODEL_WEIGHTS_PATH = "models/trained_model.hdf5"

graph = tf.get_default_graph()


class BoneOpeningModel:
    IMAGE_SHAPE = (512, 256)

    def __init__(self):
        with open(BK_MODEL_PATH, 'r') as raw_model:
            loaded_model = model_from_json(raw_model.read())

        loaded_model.load_weights(BK_MODEL_WEIGHTS_PATH)  # load weights into new model
        logging.info("Pre-trained model loaded from memory, model's weights set")

        self.model = loaded_model

    def predict(self, image_path: str) -> Tuple[str, str]:
        """
        Use pre-trained model to predict outline (morphology opening) of the bone from given image
        :param image_path: Path to image used for prediction
        :return: Paths to predicted bone morphology opening and masked image (input image + prediction)
        """
        image = io.imread(image_path, as_gray=True)
        transformed_image = self._transform_image(image)
        print(transformed_image.shape)
        image = transform.resize(image, self.IMAGE_SHAPE, mode='constant')

        with graph.as_default():
            print(*image.shape)
            prediction = self.model.predict(transformed_image)[..., 0].reshape(*self.IMAGE_SHAPE)

        prediction = prediction > 0.5
        predicted_opening = morphology.opening(img_as_ubyte(prediction))
        predicted_image_path = self._create_save_path(image_path, "prediction")
        io.imsave(predicted_image_path, predicted_opening)

        predicted_opening_mask = predicted_opening > 0.5
        masked_image = self.mask_image(image, predicted_opening_mask, 0.5)
        masked_image_path = self._create_save_path(image_path, "mask")
        io.imsave(masked_image_path, img_as_ubyte(masked_image))

        return predicted_image_path, masked_image_path

    def mask_image(self, image: np.ndarray, mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Create a new image with applied mask.
        Outlines of predicted mask marked with red color, bone prediction marked with blue color.
        :param image: input image
        :param mask: Mask to be applied on image
        :param alpha: Transparency of the mask
        :return: Image with mask mask applied
        """
        rows, cols = self.IMAGE_SHAPE #image.shape
        color_mask = np.zeros((rows, cols, 3))

        boundary = morphology.dilation(mask, morphology.disk(2)) ^ mask

        color_mask[mask == 1] = [0, 0, 1]
        color_mask[boundary == 1] = [1, 0, 0]
        img_color = np.dstack((image, image, image))

        img_hsv = color.rgb2hsv(img_color)
        color_mask_hsv = color.rgb2hsv(color_mask)

        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        img_masked = color.hsv2rgb(img_hsv)
        return img_masked

    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for model prediction, add additional 2 dimensions and performs mean ans std on given image
        :param image: Input image
        :return: Image after all transformation
        """
        tr_image = transform.resize(image, self.IMAGE_SHAPE, mode='constant')
        tr_image = np.expand_dims(tr_image, -1)
        tr_image = tr_image[None, ...]

        tr_image -= tr_image.mean()
        tr_image /= tr_image.std()

        return tr_image

    def _create_save_path(self, image_path: str, suffix: str):
        directory, extension = image_path.rsplit(".", 1)
        return f"{directory}{suffix}.{extension}"
