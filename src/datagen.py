from tensorflow.keras.utils import Sequence
import math
from src.models.text_encoder import get_tokens
import numpy as np
import cv2
from tensorflow import image

class DataGenerator(Sequence):
    def __init__(self, input_paths, captions, output_paths, batch_size, is_training=True):
        self.input_paths = input_paths
        self.captions = captions
        self.output_paths = output_paths
        self.batch_size = batch_size
        self.is_training = is_training
        
    def __len__(self):
        return math.ceil(len(self.input_paths) / self.batch_size)
    
    def _transform_images(self, images):
        images = image.flip_left_right(images)
        images = image.random_saturation(images, 0.6, 1.4)
        images = image.random_brightness(images, 0.2)
        images = images / 255.
        return images
    
    def __getitem__(self, idx):
        batch_input_paths = self.input_paths[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_output_paths = self.output_paths[idx*self.batch_size : (idx+1)*self.batch_size]

        batch_input_images = np.array([cv2.imread(image_path) for image_path in batch_input_paths], dtype='float32')
        batch_output_images = np.array([cv2.imread(image_path) for image_path in batch_output_paths], dtype='float32')

        batch_captions = self.captions[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_token, batch_mask = get_tokens(batch_captions)
        
        if self.is_training:
            return (self._transform_images(batch_input_images), batch_token, batch_mask, self._transform_images(batch_output_images)), self._transform_images(batch_output_images)
        else:
            return (batch_input_images/255., batch_token, batch_mask, batch_output_images/255.), batch_output_images/255.