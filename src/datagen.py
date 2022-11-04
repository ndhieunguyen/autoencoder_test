from tensorflow.keras.utils import Sequence
import math
from tensorflow.image import flip_left_right
from src.models.text_encoder import get_tokens
import numpy as np
import cv2

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
        # images = flip_left_right(images)
        images = images / 255.
        # images = tf.clip_by_value(images, 0, 1)
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


if __name__ == '__main__':
    import cv2
    import pandas as pd

    data = pd.read_csv('D:\\emb\\data\\fashion_iq\\dress\\dataframe\\train_3.csv')
    data['input_image'] = data['input_image'].str.cat(['.png']*len(data))
    data['input_image'] = pd.Series(['D:\\emb\\data\\fashion_iq\\dress\\images\\']*len(data)).str.cat(data['input_image'])
    
    data['output_image'] = data['output_image'].str.cat(['.png']*len(data))
    data['output_image'] = pd.Series(['D:\\emb\\data\\fashion_iq\\dress\\images\\']*len(data)).str.cat(data['output_image'])
    
    datagen = DataGenerator(list(data['input_image']), 
                            list(data['caption']),
                            list(data['output_image']),
                            batch_size=16,
                            is_training=True)

    n = np.random.randint(16)
    a = datagen[n]

    X = a[0][0]
    y = a[0][1]

    cv2.imshow('input', X[0])
    cv2.imshow('output', y[0])
    print(X[1])
    cv2.waitKey(0)