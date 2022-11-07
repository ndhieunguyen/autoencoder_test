from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from src.models.text_encoder import get_text_encoder_model_tf

def create_model():
    input_image = layers.Input(shape=(224, 224, 3), name='input_image')
    text_encoder = get_text_encoder_model_tf('distilroberta-base', 'distilroberta-base')
    output_image = layers.Input(shape=(224, 224, 3), name='output_image')

    image_encoder = tf.keras.Sequential([
                layers.Conv2D(48, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), padding='same'),
                layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), padding='same'),
                layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), padding='same'),
                layers.Conv2D(32, (1, 1), activation='relu', padding='same'),
            ], name='encoder')(input_image)

    expand_text_encoder = layers.Dense(28*28*32, name='expand_text_encoder')(text_encoder.output)
    reshape_text_encoder = layers.Reshape((28, 28, 32), name='reshape_text_encoder')(expand_text_encoder)
    concat = layers.concatenate([image_encoder, reshape_text_encoder], axis=-1)

    image_decoder = tf.keras.Sequential([
            layers.Conv2D(192, (1, 1), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(48, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),
        ], name='decoder')(concat)

    loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(output_image, image_decoder)

    return Model(inputs={'input_image': input_image, 
                        'input_text_id': text_encoder.input[0], 
                        'input_text_mask': text_encoder.input[1], 
                        'output_image': output_image}, 
                outputs={'loss': loss, 
                        'image_encoder': image_encoder, 
                        'image_decoder':image_decoder}) 