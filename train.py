from src.models.autoencoder import create_model
from src.datagen import DataGenerator
import tensorflow as tf
import pandas as pd
import config
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

batch_size = 112

train_data = pd.read_csv(config.TRAIN_DATAPATH, index_col=[0])[:20000]
for f in train_data.columns:
    train_data[f] = train_data[f].astype(str)
    
train_data = train_data.dropna()
train_data['input_image'] = train_data['input_image'].str.cat(['.png']*len(train_data))
train_data['input_image'] = pd.Series([config.IMAGES_DATAPATH + config.PATH_SEPARATOR]*len(train_data)).str.cat(train_data['input_image'])

train_data['output_image'] = train_data['output_image'].str.cat(['.png']*len(train_data))
train_data['output_image'] = pd.Series([config.IMAGES_DATAPATH + config.PATH_SEPARATOR]*len(train_data)).str.cat(train_data['output_image'])

train_datagen = DataGenerator(list(train_data['input_image']),
                                list(train_data['caption']),
                                list(train_data['output_image']),
                                batch_size=batch_size)

valid_data = pd.read_csv(config.VALID_DATAPATH, index_col=[0])[:5000]
for f in valid_data.columns:
    valid_data[f] = valid_data[f].astype(str)
    
valid_data = valid_data.dropna()
valid_data['input_image'] = valid_data['input_image'].str.cat(['.png']*len(valid_data))
valid_data['input_image'] = pd.Series([config.IMAGES_DATAPATH + config.PATH_SEPARATOR]*len(valid_data)).str.cat(valid_data['input_image'])

valid_data['output_image'] = valid_data['output_image'].str.cat(['.png']*len(valid_data))
valid_data['output_image'] = pd.Series([config.IMAGES_DATAPATH + config.PATH_SEPARATOR]*len(valid_data)).str.cat(valid_data['output_image'])

valid_datagen = DataGenerator(list(valid_data['input_image']),
                                list(valid_data['caption']),
                                list(valid_data['output_image']),
                                batch_size=batch_size,
                                is_training=False)

checkpoint_callback = ModelCheckpoint('checkpoint.hdf5', 
                              save_weights_only=True, 
                              save_best_only=True, 
                              monitor='val_loss')
tensorboard_callback = TensorBoard(log_dir='logs')
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.8, 
                              patience=3, 
                              min_lr=1e-8)
early_stopping_callback = EarlyStopping(monitor='val_loss',
                                       patience=10)
callbacks = [checkpoint_callback, reduce_lr_callback, early_stopping_callback]

model = create_model()

def loss_fn(y_true, y_pred):
    return tf.reduce_sum(y_pred)

loss = {
    'loss': loss_fn,
    'image_decoder': None,
    'image_encoder': None,
}

model.compile(optimizer=optimizers.Adam(0.0005),
            loss=loss)

history = model.fit(
    train_datagen,
    validation_data=valid_datagen,
    epochs=5,
    callbacks = callbacks,
    verbose=1
)

model.save('autoencoder_model.h5')