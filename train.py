from src.models.autoencoder import create_model
from src.datagen import DataGenerator
import pandas as pd
import config
from tensorflow.keras import optimizers, losses

train_data = pd.read_csv(config.TRAIN_DATAPATH, index_col=[0])

train_data['input_image'] = train_data['input_image'].str.cat(['.png']*len(train_data))
train_data['input_image'] = pd.Series([config.IMAGES_DATAPATH + '\\']*len(train_data)).str.cat(train_data['input_image'])

train_data['output_image'] = train_data['output_image'].str.cat(['.png']*len(train_data))
train_data['output_image'] = pd.Series([config.IMAGES_DATAPATH + '\\']*len(train_data)).str.cat(train_data['output_image'])

train_datagen = DataGenerator(list(train_data['input_image']),
                                list(train_data['caption']),
                                list(train_data['output_image']),
                                batch_size=2)

# print(train_datagen[0])

model = create_model()
model.summary()
def loss_fn(y_true, y_pred):
    return y_pred
model.compile(optimizer=optimizers.Adam(0.005),
            # metrics = ['accuracy'],
            loss=loss_fn)

# print(model(train_datagen[0], training=True))

history = model.fit(
    train_datagen,
    epochs=5,
    verbose=1
)