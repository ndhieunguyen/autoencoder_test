from transformers import AutoConfig, TFAutoModel, AutoTokenizer
from tensorflow.keras import layers

import tensorflow as tf

def convert_tfmodel(dir_pytorch_model, dir_config):
    config = AutoConfig.from_pretrained(dir_config) 
    tf_model = TFAutoModel.from_pretrained(dir_pytorch_model, config=config, from_pt=True)
    tf_model.layers[0].trainable = False

    return tf_model

def get_tokenizer(pretrained="distilroberta-base"):
    return AutoTokenizer.from_pretrained(pretrained)

def get_tokens(list_text, pretrained="distilroberta-base"):
    tokenizer = get_tokenizer(pretrained)

    output_ids, output_mask = [], []
    for text in list_text:
        token = tokenizer.encode_plus(text, padding='max_length', max_length=30, 
                                    add_special_tokens=True, truncation=True,
                                    return_tensors="tf")

        output_ids.append(token['input_ids'][0])
        output_mask.append(token['attention_mask'][0])

    input_bert = (tf.convert_to_tensor(output_ids), 
                    tf.convert_to_tensor(output_mask))

    return input_bert

def build_model(bert_model):
    input_ids = layers.Input(shape=(None, ), dtype=tf.int64, name='input_ids')
    input_mask = layers.Input(shape=(None, ), dtype=tf.int64, name='attention_mask')

    embds = bert_model(input_ids, input_mask)[1]

    dense = layers.Dense(512, activation="relu")(embds)
    dropout = layers.Dropout(0.1)(dense)
    dense = layers.Dense(128)(dropout)

    model = tf.keras.models.Model(inputs=[input_ids, input_mask], outputs= dense)

    return model

def get_text_encoder_model_tf(dir_pytorch_model='distilroberta-base', dir_config='distilroberta-base'):
    bert_model = convert_tfmodel(dir_pytorch_model, dir_config)
    model = build_model(bert_model)
    
    for layer in model.layers:
        layer.trainable = False
    return model