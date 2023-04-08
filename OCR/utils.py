import os, cv2, json, time, shutil, requests
import numpy as np
import unicodedata, re
from transformerTranslation.preprocessing import DatasetLoader
import tensorflow as tf
import pandas as pd
from transformerTranslation.Transformer import *
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Layer, Embedding, LayerNormalization

import numpy as np
import streamlit as st
from urllib.request import urlretrieve
from OCR.crnn import CRNN
from OCR.dbnet import DBNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
checkpoint_path = "transformerTranslation/saved_models/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print('checkpoint_dir: ',checkpoint_dir)
@st.cache
def download_assets():
    if not os.path.exists('OCR/assets.zip'):
        urlretrieve('https://nomnaftp.000webhostapp.com/assets.zip', 'assets.zip')
    if not os.path.exists('assets'):
        shutil.unpack_archive('OCR/assets.zip', 'assets')


@st.cache(hash_funcs={DBNet: lambda _: None, CRNN: lambda _: None})
def load_models():
    det_model = DBNet()
    reg_model = CRNN()
    det_model.model.load_weights('./assets/DBNet.h5')
    reg_model.model.load_weights('./assets/CRNN.h5')
    return det_model, reg_model


def order_points_clockwise(box_points):
    points = np.array(box_points)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    quad_box = np.zeros((4, 2), dtype=np.float32)
    quad_box[0] = points[np.argmin(s)]
    quad_box[2] = points[np.argmax(s)]
    quad_box[1] = points[np.argmin(diff)]
    quad_box[3] = points[np.argmax(diff)]
    return quad_box


def get_patch(page, points):
    points = order_points_clockwise(points)
    page_crop_width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]))
    )
    page_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]))
    )
    pts_std = np.float32([
        [0, 0], [page_crop_width, 0], 
        [page_crop_width, page_crop_height],[0, page_crop_height]
    ])
    M = cv2.getPerspectiveTransform(points, pts_std)
    return cv2.warpPerspective(
        page, M, (page_crop_width, page_crop_height), 
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )

def translate(model, source_sentence, target_sentence_start=[['<sos>']]):
    if np.ndim(source_sentence) == 1: # Create a batch of 1 the input is a sentence
        source_sentence = [source_sentence]
    if np.ndim(target_sentence_start) == 1:
        target_sentence_start = [target_sentence_start]
    # Tokenizing and padding
    source_seq = tokenize_inp.texts_to_sequences(source_sentence)
    source_seq = tf.keras.preprocessing.sequence.pad_sequences(source_seq, padding='post', maxlen=30)
    predict_seq = tokenize_tar.texts_to_sequences(target_sentence_start)
    
    predict_sentence = list(target_sentence_start[0]) # Deep copy here to prevent updates on target_sentence_start
    while predict_sentence[-1] != '<eos>' and len(predict_seq) < max_token_length:
        predict_output = model([np.array(source_seq), np.array(predict_seq)], training=None)
        predict_label = tf.argmax(predict_output, axis=-1) # Pick the label with highest softmax score
        predict_seq = tf.concat([predict_seq, predict_label], axis=-1) # Updating the prediction sequence
        predict_sentence.append(tokenize_tar.index_word[predict_label[0][0].numpy()])
    return predict_sentence

def get_phonetics(text):
    # def is_nom_text(result):
    #     for phonetics_dict in result:
    #         if phonetics_dict['t'] == 3 and len(phonetics_dict['o']) <= 0: 
    #             return True
    #     return False
        
    url = 'https://hvdic.thivien.net/transcript-query.json.php'
    headers = { 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8' }
    # Request phonetics for Hán Việt (lang=1) first, if the response result is not
    # Hán Việt (contains blank lists) => Request phonetics for Nôm (lang=3)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model=Transformer()
    print(latest)
    model.load_weights(latest).expect_partial()
    print(' '.join(translate(model,' '.join(list(text)).split(' '))))
    result = ' '.join(translate(model,' '.join(list(text)).split(' ')))

    # for lang in [1, 3]: 
    #     payload = f'mode=trans&lang={lang}&input={text}'
    #     response = requests.request('POST', url, headers=headers, data=payload.encode())
    #     print('here')
    #     result = json.loads(response.text)['result']
    #     if not is_nom_text(result): break
    #     time.sleep(0.1)     
    return result