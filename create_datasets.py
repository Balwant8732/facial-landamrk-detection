import os
import random
import numpy as np
import json
import time
from tqdm import tqdm
import tensorflow as tf

def get_image_list(data_dir):
    image_list = []
    for pt in os.listdir(data_dir):
        if pt.endswith('.jpg'):
            image_list.append(os.path.join(data_dir,pt))
    return image_list

def get_ttv():
    data_dir = '/home/dilip/Code/ml/data_48_128'
    total_images = get_image_list(data_dir)
    random.shuffle(total_images)
    train_list = total_images[:200000]
    test_list = total_images[200000:210000]
    val_list = total_images[210000:]
    return train_list, test_list, val_list

def get_label_list(data_list):
    label_list = []
    for i in tqdm(data_list):
        json_file = i[:-4] + '.json'
        with open(json_file) as fid:
            marks = json.load(fid)
        label_list.append(list(marks))
    return label_list

def save_npy(data_list, parsed_data_list, filename):
    label_list = get_label_list(data_list)
    x = np.array(parsed_data_list)
    y = np.array(label_list)
    with open(filename, 'wb') as f:
        np.save(f, x)
        np.save(f, y)

def _parse(image_name, label):
    img = tf.io.read_file(image_name)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, label

def _parse_x(data_list):
    ll = []
    for image_name in tqdm(data_list):
        img = tf.io.read_file(image_name)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        ll. append(img)
    return ll



def get_dataset(npy_file, batch_size, shuffle=True):
    with open(npy_file, 'rb') as f:
        x = np.load(f, allow_pickle=True)
        y = np.load(f, allow_pickle=True)
    images = tf.constant(x)
    labels = tf.constant(y)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.map(_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


if __name__ == '__main__':
    train, test, val = get_ttv()
    parsed_train = _parse_x(train)
    parsed_test = _parse_x(test)
    parsed_val = _parse_x(val)
    save_npy(train, parsed_train, 'new_npys/train.npy')
    save_npy(test, parsed_test, 'new_npys/test.npy')
    save_npy(val, parsed_val, 'new_npys/val.npy')
    # with open('new_npys/train.npy', 'rb') as f:
    #     x = np.load(f, allow_pickle=True)
    #     y = np.load(f, allow_pickle=True)
    # print(x.shape)
    # print(y.shape)

    