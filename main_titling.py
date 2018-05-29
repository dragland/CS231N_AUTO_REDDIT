import argparse
import os

import json
from PIL import Image

from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence

from titling_model import ImageTitlingModel

from titling_data import *

NUM_SUBREDDITS = 20
START_TOKEN = '<START>'
PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
END_TOKEN = '<END>'

class EpochSaver(Callback):
    def __init__(self, path):
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        with open(self.path, 'w') as f:
            json.dump({'epoch': epoch}, f)

def train(config):
    latest_checkpoint_path = config.experiment_dir + 'latest-checkpoint.h5'
    epoch_path = config.experiment_dir + 'last_epoch.json'
    initial_epoch = 0
    if os.path.exists(latest_checkpoint_path):
        model = load_model(latest_checkpoint_path)
        with open(epoch_path) as f:
            initial_epoch = json.load(f)['epoch'] + 1
            print('Loading model from last checkpoint and resuming training on epoch {}'.format(initial_epoch))
    else:
        print('Starting new training run')
        model = ImageTitlingModel()

    # record what params we trained with
    with open(config.experiment_dir + 'config.json', 'w') as f:
        json.dump(vars(config), f)


    X_train_imgs, X_train_subreddits, X_train_titles, y_train, train_max_len = get_data('train.json')
    X_val_imgs, X_val_subreddits, x_val_titles, y_val, val_max_len = get_data('validation.json')

    model = ImageTitlingModel(word_embeddings_size=300, num_subreddits=20, max_len=train_max_len)
    model.compile(optimizer=Adam(lr=config.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    best_checkpoint_file_path = config.experiment_dir + 'best-checkpoint.hdf5'
    best_checkpoint = ModelCheckpoint(best_checkpoint_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    latest_checkpoint = ModelCheckpoint(latest_checkpoint_path, verbose=1, save_best_only=False, mode='max')
    epoch_saver = EpochSaver(epoch_path)
    tensorboard = TensorBoard(log_dir=config.experiment_dir, histogram_freq=0, write_graph=False, write_images=True)
    X_train_inputs = {
        model.images_input_key: X_train_imgs,
        model.subreddits_input_key: X_train_subreddits,
        model.titles_input_key: X_train_titles
    }
    X_val_inputs = {
        model.images_input_key: X_val_imgs,
        model.subreddits_input_key: X_val_subreddits,
        model.titles_input_key: X_val_titles
    }
    model.fit(X_inputs, y_train,
        validation_data=(X_val_inputs, y_val),
        batch_size=config.batch_size,
        epochs=config.epochs,
        initial_epoch=initial_epoch,
        callbacks=[best_checkpoint, latest_checkpoint, epoch_saver, tensorboard])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, help='unique experiment name')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--mode', type=str, help='train, evaluate')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, help='number of epochs to train for')
    parser.add_argument('--img_path', type=str, help='path of img to predict')

    config = parser.parse_args()

    experiment_dir = 'experiments/{}/'.format(config.experiment)
    config.experiment_dir = experiment_dir
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    mode_handlers = {
        'train': train
    }

    mode = config.mode
    if mode in mode_handlers:
        handler = mode_handlers[mode]
        handler(config)
    else:
        print('Invalid mode! Aborting...')
