import argparse
import os
from itertools import product

import json

import numpy as np
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import metrics
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

NUM_CLASSES=20

def get_data(json_path):
    X_train = []
    y_train = []
    with open(json_path) as f:
        data = json.load(f)
        for post in data['posts']:
            path = post['path']
            label = post['subreddit']
            img = np.array(Image.open(path))
            X_train.append(img)

            one_hot = np.zeros(NUM_CLASSES)
            one_hot[label] = 1
            y_train.append(one_hot)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train

def get_subreddit_indices_map(path):
    with open(path) as f:
        data = json.load(f)
        return data['subreddit_indices_map']

def get_subreddit_for_index(index):
    subreddit_indices_map = get_subreddit_indices_map('train.json')
    reverse_map = {}
    for k, v in subreddit_indices_map.items():
        reverse_map[v] = k
    return reverse_map[index]

def create_model():
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

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
        model = create_model()

    # record what params we trained with
    with open(config.experiment_dir + 'config.json', 'w') as f:
        json.dump(vars(config), f)

    model.compile(optimizer=Adam(lr=config.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    X_train, y_train = get_data('train.json')
    X_val, y_val = get_data('validation.json')
    # train the model on the new data for a few epochs
    best_checkpoint_file_path = config.experiment_dir + 'best-checkpoint.hdf5'
    best_checkpoint = ModelCheckpoint(best_checkpoint_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    latest_checkpoint = ModelCheckpoint(latest_checkpoint_path, verbose=1, save_best_only=False, mode='max')
    epoch_saver = EpochSaver(epoch_path)
    tensorboard = TensorBoard(log_dir=config.experiment_dir, histogram_freq=0, write_graph=False, write_images=True)
    model.fit(X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config.batch_size,
        epochs=config.epochs,
        initial_epoch=initial_epoch,
        callbacks=[best_checkpoint, latest_checkpoint, epoch_saver, tensorboard])

def evaluate(config):
    model = create_model()
    checkpoint_file_path = config.experiment_dir + 'best-checkpoint.hdf5'
    model.load_weights(checkpoint_file_path)

    # optimizer doesn't matter
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    X_val, y_val = get_data('validation.json')
    _, acc = model.evaluate(X_val, y_val)

    print('Validation accuracy: {}'.format(acc))

def plot_confusion_matrix(config):
    model = create_model()
    checkpoint_file_path = config.experiment_dir + 'best-checkpoint.hdf5'
    model.load_weights(checkpoint_file_path)

    X_train, y_train = get_data('train.json')
    X_train = X_train
    y_train = np.argmax(y_train, axis=1)
    preds = np.argmax(model.predict(X_train), axis=1)

    indices_map = get_subreddit_indices_map('train.json')
    classes = sorted(indices_map.keys(), key=lambda k: indices_map[k])

    cm = confusion_matrix(y_train, preds)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

def predict(config):
    model = create_model()
    checkpoint_file_path = config.experiment_dir + 'best-checkpoint.hdf5'
    model.load_weights(checkpoint_file_path)

    img_path = config.img_path
    img = np.array(Image.open(img_path))
    label_i = np.argmax(model.predict(np.array([img])), axis=0)[0]
    label = get_subreddit_for_index(label_i)

    print('Predicting {}'.format(label))

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

    if config.mode == 'train':
        print('Training...')
        train(config)
    elif config.mode == 'evaluate':
        print('Evaluating...')
        evaluate(config)
    elif config.mode == 'plot_cm':
        print('Plotting confusion matrix')
        plot_confusion_matrix(config)
    elif config.mode == 'predict':
        print('Making predicting for image')
        predict(config)
    else:
        print('Invalid mode! Aborting...')

