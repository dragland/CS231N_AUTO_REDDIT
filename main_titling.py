import argparse
import os

from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import numpy as np
from keras.optimizers import Adam

from titling_model import *
from titling_data import *
from vocab import *

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
    # record what params we trained with
    with open(config.experiment_dir + 'config.json', 'w') as f:
        json.dump(vars(config), f)

    max_len = 30
    embedding_matrix, words_by_id, id_by_words = vocab.load_embedding_matrix()

    latest_checkpoint_path = config.experiment_dir + 'latest-checkpoint.h5'
    epoch_path = config.experiment_dir + 'last_epoch.json'
    initial_epoch = 0
    model = ImageTitlingModel(embedding_matrix, words_by_id, id_by_words, num_subreddits=NUM_SUBREDDITS, max_len=max_len)
    if os.path.exists(latest_checkpoint_path):
        mode.load_checkpoint(latest_checkpoint_path)
        with open(epoch_path) as f:
            initial_epoch = json.load(f)['epoch'] + 1
            print('Loading model from last checkpoint and resuming training on epoch {}'.format(initial_epoch))
    else:
        print('Starting new training run')
        model.train_model.compile(optimizer=Adam(lr=config.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    train_data_generator = ImageTitlingDataGenerator('train.json',
        id_by_words,
        max_len=max_len,
        num_subreddits=NUM_SUBREDDITS,
        batch_size=config.batch_size)
    validation_data_generator = ImageTitlingDataGenerator('validation.json',
        id_by_words,
        max_len=max_len,
        num_subreddits=NUM_SUBREDDITS,
        batch_size=config.batch_size)

    # train the model on the new data for a few epochs
    best_checkpoint_file_path = config.experiment_dir + 'best-checkpoint.hdf5'
    best_checkpoint = ModelCheckpoint(best_checkpoint_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    latest_checkpoint = ModelCheckpoint(latest_checkpoint_path, verbose=1, save_best_only=False, mode='max')
    epoch_saver = EpochSaver(epoch_path)
    tensorboard = TensorBoard(log_dir=config.experiment_dir, histogram_freq=0, write_graph=False, write_images=True)
    model.train_model.fit_generator(train_data_generator,
        validation_data=validation_data_generator,
        max_queue_size=1,
        epochs=config.epochs,
        initial_epoch=initial_epoch,
        callbacks=[best_checkpoint, latest_checkpoint, epoch_saver, tensorboard])

def sample_inference(config):
    max_len = 30
    embedding_matrix, words_by_id, id_by_words = vocab.load_embedding_matrix()
    model = ImageTitlingModel(embedding_matrix, words_by_id, id_by_words, num_subreddits=NUM_SUBREDDITS, max_len=max_len)
    #checkpoint_file_path = config.experiment_dir + 'best-checkpoint.hdf5'
    #model.load_weights(checkpoint_file_path)

    with open('train.json') as f:
        data = json.load(f)
        NUM_SAMPLES = 5
        posts = data['posts']
        indices = np.random.choice(len(posts), NUM_SAMPLES)
        for i in indices:
            post = posts[i]
            img = np.array(Image.open(post['path']))
            subreddit = post['subreddit']
            actual_title = post['title']
            predicted_title = model.generate_title(img, subreddit)
            print('Ground truth: ', actual_title)
            print('Predicted: ', predicted_title)
            print()

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
        'train': train,
        'sample_inference': sample_inference,
    }

    mode = config.mode
    if mode in mode_handlers:
        handler = mode_handlers[mode]
        handler(config)
    else:
        print('Invalid mode! Aborting...')
