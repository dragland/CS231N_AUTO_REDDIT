import vocab
import json
from PIL import Image
import numpy as np
import keras
import json
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

NUM_SUBREDDITS = 20
START_TOKEN = '<START>'
PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
END_TOKEN = '<END>'

class ImageTitlingDataGenerator(keras.utils.Sequence):
    def __init__(self, json_path, ids_by_word, max_len, num_subreddits, batch_size=32):
        with open(json_path) as f:
            data = json.load(f)
            self.posts = data['posts']

        self.batch_size = batch_size
        self.num_subreddits = num_subreddits
        self.ids_by_word = ids_by_word
        self.max_len = max_len

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.posts) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        indices = self.indices[start:end]

        X_imgs = []
        X_subreddits = []
        X_title_indices = []
        y = []

        for i in indices:
            post = self.posts[i]
            img, subreddit, title, target = model_input_output_from_post(post, self.ids_by_word, self.max_len)
            X_imgs.append(img)
            X_subreddits.append(subreddit)
            X_title_indices.append(title)
            y.append(target)

        X_imgs = np.array(X_imgs)
        X_subreddits = np.array(X_subreddits)
        X_title_indices = np.array(X_title_indices)
        y = np.array(y)

        return [X_imgs, X_subreddits, X_title_indices], y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.posts))
        np.random.shuffle(self.indices)

def model_input_output_from_post(post, ids_by_word, max_len):
    path = post['path']
    subreddit = post['subreddit']
    img = np.array(Image.open(path))

    subreddit_one_hot = np.zeros(NUM_SUBREDDITS, dtype=np.float32)
    subreddit_one_hot[subreddit] = 1

    title = post['title']
    title = '{} {} {}'.format(START_TOKEN, title, END_TOKEN)
    title_indices = []
    y = []

    for word in text_to_word_sequence(title):
        if word in ids_by_word:
            word_id = ids_by_word[word]
        else:
            word_id = ids_by_word[UNKNOWN_TOKEN]

        title_indices.append(word_id)

        target = np.zeros(len(ids_by_word))
        target[word_id] = 1
        y.append(target)

    for _ in range(max_len - len(title_indices)):
        title_indices.append(ids_by_word[PAD_TOKEN])
        y.append(np.zeros(len(ids_by_word)))

    title_indices = np.array(title_indices)
    y = np.array(y)

    return img, subreddit_one_hot, title_indices, y

def get_data(json_path, ids_by_word):
    X_imgs = []
    X_subreddits = []
    X_title_indices = []
    y = []

    max_len = 100
    with open(json_path) as f:
        data = json.load(f)
        for post in data['posts']:
            img, subreddit, title, target = model_input_output_from_post(post, ids_by_word, max_len)
            X_imgs.append(img)
            X_subreddits.append(subreddit)
            X_title_indices.append(title)
            y.append(target)

    X_imgs = np.array(X_imgs)
    X_subreddits = np.array(X_subreddits)
    X_title_indices = np.array(X_title_indices)
    y = np.array(y)

    return X_imgs, X_subreddits, X_title_indices, y, max_len
