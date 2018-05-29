import vocab
import json
from PIL import Image
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from titling_model import *
from titling_data import *
from keras.optimizers import Adam

NUM_SUBREDDITS = 20
START_TOKEN = '<START>'
PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
END_TOKEN = '<END>'

embedding_matrix, words_by_id, id_by_words = vocab.load_embedding_matrix()
X_train_imgs, X_train_subreddits, X_train_titles, y_train, train_max_len = get_data('train.json', id_by_words)

print('done loading data!')
print('creating model')

model = create_titling_model(embedding_matrix, num_subreddits=20, max_len=100)
model.compile(optimizer=Adam(lr=3e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print('done model')

X_train_inputs = [
    X_train_imgs[:10],
    X_train_subreddits[:10],
    X_train_titles[:10]
]

print('fitting!')

model.fit(X_train_inputs, y_train[:10], batch_size=5, epochs=10, verbose=2)