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
data_generator = ImageTitlingDataGenerator('train.json', id_by_words, max_len=100, num_subreddits=NUM_SUBREDDITS, batch_size=32)

print('done loading data!')
print('creating model')

model = create_titling_model(embedding_matrix, num_subreddits=20, max_len=100)
model.compile(optimizer=Adam(lr=3e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print('done model')
print('fitting!')

model.fit_generator(data_generator, epochs=10, verbose=2)