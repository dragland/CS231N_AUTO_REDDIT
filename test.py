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

embedding_matrix, words_by_id, id_by_words = vocab.load_limited_embedding_matrix('train.json')
data_generator = ImageTitlingDataGenerator('train.json', id_by_words, max_len=30, num_subreddits=NUM_SUBREDDITS, batch_size=8)

print('done loading data!')
print('creating model')

model = ImageTitlingModel(embedding_matrix, words_by_id, id_by_words, num_subreddits=20, max_len=30)
model.train_model.compile(optimizer=Adam(lr=3e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print('done model')
print('fitting!')

model.train_model.fit_generator(data_generator, epochs=10, max_queue_size=1, verbose=2)
# img = np.array(Image.open('datasets/AccidentalRenaissance0.jpg'))
# subreddit = 0
# title = model.generate_title(img, subreddit)
print(title)